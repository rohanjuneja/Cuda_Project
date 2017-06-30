// ----
// ----  Computes the potential field for a volume
// ----  Input: volume file, dimensions: X, Y, Z, output file name
// ----	 Output: normalized potential field:
//		 1 vector for each point in the volume
//
// Last change: Thu May 15 15:20:38 EDT 2003 by Nicu D. Cornea
//
//

// #define TRACE


#include "potVect.h"
#include <thrust/sort.h>

#define BOUND_SIZE	7000

struct compareStruct {
	__host__ __device__
	bool operator()(VoxelPosition a, VoxelPosition b) {
		if(a.z != b.z)
			return a.z < b.z;
		else if(a.y != b.y)
			return a.y < b.y;
		else
			return a.x < b.x;
	}
};
 __constant__ VoxelPosition d_bound[BOUND_SIZE];

bool SortBoundaryArray(int numBound, VoxelPosition Bound[]);


/*
this kernel normalizes the force field by calculating the multiplication of magnitudes of force in
x,y,z direction and dividing it by the final multiplication.
*/
__global__ void normalize_vector(Vector* force,unsigned char* f, bool inOut,int slsz,int L)
{
	int k=blockIdx.x;
	int j=threadIdx.x;
	int i=blockIdx.y;
	int idx=k*slsz + j*L + i;
	if(!inOut) {
    // only for interior voxels we had calculated forces
    if(f[idx] == EXTERIOR) return;
  }
  
  float r = force[idx].xd*force[idx].xd + 
      force[idx].yd*force[idx].yd + 
      force[idx].zd*force[idx].zd;
    
  if(r > 0.00) {
    r = sqrtf(r);
    
    force[idx].xd = force[idx].xd / r;
    force[idx].yd = force[idx].yd / r;
    force[idx].zd = force[idx].zd / r;
  }
	

}
/*
This kernel computes potential field at every inside voxel using 
boundary voxel as point charges
*/

__global__ void compute_potential_field(Vector* force,int numBound,unsigned char* f,bool inOut,int slsz,int sz,int L, int fieldStrenght)
{
	int k=blockIdx.x;
	int j=threadIdx.x;
	int i=blockIdx.y;

	int zStartIndex = 0;
	int zEndIndex = numBound- 1;
	int s;
	// if((k*slsz + j*L + i)==0)
	// printf("parent kernel %.6f %.6f %.6f\n",d_bound[4].x,d_bound[4].y,d_bound[4].z );
	for (s = 0; s < numBound; s++) {
		if((k - d_bound[s].z) <= PF_THRESHOLD) {
			zStartIndex = s;
			break;
		}
	}
	for (s = numBound-1; s >= zStartIndex; s--) {
		if((d_bound[s].z - k) <= PF_THRESHOLD) {
			zEndIndex = s;
			break;
		}
	}

	int yStartIndex = zStartIndex;
	int yEndIndex = zEndIndex;
		for (s = zStartIndex; s <= zEndIndex; s++) {
			if((j - d_bound[s].y) <= PF_THRESHOLD) {
				yStartIndex = s;
				break;
			}
		}
		for (s = zEndIndex; s >= yStartIndex; s--) {
			if((d_bound[s].y - j) <= PF_THRESHOLD) {
				yEndIndex = s;
				break;
			}
		}

	int idx=k*slsz + j*L + i;
	force[idx].xd = 0.00;
	force[idx].yd = 0.00;
	force[idx].zd = 0.00;

	
	if(!inOut) {
			  if(f[idx] == 0) {
			    return;
			  }
			}
	

	if(f[idx] == SURF) return;
	if(f[idx] == BOUNDARY) return;

	int startIndex = yStartIndex;
	int endIndex = yEndIndex;
	for (s = yStartIndex; s <= yEndIndex; s++) {
		if((i - d_bound[s].x) <= PF_THRESHOLD) {
			startIndex = s;
			break;
				}
			}
	for (s = yEndIndex; s >= startIndex; s--) {
		if((d_bound[s].x - i) <= PF_THRESHOLD) {
			endIndex = s;
			break;
		}
	}

	if(endIndex < startIndex) {
				
				startIndex = 0;
				endIndex = numBound - 1;
	}

	for (s = startIndex; s <= endIndex; s++) {
				
				float v1 = i - d_bound[s].x;
				float v2 = j - d_bound[s].y;
				float v3 = k - d_bound[s].z;
				float r, t;
#ifdef EUCLIDEAN_METRIC
				// euclidean metric
				r = sqrtf(v1*v1 + v2*v2 + v3*v3);
#else
				// simpler metric
				r = abs(v1) + abs(v2) + abs(v3);
#endif

			
				if(r != 0.00) {
				
				  
				  t = 1.00;
				  for(int p = 0; p <= fieldStrenght; p++) {
				    t = t * r;
				  }
				  r = t;
				  

				  force[idx].xd+=(v1/r);
				  force[idx].yd+=(v2/r);
				  force[idx].zd+=(v3/r);
				}
			}
			
}

/*
This kernel computes potential field at every boundary voxel using 
the neighbours stored in ng-array(shared memory)
*/
__global__ void computePotentialFieldForBoundaryVoxels(unsigned char* f, Vector* force, int slsz, bool inOut, int L) {
	
	int k=blockIdx.x+1;
	int j=threadIdx.x+1;
	int i=blockIdx.y+1;
	__shared__ int ng[26];
	if(threadIdx.x==0)
	{
		 // face neighbors
			    ng[0]	= + slsz + 0 + 0;
			    ng[1]	= - slsz + 0 + 0;
			    ng[2]	= +    0 + L + 0;
			    ng[3]	= +    0 - L + 0;
			    ng[4]	= +    0 + 0 + 1;
			    ng[5]	= +    0 + 0 - 1;
			    // v-neighbors
			    ng[6]	= - slsz - L - 1;
			    ng[7]	= - slsz - L + 1;
			    ng[8]	= - slsz + L - 1;
			    ng[9]	= - slsz + L + 1;
			    ng[10]	= + slsz - L - 1;
			    ng[11]	= + slsz - L + 1;
			    ng[12]	= + slsz + L - 1;
			    ng[13]	= + slsz + L + 1;
			    // e-neighbors
			    ng[14]	= + slsz + L + 0;
			    ng[15]	= + slsz - L + 0;
			    ng[16]	= - slsz + L + 0;
			    ng[17]	= - slsz - L + 0;
			    ng[18]	= + slsz + 0 + 1;
			    ng[19]	= + slsz + 0 - 1;
			    ng[20]	= - slsz + 0 + 1;
			    ng[21]	= - slsz + 0 - 1;
			    ng[22]	= +    0 + L + 1;
			    ng[23]	= +    0 + L - 1;
			    ng[24]	= +    0 - L + 1;
			    ng[25]	= +    0 - L - 1;
	}
	__syncthreads();	
	long idx = k*slsz + j*L + i;
	  
	if((f[idx] == SURF) ||
	 (f[idx] == BOUNDARY))
	{
	  force[idx].xd = 0.00;
	  force[idx].yd = 0.00;
	  force[idx].zd = 0.00;
	  float var_xd=0.00;
	  float var_yd=0.00;
	  float var_zd=0.00;
	  // look at the neighbors and average the forces if not 0
	  //
	int v1 = 0;
	for(int s=0; s < 26; s++) {

	long iidx = idx + ng[s];		
	if(f[iidx] == SURF)		continue;
	if(f[iidx] == BOUNDARY)	continue;

	// if we know the interior of the object, take only interior
	// neighbors
	if(!inOut) {
	  if(f[iidx] == EXTERIOR)	continue;
	}

	var_xd = var_xd + force[iidx].xd;
	var_yd = var_yd + force[iidx].yd;
	var_zd = var_zd + force[iidx].zd;
	v1 = v1 + 1;

	  }
	  
	  // average
	  if(v1 != 0) {
	var_xd = var_xd / (double) v1;
	var_yd= var_yd / (double) v1;
	var_zd = var_zd / (double) v1;
	  }
	  else {
	printf("Boundary voxel has no interior neighbor !!! - Force = 0\n");
	  }
	  
	  // normalize
	float r = var_xd*var_xd + 
	var_yd*var_yd + 
	var_zd*var_zd;
	  
	  if(r > 0.00) {
	r = sqrtf(r);

	force[idx].xd = var_xd / r;
	force[idx].yd = var_yd / r;
	force[idx].zd = var_zd/ r;	
	  }
	}
}



bool CalculatePotentialField(
	int L, int M, int N, 	      // [in] size of volume
	unsigned char* f, 	      // [in] volume flags
	int fieldStrenght,	      // [in] potential field strenght
	Vector* force,		      // [out] force field	
	bool inOut                    // [in] flag indicating that we don't 
	                              //    know what the inside/outside of 
	                              //    the object is. We have only point 
	                              //    samples of the boundary.
	                              //  DEFAULT: false (only interior)
) {
	//cudaSetDevice(1);
  int Lm1, Mm1, Nm1;
  int i,j,k, s, p;
  long idx, iidx, slsz, sz;
  VoxelPosition* Bound;
  int numBound = 0;
  bool flagSurf, flagBound;
  double r, t;
  int v1, v2, v3;
  int startIndex, tmpStartIndex, endIndex, tmpEndIndex, zStartIndex, zEndIndex, yStartIndex, yEndIndex;

  //
  // check volume padding - fast version
  //
  if(!CheckVolumePadding(f, L, M, N)) {
    printf("** Error - Object touches bounding box. Abort.\n");
    exit(1);
  }


#ifdef _DEBUG
	printf("\t************ Potential Field calculation parameters: ******************\n");
#ifdef HALF_BOUNDARY_POINTS
	printf("\t** Using only HALF of the boundary points.\n");
#else
	printf("\t** Using ALL boundary points.\n");
#endif

#ifdef EUCLIDEAN_METRIC
	printf("\t** Using EUCLIDEAN metric.\n");
#else
	printf("\t** Using NON EUCLIDEAN metric.\n");
#endif	
	if(inOut) {
	  printf("\t** Inside and Outside.\n");
	}
	else {
	  printf("\t** Inside ONLY.\n");
	}
	printf("\t********* Potential Field calculation parameters - end ****************\n");

#endif

  if((Bound = new VoxelPosition[BOUND_SIZE]) == NULL) {
	printf("\nERROR allocating memory for boundary array! - Abort\n");
	exit(1);
  }

  Lm1 = L - 1;
  Mm1 = M - 1;
  Nm1 = N - 1;
  slsz = L*M;		// slice size
  sz = slsz*N;

  // save all the boundary voxels in array Bound[]
	for (k = 1; k < Nm1; k++) {
		for (j = 1; j < Mm1; j++) {
			for (i = 1; i < Lm1; i++) {
				flagSurf = false;
				flagBound = true;
				idx = k*slsz + j*L + i;

				// CASE 1: treat the inner layer
				if (f[idx] == 0) continue;

				//consider six face neighbors, if anyone is zero, it is a boundary voxel
				iidx = k*slsz + j*L + i-1;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select this one as part of the boundary.
						flagBound = false;
					}

				}
#endif

				if(!flagSurf || flagBound) {

				iidx = k*slsz + j*L + i+1;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select it as part of the boundary.
						flagBound = false;
					}
				}
#endif

				if(!flagSurf || flagBound) {

				iidx = k*slsz + (j-1)*L + i;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select it as part of the boundary.
						flagBound = false;
					}
				}
#endif

				if(!flagSurf || flagBound) {

				iidx = k*slsz + (j+1)*L + i;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select it as part of the boundary.
						flagBound = false;
					}
				}
#endif

				if(!flagSurf || flagBound) {

				iidx = (k-1)*slsz + j*L + i;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select it as part of the boundary.
						flagBound = false;
					}
				}
#endif

				if(!flagSurf || flagBound) {

				iidx = (k+1)*slsz + j*L + i;
				if (f[iidx] == 0) {
					flagSurf = true;
				}
#ifdef HALF_BOUNDARY_POINTS
				// consider only half of the boundary points
				else {
					if (f[iidx] == BOUNDARY) {
						// a neighbour of the point was already selected so we will not select it as part of the boundary.
						flagBound = false;
					}
				}
#endif

				}
				}
				}
				}
				}

				// restore idx to the right value
				idx = k*slsz + j*L + i;
				if (flagSurf) {
					f[idx] = SURF;

					if(flagBound) {
							// if no neighbour of this voxel is already marked as boundary, then mark this one.
							// or if we are taking all the boundary voxels 
							// 	(in this case flagBound stays true)
						f[idx] = BOUNDARY;
						Bound[numBound].x = i;
						Bound[numBound].y = j;
						Bound[numBound].z = k;
						numBound++;
						if(numBound >= BOUND_SIZE) {
							printf("ERROR: too many boundary points detected !! - Abort.\n");
							exit(1);
						}
					}
				}
			}
		}
	}

	//printf("numBound = %d \n", numBound);

#ifdef _DEBUG
	PrintElapsedTime("\tPF-1: finding the boundary voxels.");
	printf("\t--Found %d boundary voxels.\n", numBound);
#endif


// sort the boundary array.
SortBoundaryArray(numBound, Bound);

#ifdef _DEBUG
	PrintElapsedTime("\tPF-2: sorting the boundary voxels.");
#ifdef TRACE
	// print the boundary voxels
	for(i=0; i < numBound; i++) {
		printf("%d %d %d 0.5\n", Bound[i].x, Bound[i].y, Bound[i].z);
	}
	exit(1);
#endif	

#endif


// Compute the potential field
	printf("Computing potential field.\n");
	dim3 dimBlock(M,1);
	dim3 dimGrid(N,L);
	
	unsigned char* d_f;
	Vector* d_force;
	//VoxelPosition *d_bound;
	cudaMalloc((void **)&d_f,sizeof(unsigned char)*L*M*N);
	cudaMalloc((void **)&d_bound,sizeof(VoxelPosition)*BOUND_SIZE);
	cudaMalloc((void **)&d_force,sizeof(Vector)*L*M*N);
	cudaMemcpy(d_f,f,sizeof(unsigned char)*L*M*N,cudaMemcpyHostToDevice);
	//cudaMemcpy(d_bound,Bound,sizeof(VoxelPosition)*BOUND_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_bound,Bound, numBound* sizeof(VoxelPosition),cudaMemcpyHostToDevice);
	cudaMemcpy(d_force,force,sizeof(Vector)*L*M*N,cudaMemcpyHostToDevice);
	compute_potential_field<<<dimGrid,dimBlock>>>(d_force,numBound,d_f,inOut,slsz,sz,L, fieldStrenght);
	
	normalize_vector<<<dimGrid,dimBlock>>>(d_force,d_f,inOut,slsz,L);


delete [] Bound;


#ifdef _DEBUG
	PrintElapsedTime("\tPF-3: computing potential field for inside voxels.");
#endif

#ifdef _DEBUG
  PrintElapsedTime("\tPF-4: normalizing force vectors for inside voxels.");
#endif

  if (!inOut) {
    //neighbors:
    

    dim3 dimBlock(Mm1-1,1);
	dim3 dimGrid(Nm1-1,Lm1-1);
    

	computePotentialFieldForBoundaryVoxels<<<dimGrid,dimBlock>>>(d_f, d_force, slsz, inOut, L);


    
  }
  cudaMemcpy(force,d_force,sizeof(Vector)*L*M*N,cudaMemcpyDeviceToHost);
  

#ifdef _DEBUG
	PrintElapsedTime("\tPF-5: computing potential field for boundary voxels.");
#endif


  return true;
}




compareStruct comp;

bool SortBoundaryArray(int numBound, VoxelPosition Bound[]) {
	thrust::sort(Bound, Bound+numBound, comp);
	return true;
}







