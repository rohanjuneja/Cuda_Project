// Find high divergence points of a vector field
// --- Input: 1. normalized 3D vector field
//
//               dFx     dFy     dFz
// divergence = ----- + ----- + -----
//               dx      dy      dz
//
// --- Output: highest ...% divergence point list
// --- Author: Nicu D. Cornea, Vizlab, Rutgers University
// --- Date: Wed Aug 20 17:53:56 EDT 2003
//

#include "HighDiverg.h"
#include <thrust/copy.h>


// #define TRACE

#define SEARCH_GRID		1
#define CELL_SIZE		1.00 / SEARCH_GRID

#define MAX_NUM_HDPTS	5000

/*
this structure will be used in the buffer for storing the divergence values
*/
typedef struct Divergence_vals
{
  /* data */
  bool flag;
  double val;
  Divergence_vals()
  {
    flag=false;
  }

}Divergence_vals;
      
struct find_div_pts
{
  __host__ __device__
  bool operator()(const Divergence_vals x)
  {
    return x.flag;
  }
};



typedef struct {
	int* Points;
	int numPoints;
} HDGroup;
// Lock mylock=new Lock();

inline bool PointIsCloseToGroup(int pt, int grp, HDGroup *Groups, VoxelPositionDouble **HDPts);

__host__ __device__ Vector interpolation(double x, double y, double z, int sizx, int sizy, int sizz, Vector *forcevec);
/*
this kernel calculates the divergence value at each voxel with the help of force field
and saves in the custom structure buffer 
*/
__global__ void max_min_divergence(unsigned char *flags,Vector *ForceField,double *maxDiv,double *minDiv,bool inOut,int slsz,int L,int M, int N, double *vdist,Divergence_vals *d_arr)
{
  int k=blockIdx.x+1;
  int j=blockIdx.y+1;
  int i=threadIdx.x+1;
  double div;
  int idx=k*slsz + j*L +i;
  d_arr[idx].flag = false;

  if(!inOut) {
    // - if this point is EXTERIOR, BOUNDARY or SURF, skip it
    if( (flags[idx] == EXTERIOR) ||
    (flags[idx] == BOUNDARY) ||
    (flags[idx] == SURF))
    {
      return;
    }
  }
  else {
    // we look for high divergence points outside the object too
    // ignore only boundary points.
    if( (flags[idx] == BOUNDARY) ||
        (flags[idx] == SURF))
    {
      return;
    }
  }

  double x, y, z;
  for(int kk=0; kk < SEARCH_GRID; kk++) {
    for(int jj=0; jj < SEARCH_GRID; jj++) {
      for(int ii=0; ii < SEARCH_GRID; ii++) {
        x = i + (ii * CELL_SIZE);
        y = j + (jj * CELL_SIZE);
        z = k + (kk * CELL_SIZE);
#ifdef TRACE
        //              printf("At point: (%lf, %lf, %lf)\n", x, y, z);
#endif              
        // interpolate force vectors arround the point
        
        Vector v_0 = interpolation(x + *vdist, y, z, L, M, N, ForceField);
        Vector v_1 = interpolation(x - *vdist, y, z, L, M, N, ForceField);
        Vector v_2 = interpolation(x, y + *vdist, z, L, M, N, ForceField);
        Vector v_3 = interpolation(x, y - *vdist, z, L, M, N, ForceField);
        Vector v_4 = interpolation(x, y, z + *vdist, L, M, N, ForceField);
        Vector v_5 = interpolation(x, y, z - *vdist, L, M, N, ForceField);
        
        div = ((v_0.xd - v_1.xd) + (v_2.yd - v_3.yd) + (v_4.zd - v_5.zd)) / (2 * *vdist);
        d_arr[idx].flag=true;
        d_arr[idx].val=div;
      }
    }
  }



}

// double GetDiv(double x, double y, double z);

bool GetHighDivergencePoints(
	Vector* ForceField, 	      // [in] vector field
	int L, int M, int N,	      // [in] size of vector field (X, Y and Z)
	unsigned char *flags,	      // [in] flags array
	float perc,		      // [in] percentage of high div. points 
	                              //         to be returned (top <perc> %)
	VoxelPositionDouble **HDPts,  // [out] high divergence point list
	int *numHDPts,		      // [out] number of points in the list
	bool inOut                    // [in] flag specifying if we should look
	                              //    outside the object too (if true).
	                              // DEFAULT: false
) {

#ifdef TRACE
  printf("TRACE: Starting GetHighDivergencePoints function. Cellsize = %lf\n", CELL_SIZE);
#endif

  (*HDPts) = NULL;
  (*numHDPts) = 0;

  if(perc == 0) {
    return true;
  }
  
  long idx, slsz;
  int i,j,k, ii, jj, kk, s;
  double x, y, z;
  long cntz, cntnz;
  
  slsz = L*M;		// slice size
  double adiv[MAX_NUM_HDPTS];	// divergence array
  
    
  if(((*HDPts) = new VoxelPositionDouble[MAX_NUM_HDPTS]) == NULL) {
    printf("GetHighDivergencePoints: UPS! - Error allocating memory for the output array. Abort.\n");
    exit(1);
  }

  
  // calculate divergence throughout the dataset
  double maxDiv = -999999.99;
  double minDiv =  999999.99;
  double div;
  
  cntz = 0;
  cntnz = 0;
  double zerodiv = 0.1;
  
  /////////////////////////////////////
  Vector v[6];
  double vdist = (CELL_SIZE) / 2.00;
// #ifdef TRACE
  // printf("vdist = %lf\n", vdist);
// #endif	
  
  printf("Finding high divergence points (1).\n");
  unsigned char *d_flags;
  Vector *d_ForceField;
  double *d_maxDiv;
  double *d_minDiv;
  double *d_vdist;
  Divergence_vals *d_arr;
  Divergence_vals *h_arr,*h_result;
  h_arr=(Divergence_vals *)malloc(sizeof(Divergence_vals)*L*M*N);
  h_result=(Divergence_vals *)malloc(sizeof(Divergence_vals)*L*M*N);
  cudaMalloc((void **)&d_flags,sizeof(unsigned char)*L*M*N);
  cudaMalloc((void **)&d_ForceField,sizeof(Vector)*L*M*N);
  cudaMalloc((void **)&d_minDiv,sizeof(double));
  cudaMalloc((void **)&d_arr, sizeof(Divergence_vals)*L*M*N);
  cudaMalloc((void **)&d_vdist,sizeof(double));

  cudaMemcpy(d_flags,flags,sizeof(unsigned char)*L*M*N,cudaMemcpyHostToDevice);
  cudaMemcpy(d_ForceField,ForceField,sizeof(Vector)*L*M*N,cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxDiv,&maxDiv,sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_minDiv,&minDiv,sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_vdist, &vdist, sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimBlock(L-2);
  dim3 dimGrid(N-2,M-2);

  max_min_divergence<<<dimGrid,dimBlock>>>(d_flags,d_ForceField,d_maxDiv,d_minDiv,inOut,slsz,L,M,N,d_vdist,d_arr);
 

  cudaMemcpy(h_arr,d_arr,sizeof(Divergence_vals)*L*M*N,cudaMemcpyDeviceToHost);
 
  for(k=1;k<N-1;k++){   
    for(j=1;j<M-1;j++){
      for(i=1;i<L-1;i++){
        idx = k*slsz + j*L +i;
        if(h_arr[idx].flag==true){
          if(h_arr[idx].val > maxDiv) {
              // printf("%.6f %.6f\n", maxDiv, h_arr[idx].val);
              maxDiv = h_arr[idx].val;
          }
          if(h_arr[idx].val < minDiv) {
              // printf("%.6f %.6f\n", minDiv, h_arr[idx].val);
              minDiv = h_arr[idx].val ;
          }
        }

      }
    }

  }

// #ifdef _DEBUG
  printf("Divergence: max = %.6f, min = %.6f\n", maxDiv, minDiv);
// #endif
  
  double threshold;
  
  
  // case 1:
  // take <perc> percent of the lowest negative value:
  // !! have to change the comparison
  threshold = maxDiv - minDiv;
  threshold = ((double)perc / 100.00) * threshold;
  threshold = minDiv + threshold;
  printf("Finding high divergence points (2).\n"); 
	for(k=1;k<N-1;k++){   
    for(j=1;j<M-1;j++){
      for(i=1;i<L-1;i++){
        idx = k*slsz + j*L +i;
        if(h_arr[idx].flag==true){
          if(h_arr[idx].val <= threshold) {
            // add the point to the HD list
            (*HDPts)[(*numHDPts)].x = i;
            (*HDPts)[(*numHDPts)].y = j;
            (*HDPts)[(*numHDPts)].z = k;
    
            adiv[(*numHDPts)] = h_arr[idx].val;
    
            (*numHDPts) = (*numHDPts) + 1;
            if((*numHDPts) >= MAX_NUM_HDPTS) {
              printf("UPS! Too many high divergence points detected. \
                   Reached maximum of %d. Abort\n", MAX_NUM_HDPTS);
              exit(1);
            } 
          }      
        }

      }
    }
  }
  
  /*
  // case 2:
  // take <perc> percent of the highest pozitive value:
  // !! have to change the comparison
  // NOT GOOD
  threshold = maxDiv - minDiv;
  threshold = ((double)perc / 100.00) * threshold;
  threshold = maxDiv - threshold;
  */
  /*
  // case 3:
  // take <perc> percent of the lowest value (must be negative):
  // !! have to change the comparison
  // NOT GOOD
  threshold = minDiv;
  threshold = ((double)perc / 100.00) * threshold;
  threshold = minDiv - threshold;
  */

#ifdef _DEBUG
  printf("Threshold set to: %lf\n", threshold);
  printf("Number of close to 0 divergence points [-%lf..%lf]: %ld. \n \
                Number of non 0 divergence points: %ld.\n", 
	 zerodiv, zerodiv, cntz, cntnz);
#endif

  
	
  //
  // sort the points on the divergence value;
  //
  
  double minval, tmp;
  int minpos;
  
  for(i=0; i < (*numHDPts); i++) {
    minval = adiv[i];
    minpos = i;
    for(j=i+1; j < (*numHDPts); j++) {
      if(adiv[j] < minval) {
	minval = adiv[j];
	minpos = j;
      }
    }
    if(minpos != i) {
      // exchange points and div values
      tmp = adiv[i];
      adiv[i] = adiv[minpos];
      adiv[minpos] = tmp;
      
      tmp = (*HDPts)[i].x; (*HDPts)[i].x = (*HDPts)[minpos].x; (*HDPts)[minpos].x = tmp;
      tmp = (*HDPts)[i].y; (*HDPts)[i].y = (*HDPts)[minpos].y; (*HDPts)[minpos].y = tmp;
      tmp = (*HDPts)[i].z; (*HDPts)[i].z = (*HDPts)[minpos].z; (*HDPts)[minpos].z = tmp;
    }
  }

#ifdef TRACE
  printf("Points: \n");
  for(i=0; i < (*numHDPts); i++) {
    printf("%f %f %f - %f\n", (*HDPts)[i].x, (*HDPts)[i].y, (*HDPts)[i].z, adiv[i]);
  }
#endif
  
  //
  // cluster the points
  //
  // Algorithm:
  //	First point creates the first group.
  //	For all the other points:
  //		If the point is close to an existing group
  //			add the point to that group
  //		else
  //			the point starts a new group
  //		endif
  //	endfor
  // end
  //
  
  // initialize data structure
  HDGroup *Groups;
  int numGroups = 0;
  
  if((Groups = new HDGroup[(*numHDPts)]) == NULL) {
    printf("Error allocating memory for working data structures. Abort\n");
    exit(1);
  }
  for(i=0; i < (*numHDPts); i++) {
    if((Groups[i].Points = new int[(*numHDPts)]) == NULL) {
      printf("Error allocating memory for working data structures. Abort\n");
      exit(1);
    }
    Groups[i].numPoints = 0;
  }
  
  bool closeToSomeGroup = false;
  
  // first point creates the first group
  Groups[0].Points[0] = 0;
  Groups[0].numPoints = 1;
  numGroups = 1;
  
  for(i=1; i < (*numHDPts); i++) {
    closeToSomeGroup = false;
    for(j=0; j < numGroups; j++) {
      if(PointIsCloseToGroup(i, j, Groups, HDPts)) {
	// add the point to that group
	Groups[j].Points[Groups[j].numPoints] = i;
	Groups[j].numPoints = Groups[j].numPoints + 1;
	closeToSomeGroup = true;
	break;
      }
    }
    if(!closeToSomeGroup) {
      // start a new group
      Groups[numGroups].Points[0] = i;
      Groups[numGroups].numPoints = 1;
      numGroups++;
    }
  }
  
#ifdef TRACE	
  // print the clustered points:
  printf("Clustered points:\n");
  for(i=0; i < numGroups; i++) {
    printf("%f %f %f\n", 
	   (*HDPts)[Groups[i].Points[0]].x, (*HDPts)[Groups[i].Points[0]].y, (*HDPts)[Groups[i].Points[0]].z);
    for(j=1; j < Groups[i].numPoints; j++) {
      printf("\t%f %f %f\n", 
	     (*HDPts)[Groups[i].Points[j]].x, (*HDPts)[Groups[i].Points[j]].y, (*HDPts)[Groups[i].Points[j]].z);
    }
    
  }
#endif
  
  //
  // Return only the first point in each group as the high divergence points
  //
  
  VoxelPositionDouble* newHDPts;
  
  if((newHDPts = new VoxelPositionDouble[numGroups]) == NULL) {
    printf("GetHighDivergencePoints: UPS! - Error allocating memory for the output array. Abort.\n");
    exit(1);
  }
  
  for(i=0; i < numGroups; i++) {
    newHDPts[i].x = (*HDPts)[Groups[i].Points[0]].x;
    newHDPts[i].y = (*HDPts)[Groups[i].Points[0]].y;
    newHDPts[i].z = (*HDPts)[Groups[i].Points[0]].z;
  }
  
  // delete the old array
  delete [] (*HDPts);
  
  // delete Group data structure
  for(i=0; i < numGroups; i++) {
    delete [] Groups[i].Points;
  }
  delete [] Groups;
  
  // return the new array
  (*HDPts) = newHDPts;
  (*numHDPts) = numGroups;
  
#ifdef TRACE
  printf("Returning points: \n");
  for(i=0; i < (*numHDPts); i++) {
    printf("%f %f %f - %f\n", (*HDPts)[i].x, (*HDPts)[i].y, (*HDPts)[i].z, adiv[i]);
  }
#endif
  
  return true;
}


__device__ __host__ inline Vector interpolation(double x, double y, double z, int sizx, int sizy, int sizz, Vector *forcevec)
{
  float alpha, beta, gamma;
  Vector forceInt;
  long slsz;
  
  alpha=x-int(x);
  beta=y-int(y);
  gamma=z-int(z);
  slsz=sizy*sizx;
  
  forceInt.xd=forcevec[int(z)*slsz + int(y)*sizx + int(x)].xd*(1-alpha)*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + int(x)].xd*(1-alpha)*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + int(x)].xd*(1-alpha)*beta*(1-gamma)
    +forcevec[int(z)*slsz + int(y)*sizx + (int(x)+1)].xd*alpha*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + (int(x)+1)].xd*alpha*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + (int(x)+1)].xd*alpha*beta*(1-gamma)
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + int(x)].xd*(1-alpha)*beta*gamma
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + (int(x)+1)].xd*(alpha*beta*gamma);
  
  forceInt.yd=forcevec[int(z)*slsz + int(y)*sizx + int(x)].yd*(1-alpha)*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + int(x)].yd*(1-alpha)*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + int(x)].yd*(1-alpha)*beta*(1-gamma)
    +forcevec[int(z)*slsz + int(y)*sizx + (int(x)+1)].yd*alpha*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + (int(x)+1)].yd*alpha*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + (int(x)+1)].yd*alpha*beta*(1-gamma)
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + int(x)].yd*(1-alpha)*beta*gamma
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + (int(x)+1)].yd*alpha*beta*gamma;
  
  forceInt.zd=forcevec[int(z)*slsz + int(y)*sizx + int(x)].zd*(1-alpha)*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + int(x)].zd*(1-alpha)*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + int(x)].zd*(1-alpha)*beta*(1-gamma)
    +forcevec[int(z)*slsz + int(y)*sizx + (int(x)+1)].zd*alpha*(1-beta)*(1-gamma)
    +forcevec[(int(z)+1)*slsz + int(y)*sizx + (int(x)+1)].zd*alpha*(1-beta)*gamma
    +forcevec[int(z)*slsz + (int(y)+1)*sizx + (int(x)+1)].zd*alpha*beta*(1-gamma)
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + int(x)].zd*(1-alpha)*beta*gamma
    +forcevec[(int(z)+1)*slsz + (int(y)+1)*sizx + (int(x)+1)].zd*alpha*beta*gamma;
  
  return(forceInt);
}


inline bool PointIsCloseToGroup(int pt, int grp, HDGroup *Groups, VoxelPositionDouble **HDPts) {
  int i;
  for(i=0; i < Groups[grp].numPoints; i++) {
    if(
       (fabs((*HDPts)[pt].x - (*HDPts)[Groups[grp].Points[i]].x) <= 1)	&&
       (fabs((*HDPts)[pt].y - (*HDPts)[Groups[grp].Points[i]].y) <= 1)	&&
       (fabs((*HDPts)[pt].z - (*HDPts)[Groups[grp].Points[i]].z) <= 1))
      {
	return true;
      }
  }
  return false;
}
