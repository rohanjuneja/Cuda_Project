COMPILE_OPTIONS		= -D_DEBUG

all: driver driverParallel driverParallel_3

common.o : ../common.h ../common.cpp
	g++ -c ../common.cpp -o common.o $(COMPILE_OPTIONS)
potVect.o : potVect.h potVect.cpp ../common.h
	g++ -c potVect.cpp -o potVect.o $(COMPILE_OPTIONS)

potVectParallel.o : potVect.h potVect_global.cu ../common.h
	/usr/local/cuda-8.0/bin/nvcc -arch=sm_35 -rdc=true -c potVect_global.cu -o potVectParallel.o -lcudadevrt $(COMPILE_OPTIONS)

potVectParallel_3.o : potVect.h potVect_3.cu ../common.h
	/usr/local/cuda-8.0/bin/nvcc -c potVect_3.cu -o potVectParallel_3.o $(COMPILE_OPTIONS)
driver: driver.cpp common.o potVect.o
	g++	driver.cpp common.o potVect.o -o driver $(COMPILE_OPTIONS)

driverParallel:driver.cpp common.o potVectParallel.o
	/usr/local/cuda-8.0/bin/nvcc -arch=sm_35 -rdc=true driver.cpp common.o potVectParallel.o -o driverParallel -lcudadevrt $(COMPILE_OPTIONS)
driverParallel_3: driver.cpp common.o potVectParallel_3.o
	/usr/local/cuda-8.0/bin/nvcc driver.cpp common.o potVectParallel_3.o -o driverParaillel_3 $(COMPILE_OPTIONS)

clean:
	rm -f *.o driver *~
