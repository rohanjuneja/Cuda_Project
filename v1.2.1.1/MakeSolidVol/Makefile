all: driver driverParallel

makeSolidVol.o : makeSolidVol.cpp makeSolidVol.h
	g++ -c makeSolidVol.cpp -o makeSolidVol.o

makeSolidVolParallel.o : makeSolidVol.cu makeSolidVol.h
	/usr/local/cuda-8.0/bin/nvcc -c makeSolidVol.cu -o makeSolidVolParallel.o

driver: makeSolidVol.o driver.cpp
	g++ driver.cpp makeSolidVol.o -o driver

driverParallel: makeSolidVolParallel.o driver.cpp
	/usr/local/cuda-8.0/bin/nvcc driver.cpp makeSolidVolParallel.o -o driverParallel

clean:
	rm -f driver driverParallel *.o *~

