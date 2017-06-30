COMPILE_OPTIONS		= -D_DEBUG

TNT_HOME = /home/cornea/Software/TNT
JAMA_HOME = /home/cornea/Software/JAMA

all: driver driverParallel

common.o:	../common.h ../common.cpp
	g++ -c ../common.cpp -o common.o $(COMPILE_OPTIONS)

HighDiverg.o:	HighDiverg.cpp HighDiverg.h ../common.h
	g++ -c HighDiverg.cpp -o HighDiverg.o $(COMPILE_OPTIONS) -I$(TNT_HOME) -I$(JAMA_HOME)

HighDivergParallel.o: HighDiverg.cu HighDiverg.h
	/usr/local/cuda-8.0/bin/nvcc -c HighDiverg.cu -o HighDivergParallel.o

driver:		driver.cpp HighDiverg.o common.o
	g++ driver.cpp HighDiverg.o common.o -o driver $(COMPILE_OPTIONS) -I$(TNT_HOME) -I$(JAMA_HOME)

driverParallel:	driver.cpp HighDivergParallel.o common.o
	/usr/local/cuda-8.0/bin/nvcc driver.cpp HighDivergParallel.o common.o -o driverParallel  $(COMPILE_OPTIONS) -I$(TNT_HOME) -I$(JAMA_HOME)

clean:	
	rm -f driver driverParallel *.o *~
