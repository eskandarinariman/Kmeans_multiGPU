MPICC=~/MPI/bin/mpicc
MPILD=~/MPI/bin/mpic++
NVCC=$(PREP) nvcc

#CFLAGS=-std=c99 -march=native -Wall
MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

NVCCFLAGS=-Xcompiler -march=native



CUDALDFLAGS=-L${CUDA_INSTALL_PATH}/lib64 -lcudart

all: kmeans.o mpi_hello_world.o mpi_kmeans

kmeans.o: kmeans.h kmeans.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c kmeans.cu -o kmeans.o

util.o: kmeans.h util.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c util.cu -o util.o

mpi_hello_world.o: mpi_hello_world.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c mpi_hello_world.c -o mpi_hello_world.o

mpi_kmeans: kmeans.o util.o mpi_hello_world.o Makefile
	$(MPICC) $(CUDALDFLAGS) -o mpi_kmeans kmeans.o util.o mpi_hello_world.o 


clean:
	rm -rf *.o mpi_kmeans

