LD_LIBRARY_PATH?=/usr/local/cuda-8.0
MPI_HOME?=~/MPI
CUDA_HOME?=/usr/local/cuda-8.0

MPICC=${MPI_HOME}/bin/mpicc
NVCC=nvcc
CC=gcc

CCFLAGS =-Wall
MPICCFLAGS=
NVCCFLAGS=-arch sm_52
NVLDFLAGS=-L${CUDA_HOME}/lib64 -lcudart

default: mpi_c

mpi_kmeans_one_vector.o: DFLAGS=-DONE_VECTOR
mpi_kmeans_max_threads.o: DFLAGS=-DMAX_THREADS
mpi_kmeans_coalesce.o: DFLAGS=-DCOALESCE

util.o: kmeans.h util.c Makefile
	$(CC) $(CCFLAGS) -c util.c

mpi_hello_world.o: mpi_hello_world.c Makefile
	$(MPICC) -c mpi_hello_world.c

mpi_c: util.o mpi_hello_world.o Makefile
	$(MPICC) -o $@ util.o mpi_hello_world.o $(NVLDFLAGS)

clean:
	rm -rf *.o $(kmeans)

# this is just a default/example run
# export LD_LIBRARY_PATH for linking on other machines
# -rf <file> specifies hosts and ranks/slots to run on
run: mpi_c
	${MPI_HOME}/bin/mpiexec -x LD_LIBRARY_PATH -rf ug.mpi mpi_c /tmp/data 10240 256 1024 15
