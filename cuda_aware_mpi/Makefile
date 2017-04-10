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

kmeans  = one_vector_cpu
#kmeans += one_vector_gpu
kmeans += max_threads_cpu
#kmeans += max_threads_gpu
kmeans += coalesce_cpu
#kmeans += coalesce_gpu

default: $(kmeans)

one_vector_cpu.o: DFLAGS=-DONE_VECTOR -DCPU_SUM
one_vector_gpu.o: DFLAGS=-DONE_VECTOR -DGPU_SUM
max_threads_cpu.o: DFLAGS=-DMAX_THREADS -DCPU_SUM
max_threads_gpu.o: DFLAGS=-DMAX_THREADS -DGPU_SUM
coalesce_cpu.o: DFLAGS=-DCOALESCE -DCPU_SUM
coalesce_gpu.o: DFLAGS=-DCOALESCE -DGPU_SUM

$(kmeans:=.o): kmeans.h kmeans.cu Makefile
	$(NVCC) $(NVCCFLAGS) -c $(DFLAGS) kmeans.cu -o $@

util.o: kmeans.h util.c Makefile
	$(CC) $(CCFLAGS) -c util.c

mpi_hello_world.o: mpi_hello_world.c Makefile
	$(MPICC) -c mpi_hello_world.c

$(kmeans): %: %.o util.o mpi_hello_world.o Makefile
	$(MPICC) -o $@ $@.o util.o mpi_hello_world.o $(NVLDFLAGS)

clean:
	rm -rf *.o $(kmeans)

# this is just a default/example run
# export LD_LIBRARY_PATH for linking on other machines
# -rf <file> specifies hosts and ranks/slots to run on
run: mpi_kmeans_one_vector
	${MPI_HOME}/bin/mpiexec -x LD_LIBRARY_PATH -rf ug.mpi mpi_kmeans_one_vector /tmp/data 3800000 256 2 1
