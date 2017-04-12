LD_LIBRARY_PATH?=/usr/local/cuda-8.0
MPI_HOME?=~/MPI
CUDA_HOME?=/usr/local/cuda-8.0

MPICC=${MPI_HOME}/bin/mpicc
NVCC=nvcc
CC=gcc

CCFLAGS =-Wall
MPICCFLAGS=-Wall
NVCCFLAGS=-arch sm_52
MPICCLD=-L${CUDA_HOME}/lib64 -lcudart

kmeans  = one_vector_cpu_sum
#kmeans += one_vector_gpu_sum
kmeans += max_threads_cpu_sum
#kmeans += max_threads_gpu_sum
kmeans += coalesce_cpu_sum
#kmeans += coalesce_gpu_sum

default: $(kmeans)

one_vector_cpu_sum.o: DFLAGS=-DONE_VECTOR -DCPU_SUM
one_vector_gpu_sum.o: DFLAGS=-DONE_VECTOR -DGPU_SUM
max_threads_cpu_sum.o: DFLAGS=-DMAX_THREADS -DCPU_SUM
max_threads_gpu_sum.o: DFLAGS=-DMAX_THREADS -DGPU_SUM
coalesce_cpu_sum.o: DFLAGS=-DCOALESCE -DCPU_SUM
coalesce_gpu_sum.o: DFLAGS=-DCOALESCE -DGPU_SUM

$(kmeans:=.o): kmeans.h kmeans.cu Makefile
	$(NVCC) $(NVCCFLAGS) -c $(DFLAGS) kmeans.cu -o $@

util.o: kmeans.h util.c Makefile
	$(CC) $(CCFLAGS) -c util.c

main.o: main.c Makefile
	$(MPICC) $(MPICCFLAGS) -c main.c

$(kmeans): %: %.o util.o main.o Makefile
	$(MPICC) $(MPICCFLAGS) -o $@ $@.o util.o main.o $(MPICCLD)

clean:
	rm -f *.o $(kmeans)

# this is just a default/example run
# export LD_LIBRARY_PATH for linking on other machines
# -rf <file> specifies hosts and ranks/slots to run on
run: mpi_kmeans_one_vector
	${MPI_HOME}/bin/mpiexec -x LD_LIBRARY_PATH -rf ug.mpi mpi_kmeans_one_vector /tmp/data 10240 256 1024 15
