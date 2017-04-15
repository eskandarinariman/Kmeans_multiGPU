NVCC=nvcc
CC=gcc
CPP=g++

CPPFLAGS = -Wall -std=c++14
CPPLD    = -lpthread
CCFLAGS  = -Wall
NVFLAGS  = -arch sm_52 -default-stream per-thread
kmeans  = one_vector_cpu_sum
#kmeans += one_vector_gpu_sum
kmeans += max_threads_cpu_sum
#kmeans += max_threads_gpu_sum
kmeans += coalesce_cpu_sum_nopin
kmeans += coalesce_cpu_sum_pin
#kmeans += coalesce_gpu_sum

default: opt

# note read_data malloc fails with O3 sometimes
opt: CPPFLAGS += -O3
opt: CCFLAGS  += -O3
opt: NVFLAGS  += -O3
opt: $(kmeans) data_gen

debug: CPPFLAGS += -g
debug: CCFLAGS  += -g
debug: NVFLAGS  += -g
debug: $(kmeans) data_gen

$(kmeans): %: %.o util.o
	$(NVCC) $(NVFLAGS) -o $@ $^

one_vector_cpu_sum.o: DFLAGS=-DONE_VECTOR -DCPU_SUM
one_vector_gpu_sum.o: DFLAGS=-DONE_VECTOR -DGPU_SUM
max_threads_cpu_sum.o: DFLAGS=-DMAX_THREADS -DCPU_SUM
max_threads_gpu_sum.o: DFLAGS=-DMAX_THREADS -DGPU_SUM
coalesce_cpu_sum.o: DFLAGS=-DCOALESCE -DCPU_SUM
coalesce_gpu_sum.o: DFLAGS=-DCOALESCE -DGPU_SUM
coalesce_cpu_sum_nopin.o: DFLAGS=-DCOALESCE -DCPU_SUM
coalesce_cpu_sum_pin.o: DFLAGS=-DCOALESCE -DCPU_SUM -DPINNED

# following line also works (need -L to add -lcudart)
# gcc -o $@ $^ -L/usr/local/cuda-8.0/lib64 -lcudart
$(kmeans:=.o): kmeans.h kmeans.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $(DFLAGS) kmeans.cu

%.o: %.[ch]
	$(CC) $(CCFLAGS) $^ -c

data_gen: data_gen.cpp
	$(CPP) $(CPPFLAGS) -o $@ $< $(CPPLD)

clean:
	rm *.o $(kmeans) data_gen

