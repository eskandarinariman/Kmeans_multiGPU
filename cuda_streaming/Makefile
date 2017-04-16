NVCC=nvcc
CC=gcc
CPP=g++

CPPFLAGS = -Wall -std=c++14
CPPLD    = -lpthread
CCFLAGS  = -Wall
NVFLAGS  = -arch sm_52 -default-stream per-thread
kmeans  = one_vector_cpu_sum_nopin
kmeans  = one_vector_cpu_sum_pin
kmeans += one_vector_gpu_sum_nopin
kmeans += one_vector_gpu_sum_pin
kmeans += max_threads_cpu_sum_nopin
kmeans += max_threads_cpu_sum_pin
kmeans += max_threads_gpu_sum_nopin
kmeans += max_threads_gpu_sum_pin
kmeans += coalesce_cpu_sum_nopin
kmeans += coalesce_cpu_sum_pin
kmeans += coalesce_gpu_sum_nopin
kmeans += coalesce_gpu_sum_pin

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

one_vector_cpu_sum_nopin.o: DFLAGS=-DONE_VECTOR -DCPU_SUM
one_vector_cpu_sum_pin.o: DFLAGS=-DONE_VECTOR -DCPU_SUM -DPINNED
one_vector_gpu_sum_nopin.o: DFLAGS=-DONE_VECTOR -DGPU_SUM
one_vector_gpu_sum_pin.o: DFLAGS=-DONE_VECTOR -DGPU_SUM -DPINNED
max_threads_cpu_sum_nopin.o: DFLAGS=-DMAX_THREADS -DCPU_SUM
max_threads_cpu_sum_pin.o: DFLAGS=-DMAX_THREADS -DCPU_SUM -DPINNED
max_threads_gpu_sum_nopin.o: DFLAGS=-DMAX_THREADS -DGPU_SUM
max_threads_gpu_sum_pin.o: DFLAGS=-DMAX_THREADS -DGPU_SUM -DPINNED
coalesce_cpu_sum_nopin.o: DFLAGS=-DCOALESCE -DCPU_SUM
coalesce_cpu_sum_pin.o: DFLAGS=-DCOALESCE -DCPU_SUM -DPINNED
coalesce_gpu_sum_nopin.o: DFLAGS=-DCOALESCE -DGPU_SUM
coalesce_gpu_sum_pin.o: DFLAGS=-DCOALESCE -DGPU_SUM -DPINNED

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

