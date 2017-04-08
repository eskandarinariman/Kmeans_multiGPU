cppflags = -Wall -std=c++14
cpplinks = -lpthread
ccflags  = -Wall
nvflags  = -arch sm_52
kmeans   = kmeans_one_vector kmeans_max_threads kmeans_coalesce

default: opt

# note read_data malloc fails with O3 sometimes
opt: cppflags += -O3
opt: ccflags  += -O3
opt: nvflags  += -O3
opt: $(kmeans) data_gen

debug: cppflags += -g
debug: ccflags  += -g
debug: nvflags  += -g
debug: $(kmeans) data_gen

$(kmeans): %: %.o util.o
	# this also works (need -L to add -lcudart LD_ too late)
	#gcc -o $@ $^ -L/usr/local/cuda-8.0/lib64 -lcudart
	nvcc $(nvflags) -o $@ $^

kmeans_one_vector.o: kmeans.cu
	nvcc $(nvflags) -DONE_VECTOR -o $@ $^ -c

kmeans_max_threads.o: kmeans.cu
	nvcc $(nvflags) -DMAX_THREADS -o $@ $^ -c

kmeans_coalesce.o: kmeans.cu
	nvcc $(nvflags) -DCOALESCE -o $@ $^ -c

%.o: %.[ch]
	gcc $(ccflags) $^ -c

data_gen: data_gen.cpp
	g++ $(cppflags) -o $@ $< $(cpplinks)

clean:
	rm *.o $(kmeans) data_gen

