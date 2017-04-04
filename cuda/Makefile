cppflags = -Wall -std=c++14
cpplinks = -lpthread
nvflags  = -arch sm_52

default: opt

# note read_data malloc fails with O3 sometimes
opt: cppflags += -O3
opt: nvflags  += -O3
opt: kmeans data_gen

debug: cppflags += -g
debug: ccflags  += -g
debug: nvflags  += -g
debug: kmeans data_gen

kmeans: util.cu kmeans.cu
	nvcc $(nvflags) -o $@ $^ 

data_gen: data_gen.cpp
	g++ $(cppflags) -o $@ $< $(cpplinks)

clean:
	rm data_gen
	rm kmeans

