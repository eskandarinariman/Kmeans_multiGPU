#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <iostream>
#include <cassert>
#include <mutex>

/*
 * generate n files with 4m 256 dimension vectors
 * 1 for each gpu (~3.9GB)
 */

#define VECTORS 3800000
#define DIMS    256
#define THREADS 32

void gen_data(float data[][VECTORS][DIMS], int total_vectors);
void gen_chunk(float data[][VECTORS][DIMS], int vec_start, int chunksize, unsigned int tid);
void write_data(float data[][VECTORS][DIMS], int files);

class clk {
	private:
		struct timespec _start;
		struct timespec _end;

	public:
		void start() { clock_gettime(CLOCK_MONOTONIC, &_start); }
		void end() { clock_gettime(CLOCK_MONOTONIC, &_end); }
		uint64_t diff();
		uint64_t progress() { end(); return diff(); }
};

uint64_t clk::diff() {
	uint64_t s = _start.tv_sec * 1e9 + _start.tv_nsec;
	uint64_t e = _end.tv_sec   * 1e9 + _end.tv_nsec;
	return e - s;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cout << "usage: datagen <no files>" << std::endl;
		exit(1);
	}

	int files = atoi(argv[1]);
	int total_vectors = files * VECTORS;

	float (*data)[VECTORS][DIMS] = new float[files][VECTORS][DIMS];

	clk gen, write;

	gen.start();
	gen_data(data, total_vectors);
	gen.end();

	write.start();
	write_data(data, files);
	write.end();

	delete []data;

	std::cout << "gen: " << gen.diff() << " (ns) " << std::endl;
	std::cout << "write: " << write.diff() << " (ns) " << std::endl;
}

void gen_chunk(float data[][VECTORS][DIMS], int vec_start, int chunksize, unsigned int tid) {
	int file = vec_start / VECTORS;
	int start = vec_start - (file * VECTORS);
	int end = start + chunksize;

	for (int i = start; i < end; i++) {
		for (int j = 0; j < DIMS; j++) {
			float x = static_cast<float>(rand_r(&tid)) / static_cast<float>(RAND_MAX);
			data[file][i][j] = x;
		}
	}
}

void gen_data(float data[][VECTORS][DIMS], int total_vectors) {

	assert(total_vectors % THREADS == 0); // assume even work per thread
	int chunksize = total_vectors / THREADS;
	assert(chunksize < VECTORS); // 1 file per thread at most

	std::thread thread[THREADS];
	int vec_start = 0;
	for (int i = 0; i < THREADS; i++, vec_start += chunksize) {
		thread[i] = std::thread(gen_chunk, data, vec_start, chunksize, i);
	}
	for (int i = 0; i < THREADS; i++) {
		thread[i].join();
	}
	std::cout << "data[0][2][3]: "        << data[0][2][3]        << std::endl;
	std::cout << "data[0][43125][0]: "    << data[0][43125][0]    << std::endl;
	std::cout << "data[0][123456][123]: " << data[0][123456][123] << std::endl;
	std::cout << "data[1][2][3]: "        << data[1][2][3]        << std::endl;
	std::cout << "data[1][43125][0]: "    << data[1][43125][0]    << std::endl;
	std::cout << "data[1][123456][123]: " << data[1][123456][123] << std::endl;
	std::cout << "data[2][2][3]: "        << data[2][2][3]        << std::endl;
	std::cout << "data[2][43125][0]: "    << data[2][43125][0]    << std::endl;
	std::cout << "data[2][123456][123]: " << data[2][123456][123] << std::endl;
	std::cout << "data[3][2][3]: "        << data[3][2][3]        << std::endl;
	std::cout << "data[3][43125][0]: "    << data[3][43125][0]    << std::endl;
	std::cout << "data[3][123456][123]: " << data[3][123456][123] << std::endl;
}

void write_data(float data[][VECTORS][DIMS], int files) {
	char pathname[6] = "data_";
	for (int i = 0; i < files; i++) {
		pathname[4] = '0' + i;

		FILE* stream;
		stream = fopen(pathname, "wb");
		for (int j = 0; j < VECTORS; j++) {
			fwrite(data[i][j], 1, DIMS*sizeof(float), stream);
		}
		fclose(stream);
	}
}

