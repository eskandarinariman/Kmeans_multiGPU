#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include "kmeans.h"

#define THREADS 32

uint64_t time_diff(struct timespec start, struct timespec end) {
	uint64_t s = start.tv_sec * 1e9 + start.tv_nsec;
	uint64_t e = end.tv_sec   * 1e9 + end.tv_nsec;
	return e - s;
}

int
read_data(float *data, off_t start_offset, ssize_t size, const char *filename)
{
	int infile;
	if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
		fprintf(stderr, "Error: no such file (%s)\n", filename);
		return 1;
	}

	off_t ret = lseek(infile, start_offset, SEEK_SET);
	if (ret != start_offset) {
		fprintf(stderr, "read_data (%ld %ld %s) offset = %ld\n",
				start_offset, size, filename, ret);
		perror("read");
		return 1;
	}

	ssize_t bytes_left = size;
	ssize_t offset = 0;
	while (bytes_left > 0) {
		// cast to char * since offset in bytes not float
		ssize_t bytes = read(infile, ((char *)data) + offset, bytes_left);
		if (bytes == -1) {
			fprintf(stderr, "read_data (%ld %ld %s) bytes_left = %ld\n",
					start_offset, size, filename, bytes_left);
			perror("read");
			return 1;
		}
		offset     += bytes;
		bytes_left -= bytes;
	}

	close(infile);
	if (bytes_left)
		fprintf(stderr, "read_data (%ld %ld %s) bytes_left = %ld\n", start_offset, size, filename, bytes_left);
	return bytes_left;
}

