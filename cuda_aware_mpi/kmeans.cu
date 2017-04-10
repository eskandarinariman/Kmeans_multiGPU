#include <stdio.h>
#include <time.h>
#include <assert.h>
extern "C" {
#include "kmeans.h"
}

/*
 * data       [nvectors  * ndims]
 * clusters   [nclusters * ndims]
 * membership [nvectors]
 */
__device__ inline void
vector_dist(unsigned int vector, const float *data, const float *clusters,
		int *membership, int ndims, int nclusters)
{
	int index = -1;
	float min_dist = FLT_MAX;

	for (int i = 0; i < nclusters; i++) {
		float dist = 0.0;

		for (int j = 0; j < ndims; j++) {
			float diff = data[vector * ndims + j] - clusters[i * ndims + j];
			dist += diff * diff;
		}

		if (dist < min_dist) {
			min_dist = dist;
			index    = i;
		}
	}
	membership[vector] = index;
}

__global__ void
kmeans_one_vector(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters)
{
	unsigned int vector = blockIdx.x * blockDim.x + threadIdx.x;
	vector_dist(vector, data, clusters, membership, ndims, nclusters);
}

__global__ void
kmeans_max_threads(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters, int nvectors, int thread_vectors)
{
	unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * thread_vectors;
	unsigned int end   = start + thread_vectors;
	for (int vector = start; vector < end; vector++) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, membership, ndims, nclusters);
	}
}

__global__ void
kmeans_coalesce(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters, int nvectors, int threads)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int vector = tid; vector < nvectors; vector+=threads) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, membership, ndims, nclusters);
	}
}

/*
 * [hd]_data          [nvectors  * ndims]
 * [hd]_clusters      [nclusters * ndims]
 * [hd]_membership    [nvectors]
 * h_clusters_members [nclusters]
 * h_clusters_sums    [nclusters * ndims]
 */
extern "C" int
run_kmeans(const float *h_data, const float *d_data, float *h_clusters,
		float *d_clusters, int *h_membership, int *d_membership,
		int *h_clusters_members, float *h_clusters_sums, long nvectors,
		int ndims, int nclusters, int niters)
{
	//todo: make threadsPerBock and
#ifdef ONE_VECTOR
	int thread_vectors = 1;
	int block_threads = 64;
	assert(nvectors % thread_vectors == 0);
	assert((nvectors / thread_vectors) % block_threads == 0);
	int grid_blocks = (nvectors / thread_vectors) / block_threads;
#elif MAX_THREADS || COALESCE
	int grid_blocks = 128;
	int block_threads = 16;
	int threads = grid_blocks * block_threads;
	assert(threads == 2048);
#if MAX_THREADS
	int thread_vectors = (nvectors + (threads - 1))/threads;
#endif
#endif

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

//	for (int i = 0; i < niters; i++) {
		cudaError_t err = cudaMemcpy(d_clusters, h_clusters, nclusters * ndims * sizeof(float),
				cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(err));
			return -1;
		}

#ifdef ONE_VECTOR
		kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters);
#elif MAX_THREADS
		kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters, nvectors, thread_vectors);
#elif COALESCE
		kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters, nvectors, threads);
#endif
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "kmeans_kernel error %s\n", cudaGetErrorString(err));
			return -1;
		}
		cudaDeviceSynchronize();

		err = cudaMemcpy(h_membership, d_membership, nvectors * sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy h_membership error %s\n", cudaGetErrorString(err));
			return -1;
		}

		cpu_sum_clusters(h_data, h_membership, h_clusters_members,
				h_clusters_sums, nvectors, ndims, nclusters);

		// for (int i = 0; i < nclusters; i++)
		// 	for (int j = 0; j < ndims; j++)
		// 		h_clusters[i * ndims + j] = h_clusters_sums[i * ndims + j] / h_clusters_members[i];
//	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	// printf("runtime = %luns\n", time_diff(start, end));
	return 0;
}

/*
 * data               [nvectors  * ndims]
 * membership         [nvectors]
 * h_clusters_members [nclusters]
 * h_clusters_sums    [nclusters * ndims]
 */
void
cpu_sum_clusters(const float *data, const int *membership, int *clusters_members,
		float *clusters_sums, long nvectors, int ndims, int nclusters)
{
	for (int i = 0; i < nvectors; i++) {
		int cluster = membership[i];
		clusters_members[cluster]++;
		for (int j = 0; j < ndims; j++)
			clusters_sums[cluster * ndims + j] += data[i * ndims + j];
	}
}

/*
 * [hd]_data          [nvectors  * ndims]
 * [hd]_clusters      [nclusters * ndims]
 * [hd]_membership    [nvectors]
 * h_clusters_members [nclusters]
 * h_clusters_sums    [nclusters * ndims]
 */
extern "C" int
device_setup_data(float **h_data, float **d_data, float **d_clusters,
		int **d_membership, long nvectors, int ndims, int nclusters)
{

	cudaError_t err = cudaMalloc(d_data, nvectors * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_data error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(*d_data, *h_data, nvectors * ndims * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_data error %s\n", cudaGetErrorString(err));
		return 1;
	}

	// for (int i = 0; i < nclusters; i++)
	// 	for (int j = 0; j < ndims; j++)
	// 		//(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];
	// 		printf("%f ",(*h_clusters)[i * ndims + j]);

	err = cudaMalloc(d_clusters, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters error %s\n", cudaGetErrorString(err));
		return -1;
	}

	err = cudaMalloc(d_membership, nvectors * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_membership error %s\n", cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

extern "C" int
setup_data(float **h_data, float **d_data, float **h_clusters, float **d_clusters,
		int **h_membership, int **d_membership, int **h_clusters_members,
		float **h_clusters_sums, long nvectors, int ndims, int nclusters, const char *infile)
{
	*h_data = (float *)malloc(nvectors * ndims * sizeof(float));
	if (*h_data == NULL) {
		fprintf(stderr, "malloc h_data failed\n");
		return 1;
	}

	int errr = read_data(h_data, nvectors * ndims * sizeof(float), infile);
	if (errr) {
		fprintf(stderr, "read_data error: %d\n", errr);
		return 1;
	}

	cudaError_t err = cudaMalloc(d_data, nvectors * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_data error %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(*d_data, *h_data, nvectors * ndims * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_data error %s\n", cudaGetErrorString(err));
		return 1;
	}

	*h_clusters = (float *)malloc(nclusters * ndims * sizeof(float));
	if (*h_clusters == NULL) {
		fprintf(stderr, "malloc h_clusters failed\n");
		return 1;
	}

	for (int i = 0; i < nclusters; i++){
		for (int j = 0; j < ndims; j++){
			(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];
			//printf("%f ",(*h_clusters)[i * ndims + j]);
		}
		//printf("\n");
	}

	err = cudaMalloc(d_clusters, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters error %s\n", cudaGetErrorString(err));
		return -1;
	}

	*h_membership = (int *)malloc(nvectors * sizeof(int));
	if (*h_membership == NULL) {
		fprintf(stderr, "malloc h_membership failed\n");
		return -1;
	}

	err = cudaMalloc(d_membership, nvectors * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_membership error %s\n", cudaGetErrorString(err));
		return -1;
	}

	*h_clusters_members = (int *)malloc(nclusters * sizeof(int));
	if (*h_clusters_members == NULL) {
		fprintf(stderr, "malloc h_clusters_members failed\n");
		return -1;
	}

	*h_clusters_sums = (float *)malloc(nclusters * ndims * sizeof(float));
	if (*h_clusters_sums == NULL) {
		fprintf(stderr, "malloc h_clusters_sums failed\n");
		return -1;
	}

	return 0;
}

// int
// main(int argc, char *argv[])
// {
// 	if (argc != 6) {
// 		printf("usage: ./kmeans <infile> <vectors> <dimensions> <clusters> <iterations>\n");
// 		return 1;
// 	}

// 	char *infile  = argv[1];
// 	// need to be careful with large sizes nvectors * ndims can overflow a signed int
// 	long nvectors  = strtol(argv[2], NULL, 10);
// 	int  ndims     = atoi(argv[3]);
// 	int  nclusters = atoi(argv[4]);
// 	int  niters    = atoi(argv[5]);

// 	float *h_data, *d_data, *h_clusters, *d_clusters, *h_clusters_sums;
// 	int *h_membership, *d_membership, *h_clusters_members;
// 	int err = setup_data(&h_data, &d_data, &h_clusters, &d_clusters,
// 			&h_membership, &d_membership, &h_clusters_members,
// 			&h_clusters_sums, nvectors, ndims, nclusters, infile);
// 	if (err)
// 		return err;

// 	printf("setup complete running kmeans...\n");

// 	err = run_kmeans(h_data, d_data, h_clusters, d_clusters, h_membership,
// 			d_membership, h_clusters_members, h_clusters_sums, nvectors, ndims,
// 			nclusters, niters);
// 	if (err)
// 		return err;

// 	free(h_data);
// 	cudaFree(d_data);
// 	free(h_clusters);
// 	cudaFree(d_clusters);
// 	free(h_membership);
// 	cudaFree(d_membership);
// 	free(h_clusters_members);
// 	free(h_clusters_sums);

// 	cudaDeviceReset();
// 	return err;
// }
