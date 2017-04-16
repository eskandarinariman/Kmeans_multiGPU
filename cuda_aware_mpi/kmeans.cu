#include <stdio.h>
#include <time.h>
#include <assert.h>
extern "C" {
#include "kmeans.h"
}

#define D_THREADS 2048

#ifdef CPU_SUM
/*
 * data       [nvectors  * ndims]
 * clusters   [nclusters * ndims]
 * membership [nvectors]
 */
__inline__ __device__ void
vector_dist(unsigned int vector, const float* data, const float* clusters,
		int* membership, int ndims, int nclusters)
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
		int ndims, int nclusters, int nvectors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int vector = tid; vector < nvectors; vector += D_THREADS) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, membership, ndims, nclusters);
	}
}
#elif GPU_SUM
/*
 * data       [nvectors  * ndims]
 * clusters   [nclusters * ndims]
 * clusters_sums [nclusters * ndims]
 * clusters_members [nclusters]
 */
__inline__ __device__ void
vector_dist(unsigned int vector, const float* data, const float* clusters,
		float *clusters_sums, int *clusters_members, int ndims, int nclusters)
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

	atomicAdd(&clusters_members[index], 1);
	for (int j = 0; j < ndims; j++) {
		float val = data[vector * ndims + j] ;
		atomicAdd(&clusters_sums[index * ndims + j], val);
	}
}

__global__ void
kmeans_one_vector(const float *data, const float *clusters, float *clusters_sums,
		int *clusters_members, int ndims, int nclusters)
{
	unsigned int vector = blockIdx.x * blockDim.x + threadIdx.x;
	vector_dist(vector, data, clusters, clusters_sums, clusters_members, ndims, nclusters);
}

__global__ void
kmeans_max_threads(const float *data, const float *clusters, float *clusters_sums,
		int *clusters_members, int ndims, int nclusters, int nvectors, int thread_vectors)
{
	unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * thread_vectors;
	unsigned int end   = start + thread_vectors;
	for (int vector = start; vector < end; vector++) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, clusters_sums, clusters_members, ndims, nclusters);
	}
}

__global__ void
kmeans_coalesce(const float *data, const float *clusters, float *clusters_sums,
		int *clusters_members, int ndims, int nclusters, int nvectors)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int vector = tid; vector < nvectors; vector += D_THREADS) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, clusters_sums, clusters_members, ndims, nclusters);
	}
}
#endif

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
int
run_kmeans(const float *h_data, const float *d_data, float *h_clusters, float *d_clusters,
		int *h_membership, int *d_membership, int *h_clusters_members, int *d_clusters_members,
		float *h_clusters_sums, float *d_clusters_sums, long nvectors, int ndims, int nclusters, int niters)
{
#ifdef ONE_VECTOR
	int thread_vectors = 1;
	int block_threads = 64;
	assert(nvectors % thread_vectors == 0);
	assert((nvectors / thread_vectors) % block_threads == 0);
	int grid_blocks = (nvectors / thread_vectors) / block_threads;
#elif MAX_THREADS || COALESCE
	int grid_blocks = 16;
	int block_threads = 128;
	assert(grid_blocks * block_threads == D_THREADS);
#if MAX_THREADS
	int thread_vectors = (nvectors + (D_THREADS - 1)) / D_THREADS;
#endif
#endif

	cudaError_t err = cudaMemcpy(d_clusters, h_clusters, nclusters * ndims * sizeof(float),
			cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(err));
		return -1;
	}

#if CPU_SUM
#ifdef ONE_VECTOR
	kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_membership, ndims, nclusters);
#elif MAX_THREADS
	kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_membership, ndims, nclusters, nvectors, thread_vectors);
#elif COALESCE
	kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_membership, ndims, nclusters, nvectors);
#endif /* membership type */
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

#elif GPU_SUM
#if ONE_VECTOR
	kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_clusters_sums, d_clusters_members, ndims, nclusters);
#elif MAX_THREADS
	kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_clusters_sums, d_clusters_members, ndims, nclusters, nvectors, thread_vectors);
#elif COALESCE
	kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
			d_clusters_sums, d_clusters_members, ndims, nclusters, nvectors);
#endif /* membership type */
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "kmeans_kernel error %s\n", cudaGetErrorString(err));
		return -1;
	}
	cudaDeviceSynchronize();

	err = cudaMemcpy(h_clusters_sums, d_clusters_sums,
			nclusters * ndims * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy h_clusters_sums error %s\n", cudaGetErrorString(err));
		exit(2);
	}	

	err = cudaMemcpy(h_clusters_members, d_clusters_members,
			nclusters * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy h_clusters_members error %s\n", cudaGetErrorString(err));
		exit(2);
	}
#endif /* sum type */
	return 0;
}

/*
 * [hd]_data          [nvectors  * ndims]
 * [hd]_clusters      [nclusters * ndims]
 * [hd]_membership    [nvectors]
 * h_clusters_members [nclusters]
 * h_clusters_sums    [nclusters * ndims]
 */
int
device_setup_data(float **h_data, float **d_data, float **d_clusters,
		int **d_membership, int **d_clusters_local_members, float **d_clusters_local_sums,
		long nvectors, int ndims, int nclusters)
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

	err = cudaMalloc(d_clusters_local_members, nclusters * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_local_members error %s\n", cudaGetErrorString(err));
		return -1;
	}
	err = cudaMalloc(d_clusters_local_sums, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_local_sums error %s\n", cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

int 
host_setup_data(float **h_data, float **h_clusters, int **h_membership, int **h_clusters_members,
		float **h_clusters_sums, float ** h_clusters_global_sums, int ** h_clusters_global_members,
		long nvectors, int ndims, int nclusters, const char *infile, int world_rank)
{
	int i,j;

	*h_data = (float *)malloc(nvectors * ndims * sizeof(float));
	if (*h_data == NULL) {
		fprintf(stderr, "malloc h_data failed\n");
		return 1;
	}

#ifdef REMOTE_DATA

	if(world_rank == 0){

		int errr = read_data(h_data, nvectors * ndims * sizeof(float), infile);
		if (errr) {
			fprintf(stderr, "read_data error: %d\n", errr);
			return 1;
		}
	}
#else
	int errr = read_data(h_data, nvectors * ndims * sizeof(float), infile);
	if (errr) {
		fprintf(stderr, "read_data error: %d\n", errr);
		return 1;
	}
#endif

	*h_clusters = (float *)malloc(nclusters * ndims * sizeof(float));
	if (*h_clusters == NULL) {
		fprintf(stderr, "malloc h_clusters failed\n");
		return 1;
	}

	if(world_rank == 0) {
		for (i = 0; i < nclusters; i++)
			for (j = 0; j < ndims; j++)
				(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];
	}

	*h_membership = (int *)malloc(nvectors * sizeof(int));
	if (*h_membership == NULL) {
		fprintf(stderr, "malloc h_membership failed\n");
		return -1;
	}
	memset(*h_membership, 0, nvectors * sizeof(int)); 

	*h_clusters_members = (int *)malloc(nclusters * sizeof(int));
	if (*h_clusters_members == NULL) {
		fprintf(stderr, "malloc h_clusters_members failed\n");
		return -1;
	}
	memset(*h_clusters_members, 0, nclusters * sizeof(int)); 

	if(world_rank == 0) {
		*h_clusters_global_members = (int *)malloc(nclusters * sizeof(int));
		if (*h_clusters_global_members == NULL) {
			fprintf(stderr, "malloc h_clusters_members failed\n");
			return -1;
		}
		memset(*h_clusters_global_members, 0, nclusters * sizeof(int)); 
	}

	*h_clusters_sums = (float *)malloc(nclusters * ndims * sizeof(float));
	if (*h_clusters_sums == NULL) {
		fprintf(stderr, "malloc h_clusters_sums failed\n");
		return -1;
	}
	memset(*h_clusters_sums, 0, nclusters * ndims * sizeof(float)); 

	if(world_rank == 0) {
		*h_clusters_global_sums = (float *)malloc(nclusters * ndims * sizeof(float));
		if (*h_clusters_global_sums == NULL) {
			fprintf(stderr, "malloc h_clusters_sums failed\n");
			return -1;
		}
		memset(*h_clusters_global_sums, 0, nclusters * ndims * sizeof(float)); 
	}

	return 0;
}

