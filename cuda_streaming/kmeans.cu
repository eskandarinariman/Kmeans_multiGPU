#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <pthread.h>
extern "C" {
#include "kmeans.h"
}

#define D_THREADS 2048
pthread_barrier_t barrier;

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

void
print_clusters(float *h_clusters, int ndims, int nclusters)
{
	for (int i = 0; i < nclusters; i++) {
		printf("clusters[%d]:\n", i);
		printf("%f", h_clusters[i * ndims + 0]);
		for (int j = 1; j < ndims; j++)
			printf(", %f", h_clusters[i * ndims + j]);
		printf("\n");
	}
}

/*
 * [hd]_data          [nvectors  * ndims]
 * [hd]_clusters      [nclusters * ndims]
 * [hd]_membership    [nvectors]
 * h_clusters_members [stream][nclusters]
 * h_clusters_sums    [stream][nclusters * ndims]
 */
struct ks_args {
	float  *h_clusters;
	float  *d_clusters;
	float  **h_clusters_sums;
	int    **h_clusters_members;
	long   stream_nvectors;
	int    stream_nclusters;
	size_t data_size;
	size_t cluster_size;
	size_t membership_size;
	size_t clusters_members_size;
	size_t clusters_sums_size;
	int    grid_blocks;
	int    block_threads;
	int    ndims;
	int    nclusters;
	int    niters;
	int    nstreams;
	char   **infiles;
	int    nfiles;
	int    streamid;
};

void *
kmeans_stream(void *vargs)
{
	struct ks_args *args = (struct ks_args *)vargs;
	int err;
	cudaError_t cerr;
	/*struct timespec start, end;*/
	/*clock_gettime(CLOCK_MONOTONIC, &start);*/
	float  *h_clusters           = args->h_clusters;
	float  *d_clusters           = args->d_clusters;
	float  **h_clusters_sums     = args->h_clusters_sums;
	int    **h_clusters_members  = args->h_clusters_members;
	long   stream_nvectors       = args->stream_nvectors;
	int    stream_nclusters      = args->stream_nclusters;
	size_t data_size             = args->data_size;
	size_t cluster_size          = args->cluster_size;
	size_t membership_size       = args->membership_size;
	size_t clusters_members_size = args->clusters_members_size;
	size_t clusters_sums_size    = args->clusters_sums_size;
	int    grid_blocks           = args->grid_blocks;
	int    block_threads         = args->block_threads;
	int    ndims                 = args->ndims;
	int    nclusters             = args->nclusters;
	int    niters                = args->niters;
	int    nstreams              = args->nstreams;
	char   **infiles             = args->infiles;
	int    nfiles                = args->nfiles;
	int    streamid              = args->streamid;
	float *h_data, *d_data;
#if CPU_SUM
	int *h_membership, *d_membership;
#elif GPU_SUM
	float *d_clusters_sums;
	int	  *d_clusters_members;
#endif
	int start_cluster;

	start_cluster = streamid * stream_nclusters;

#if PINNED
	cerr = cudaMallocHost(&h_data, data_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamallochost h_data error %s\n", cudaGetErrorString(cerr));
		exit(1);
	}
#if CPU_SUM
	cerr = cudaMallocHost(&h_membership, membership_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamallochost h_membership error %s\n", cudaGetErrorString(cerr));
		exit(1);
	}
#endif
	cerr = cudaMallocHost(&h_clusters_members[streamid], clusters_members_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamallochost h_clusters_members[%d] error %s\n",
				streamid, cudaGetErrorString(cerr));
		exit(1);
	}
	cerr = cudaMallocHost(&h_clusters_sums[streamid], clusters_sums_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamallochost h_clusters_sums[%d] error %s\n",
				streamid, cudaGetErrorString(cerr));
		exit(1);
	}
#else
	h_data = (float *)malloc(data_size);
	if (h_data == NULL) {
		fprintf(stderr, "malloc h_data failed\n");
		exit(1);
	}
#if CPU_SUM
	h_membership = (int *)malloc(membership_size);
	if (h_membership == NULL) {
		fprintf(stderr, "malloc h_membership failed\n");
		exit(1);
	}
#endif
	h_clusters_members[streamid] = (int *)malloc(clusters_members_size);
	if (h_clusters_members[streamid] == NULL) {
		fprintf(stderr, "malloc h_clusters_members[%d] failed\n", streamid);
		exit(1);
	}
	h_clusters_sums[streamid] = (float *)malloc(clusters_sums_size);
	if (h_clusters_sums[streamid] == NULL) {
		fprintf(stderr, "malloc h_clusters_sums[%d] failed\n", streamid);
		exit(1);
	}
#endif
	cerr = cudaMalloc(&d_data, data_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_data error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#if CPU_SUM
	cerr = cudaMalloc(&d_membership, membership_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_membership error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#elif GPU_SUM
	cerr = cudaMalloc(&d_clusters_members, clusters_members_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_members error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
	cerr = cudaMalloc(&d_clusters_sums, clusters_sums_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_sums error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#endif

	/*clock_gettime(CLOCK_MONOTONIC, &end);*/
	/*printf("[s%d] init = %luns\n", streamid, time_diff(start, end));*/

	for (int iter = 0; iter < niters; iter++) {
		struct timespec iter_start, iter_end;
		clock_gettime(CLOCK_MONOTONIC, &iter_start);

		/*
		 * NOTE: we are assuming the size of 1 file is fits nicely into gpu
		 * memory, so each stream reads 1/nstream of each file
		 */
		for (int fileno = 0; fileno < nfiles; fileno++) {

			/*clock_gettime(CLOCK_MONOTONIC, &start);*/

			err = read_data(h_data, streamid * data_size, data_size, infiles[fileno]);
			if (err)
				exit(1);

			/*clock_gettime(CLOCK_MONOTONIC, &end);*/
			/*printf("[s%d] read_data = %luns\n", streamid, time_diff(start, end));*/
			/*clock_gettime(CLOCK_MONOTONIC, &start);*/

			cerr = cudaMemcpyAsync(d_data, h_data, data_size, cudaMemcpyHostToDevice);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudamemcpy d_data error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}

			/* launch kernel for membership */
#if CPU_SUM
#if ONE_VECTOR
			kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters);
#elif MAX_THREADS
			int thread_vectors = (stream_nvectors + (D_THREADS - 1)) / D_THREADS;
			kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters, stream_nvectors, thread_vectors);
#elif COALESCE
			kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters, stream_nvectors);
#endif /* membership type */

			cerr = cudaMemcpyAsync(h_membership, d_membership, membership_size, cudaMemcpyDeviceToHost);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudamemcpy h_membership error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}
			cerr = cudaStreamSynchronize(0);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudastreamsync error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}

			/*clock_gettime(CLOCK_MONOTONIC, &end);*/
			/*printf("[s%d] member sync = %luns\n", streamid, time_diff(start, end));*/
			/*clock_gettime(CLOCK_MONOTONIC, &start);*/

			/* calculate sum and count for each cluster */
			for (int i = 0; i < stream_nvectors; i++) {
				int cluster = h_membership[i];
				h_clusters_members[streamid][cluster]++;
				for (int j = 0; j < ndims; j++)
					h_clusters_sums[streamid][cluster * ndims + j] += h_data[i * ndims + j];
			}
#elif GPU_SUM
#if ONE_VECTOR
			kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters);
#elif MAX_THREADS
			int thread_vectors = (stream_nvectors + (D_THREADS - 1)) / D_THREADS;
			kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters, stream_nvectors, thread_vectors);
#elif COALESCE
			kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters, stream_nvectors);
#endif /* membership type */

			cerr = cudaMemcpyAsync(h_clusters_sums[streamid], d_clusters_sums,
					clusters_sums_size, cudaMemcpyDeviceToHost);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudamemcpy h_clusters_sums error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}	

			cerr = cudaMemcpyAsync(h_clusters_members[streamid], d_clusters_members,
					clusters_members_size, cudaMemcpyDeviceToHost);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudamemcpy h_clusters_members error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}
			cerr = cudaStreamSynchronize(0);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudastreamsync error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}
#endif /* sum type */

			/*clock_gettime(CLOCK_MONOTONIC, &end);*/
			/*printf("[s%d] sum sync = %luns\n", streamid, time_diff(start, end));*/
			/*clock_gettime(CLOCK_MONOTONIC, &start);*/
		}

		/* if all data for thread is processed barrier */
		err = pthread_barrier_wait(&barrier);
		if(err && err != PTHREAD_BARRIER_SERIAL_THREAD) {
			fprintf(stderr, "error waiting on data barrier... %d\n", err);
			exit(1);
		}

		/*clock_gettime(CLOCK_MONOTONIC, &end);*/
		/*printf("[s%d] barrier = %luns\n", streamid, time_diff(start, end));*/
		/*clock_gettime(CLOCK_MONOTONIC, &start);*/

		/* each thread computes some new cluster */
		for (int c = start_cluster; c < start_cluster + stream_nclusters; c++) {
			for (int d = 0; d < ndims; d++) {
				int members = 0;
				float sum = 0; 
				for (int s = 0; s < nstreams; s++) {
					members += h_clusters_members[s][c];
					sum += h_clusters_sums[s][c * ndims + d];
				}
				h_clusters[c * ndims + d] = sum / members;
			}
		}

		/*clock_gettime(CLOCK_MONOTONIC, &end);*/
		/*printf("[s%d] new clusters = %luns\n", streamid, time_diff(start, end));*/
		/*clock_gettime(CLOCK_MONOTONIC, &start);*/

		if (streamid == 0) {
			cerr = cudaMemcpy(d_clusters, h_clusters, cluster_size, cudaMemcpyHostToDevice);
			if (cerr != cudaSuccess) {
				fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(cerr));
				exit(2);
			}
		}

		/*clock_gettime(CLOCK_MONOTONIC, &end);*/
		/*printf("[s%d] memcpy clusters = %luns\n", streamid, time_diff(start, end));*/
		/*clock_gettime(CLOCK_MONOTONIC, &start);*/

		/* barrier for clusters to be prepared for next iter */
		err = pthread_barrier_wait(&barrier);
		if(err && err != PTHREAD_BARRIER_SERIAL_THREAD) {
			fprintf(stderr, "error waiting on cluster barrier... %d\n", err);
			exit(1);
		}

		/*clock_gettime(CLOCK_MONOTONIC, &end);*/
		/*printf("[s%d] barrier = %luns\n", streamid, time_diff(start, end));*/

		clock_gettime(CLOCK_MONOTONIC, &iter_end);
		printf("[%d] iter(%d) = %luns\n", streamid, iter, time_diff(iter_start, iter_end));
	}

#if CPU_SUM
	cerr = cudaFree(d_membership);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafree d_membership error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#elif GPU_SUM
	cerr = cudaFree(d_clusters_sums);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafree d_clusters_sums error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
	cerr = cudaFree(d_clusters_members);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafree d_clusters_members error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#endif
	cerr = cudaFree(d_data);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafree d_data error %s\n", cudaGetErrorString(cerr));
		exit(2);
	}
#if PINNED
#if CPU_SUM
	cerr = cudaFreeHost(h_membership);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafreehost h_membership error %s\n", cudaGetErrorString(cerr));
		exit(1);
	}
#elif GPU_SUM
	cerr = cudaFreeHost(h_clusters_sums[streamid]);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafreehost h_clusters_sums[%d] error %s\n",
				streamid, cudaGetErrorString(cerr));
		exit(1);
	}
	cerr = cudaFreeHost(h_clusters_members[streamid]);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafreehost h_clusters_members[%d] error %s\n",
				streamid, cudaGetErrorString(cerr));
		exit(1);
	}
#endif
	cerr = cudaFreeHost(h_data);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafreehost h_data error %s\n", cudaGetErrorString(cerr));
		exit(1);
	}
#else
	free(h_clusters_sums[streamid]);
	free(h_clusters_members[streamid]);
#if CPU_SUM
	free(h_membership);
#endif
	free(h_data);
#endif
	return NULL;
}

int
kmeans_streams(long nvectors, int ndims, int nclusters, int niters, int nstreams, char **infiles, int nfiles)
{
	int err;
	cudaError_t cerr;
	pthread_t streams[nstreams];
	struct ks_args **args;
	float *h_clusters, *d_clusters, **h_clusters_sums;
	int **h_clusters_members;
	long stream_nvectors;
	int stream_nclusters;
	size_t data_size, cluster_size, membership_size, clusters_members_size, clusters_sums_size;
	/*struct timespec start, end;*/

	/*clock_gettime(CLOCK_MONOTONIC, &start);*/
	
	/* to simplify stuff and make calculations easier... */
	assert(nvectors  % nstreams == 0);
	assert(nclusters % nstreams == 0);

	stream_nvectors       = nvectors/nstreams;
	data_size             = stream_nvectors * ndims * sizeof(float);
	cluster_size          = nclusters * ndims * sizeof(float);
	membership_size       = stream_nvectors * sizeof(int);
	clusters_members_size = nclusters * sizeof(int);
	clusters_sums_size    = nclusters * ndims * sizeof(float);
	stream_nclusters      = nclusters/nstreams;

#if ONE_VECTOR
	int thread_vectors = 1;
	int block_threads = 76; // TODO: ... does this matter?
	assert(stream_nvectors % thread_vectors == 0);
	assert((stream_nvectors / thread_vectors) % block_threads == 0);
	int grid_blocks = (stream_nvectors / thread_vectors) / block_threads;
#elif (defined MAX_THREADS) || (defined COALESCE)
	int grid_blocks = 16;
	int block_threads = 128;
	assert(grid_blocks * block_threads == D_THREADS);
#endif

	err = pthread_barrier_init(&barrier, NULL, nstreams);
	if (err) {
		fprintf(stderr, "error barrier init %d\n", err);
		return 3;
	}
	h_clusters_members = (int **)malloc(nstreams * sizeof(int *));
	if (h_clusters_members == NULL) {
		fprintf(stderr, "malloc h_clusters_members failed\n");
		return 1;
	}
	h_clusters_sums = (float **)malloc(nstreams * sizeof(float *));
	if (h_clusters_sums == NULL) {
		fprintf(stderr, "malloc h_clusters_sums failed\n");
		return 1;
	}
#if PINNED
	cerr = cudaMallocHost(&h_clusters, cluster_size, cudaHostAllocPortable);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamallochost h_clusters error %s\n", cudaGetErrorString(cerr));
		return 1;
	}
#else
	/* init clusters using begining of data in first file */
	h_clusters = (float *)malloc(cluster_size);
	if (h_clusters == NULL) {
		fprintf(stderr, "malloc h_clusters failed\n");
		return 1;
	}
#endif
	err = read_data(h_clusters, 0, cluster_size, infiles[0]);
	if (err)
		return 1;
	cerr = cudaMalloc(&d_clusters, cluster_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters error %s\n", cudaGetErrorString(cerr));
		return 2;
	}
	cerr = cudaMemcpy(d_clusters, h_clusters, cluster_size, cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(cerr));
		return 2;
	}

	/*clock_gettime(CLOCK_MONOTONIC, &end);*/
	/*printf("kmean_streams init = %luns\n", time_diff(start, end));*/
	/*clock_gettime(CLOCK_MONOTONIC, &start);*/

	args = (struct ks_args **)malloc(nstreams * sizeof(struct ks_args *));
	if (args == NULL) {
		fprintf(stderr, "malloc args failed\n");
		return 1;
	}
	for (int streamid = 0; streamid < nstreams; streamid++) {
		args[streamid] = (struct ks_args *)malloc(sizeof(struct ks_args));
		if (args[streamid] == NULL) {
			fprintf(stderr, "malloc args[%d] failed\n", streamid);
			return 1;
		}
		args[streamid]->h_clusters            = h_clusters;
		args[streamid]->d_clusters            = d_clusters;
		args[streamid]->h_clusters_sums       = h_clusters_sums;
		args[streamid]->h_clusters_members    = h_clusters_members;
		args[streamid]->stream_nvectors       = stream_nvectors;
		args[streamid]->stream_nclusters      = stream_nclusters;
		args[streamid]->data_size             = data_size;
		args[streamid]->cluster_size          = cluster_size;
		args[streamid]->membership_size       = membership_size;
		args[streamid]->clusters_members_size = clusters_members_size;
		args[streamid]->clusters_sums_size    = clusters_sums_size;
		args[streamid]->grid_blocks           = grid_blocks;
		args[streamid]->block_threads         = block_threads;
		args[streamid]->ndims                 = ndims;
		args[streamid]->nclusters             = nclusters;
		args[streamid]->niters                = niters;
		args[streamid]->nstreams              = nstreams;
		args[streamid]->infiles               = infiles;
		args[streamid]->nfiles                = nfiles;
		args[streamid]->streamid              = streamid;
		err = pthread_create(&streams[streamid], NULL, kmeans_stream, args[streamid]);
		if (err) {
			fprintf(stderr, "error creating thread %d\n", streamid);
			return 2;
		}
	}

	/*clock_gettime(CLOCK_MONOTONIC, &end);*/
	/*printf("kmean_streams launch = %luns\n", time_diff(start, end));*/
	/*clock_gettime(CLOCK_MONOTONIC, &start);*/

	for (int streamid = 0; streamid < nstreams; streamid++) {
		err = pthread_join(streams[streamid], NULL);
		if(err) {
			fprintf(stderr, "error joining thread %d\n", streamid);
			return 2;
		}
	}

	/*clock_gettime(CLOCK_MONOTONIC, &end);*/
	/*printf("kmean_streams finished = %luns\n", time_diff(start, end));*/

	/*print_clusters(h_clusters, ndims, nclusters);*/

	err = pthread_barrier_destroy(&barrier);
	if (err) {
		fprintf(stderr, "error barrier destroy %d\n", err);
		return 1;
	}

	for (int streamid = 0; streamid < nstreams; streamid++)
		free(args[streamid]);
	free(args);
	cerr = cudaFree(d_clusters);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafree d_clusters error %s\n", cudaGetErrorString(cerr));
		return 2;
	}
#if PINNED
	cerr = cudaFreeHost(h_clusters);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudafreehost h_clusters error %s\n", cudaGetErrorString(cerr));
		return 1;
	}
#else
	free(h_clusters);
#endif
	free(h_clusters_sums);
	free(h_clusters_members);
	return 0;
}

int
main(int argc, char *argv[])
{
	if (argc < 7) {
		printf("usage: ./kmeans <vectors> <dimensions> <clusters> <iterations> <streams> <infile0> <infile1> ....\n");
		printf("note: a single file's data must be able to fit on the gpu at one time\n");
		return 1;
	}

	// need to be careful with large sizes nvectors * ndims can overflow a signed int
	long nvectors  = strtol(argv[1], NULL, 10);
	int  ndims     = atoi(argv[2]);
	int  nclusters = atoi(argv[3]);
	int  niters    = atoi(argv[4]);
	int  nstreams  = atoi(argv[5]);
	int  filearg   = 6;

	int nfiles = argc - filearg;
	char **infiles = (char **)malloc(nfiles * sizeof(char *));
	for (int i = 0; i < nfiles; i++) {
		infiles[i] = argv[i + filearg];
	}

	int err;
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	err = kmeans_streams(nvectors, ndims, nclusters, niters, nstreams, infiles, nfiles);
	if (err)
		return err;

	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("runtime = %luns\n", time_diff(start, end));

	free(infiles);

	cudaDeviceReset();
	return err;
}

