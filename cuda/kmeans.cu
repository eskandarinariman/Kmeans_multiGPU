#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
extern "C" {
#include "kmeans.h"
}

#define D_THREADS 2048

using namespace std;

 __global__ void initKernel_float(float * devPtr, float val, const size_t nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}

__global__ void initKernel_int(int * devPtr, int val, const size_t nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < nwords; tidx += stride)
        devPtr[tidx] = val;
}
 

__inline__ __device__
float warpReduceSum(float val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val)
{
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void deviceReduceKernel(float *in, float* out, int N, int nclusters, int ndims)
{
	//float sum[nclusters] = 0;
	float* sum= (float*)malloc(sizeof(float)*nclusters);
	//
	for (int j = 0; j < ndims; j++){
		//
		for (int i = 0; i < N; i= i+ndims) {
			//reduce multiple elements per thread
			for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
					i < N; 
					i += blockDim.x * gridDim.x) {
				sum[j] += in[i*ndims+ j];
			}

			sum[j] = blockReduceSum(sum[j]);
			if (threadIdx.x==0)
				out[blockIdx.x *ndims +j]=sum[j];	
		}
	}
}

void deviceReduce(float *in, float* out, int N, int nclusters, int ndims)
{
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N, nclusters, ndims);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks, nclusters, ndims);
}

/*
 * data       [nvectors  * ndims]
 * clusters   [nclusters * ndims]
 * membership [nvectors]
 */
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
GPU_sum_clusters(const float *data, const int *membership, int *clusters_members,
		float *clusters_sums, long nvectors, int ndims, int nclusters)
{
	//N might be smaller for clusters
	//int N= nvectors* ndims; 
	int threads = 512;
	//int blocks = min((N + threads - 1) / threads, 1024);

	for (int i = 0; i < nvectors; i++) {
		int cluster = membership[i];

		//int data_num= clusters_members[cluster];

		//for (int j = 0; j < ndims; j++)
		//clusters_sums[cluster * ndims + j] += data[i * ndims + j];
		//for (int j = 0; j < ndims; j++)
		//h_data_clusters[cluster][data_num* ndims + j]= data[i * ndims + j];
		clusters_members[cluster]++;		
	}

	float **d_data_clusters = NULL;
	float **d_data_clusters_arrays = (float **)malloc(nclusters * sizeof(float*));
	if (d_data_clusters_arrays == NULL) {
		fprintf(stderr, "cudamalloc d_data_clusters_arrays error %s\n");
		exit(1);
	}
	cudaError_t err = cudaMalloc(&d_data_clusters, nclusters * sizeof(float*));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_data_clusters_ptrs error %s\n", cudaGetErrorString(err));
		exit(1);
	}

	for (int i = 0; i < nclusters; i++) {
		unsigned int cluster_size = clusters_members[i];
		printf("c[%d] = %u (bytes = %u)\n", i, cluster_size, cluster_size * ndims * sizeof(float));
		err = cudaMalloc(&d_data_clusters_arrays[i], cluster_size * ndims * sizeof(float));
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamalloc d_data_clusters[%d] error %s\n", i, cudaGetErrorString(err));
			exit(1);
		}
	}
	err = cudaMemcpy(d_data_clusters, d_data_clusters_arrays, nclusters * sizeof(float*),
			cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_data_clusters_arrays error %s\n", cudaGetErrorString(err));
		exit(1);
	}

	float **h_data_clusters = (float **)malloc(nclusters * sizeof(float*));

	for (int i = 0; i < nclusters; i++) {
		int cluster_size= clusters_members[i];
		h_data_clusters[i] = (float *)malloc(cluster_size*ndims * sizeof(float));
	}
	/*
	 *    for (int i = 0; i < nvectors; i++) {
	 *        int cluster = membership[i];
	 *
	 *        int data_num= clusters_members[cluster];
	 *
	 *        for (int j = 0; j < ndims; j++)
	 *            clusters_sums[cluster * ndims + j] += data[i * ndims + j];
	 *        for (int j = 0; j < ndims; j++)
	 *            h_data_clusters[cluster][data_num* ndims + j]= data[i * ndims + j];
	 *
	 *        clusters_members[cluster]++;		
	 *    }
	 */
	float **h_out = (float **)malloc(nclusters * sizeof(float*));

	for (int i = 0; i < nclusters; i++) {
		int N = clusters_members[i];
		int blocks = min((N + threads - 1) / threads, 1024);
		h_out[i] = (float *)malloc(blocks *ndims* sizeof(float));
	}

	float **d_out = NULL;
	err = cudaMalloc(d_out, (nclusters)*sizeof(float*));
	if (err != cudaSuccess) {
		cerr << "Error in allocating device output Array!" << endl;
		cout << "Error is: " << cudaGetErrorString(err) << endl;
		//return -1;
	}
	for (int i = 0; i < nclusters; i++) {
		int N = clusters_members[i];
		int blocks = min((N + threads - 1) / threads, 1024);
		err = cudaMalloc(&d_out[i], blocks * ndims * sizeof(float));
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamalloc d_out error %s\n", cudaGetErrorString(err));
			//return 1;
		}
	}
	//copy mem to device
	for(int i = 0; i < nclusters; i++) {
		int cluster_size = clusters_members[i];
		err = cudaMemcpy(d_data_clusters[i], h_data_clusters[i],
				cluster_size * ndims * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy d_data error %s\n", cudaGetErrorString(err));
			exit(1);
		}
	}

	err = cudaMemcpy(d_data_clusters, h_data_clusters, nclusters * sizeof(float*), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy d_data error %s\n", cudaGetErrorString(err));
	}
	//deviceReduce(d_A, d_out, N);
	for (int i = 0; i < nclusters; i++) {
		int N = clusters_members[i];
		deviceReduce(d_data_clusters[i], d_out[i], N*ndims, nclusters, ndims);
		//cout<<"sum"<<d_out[i][0]<<endl;
	}
	//transfer back to host
	for(int i = 0; i < nclusters; i++) {
		int N = clusters_members[i];
		int blocks = min((N + threads - 1) / threads, 1024);
		err = cudaMemcpy(h_out[i], d_out[i], blocks * ndims * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy d_out error %s\n", cudaGetErrorString(err));
			//return 1;
		}
	}

	err = cudaMemcpy(h_out, d_out, nclusters * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < nclusters; i++) {
		for (int j = 0; j < ndims; j++)
			clusters_sums[i * nclusters + j] = h_out[i][j];
		//printf("result %f", h_out[i][0]);
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
#if ONE_VECTOR
	int thread_vectors = 1;
	int block_threads = 125; // 125 as close to 128 as possible
	assert(nvectors % thread_vectors == 0);
	assert((nvectors / thread_vectors) % block_threads == 0);
	int grid_blocks = (nvectors / thread_vectors) / block_threads;
#elif MAX_THREADS || COALESCE
	int grid_blocks = 16;
	int block_threads = 128;
	assert(grid_blocks * block_threads == D_THREADS);
#endif

	for (int iter = 0; iter < niters; iter++) {
		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);

		/*int thread_init= (ndims*nclusters)/256;*/
		/*initKernel_float<<<256,thread_init >>>(d_clusters_sums,0, nclusters*ndims * sizeof(float));*/
		/*thread_init= (nclusters)/2;*/
		/*initKernel_int<<<2,thread_init >>>(d_clusters_members,0, nclusters * sizeof(int));*/

		cudaError_t err = cudaMemcpy(d_clusters, h_clusters, nclusters * ndims * sizeof(float),
				cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(err));
			return -1;
		}

#if CPU_SUM
#if ONE_VECTOR
			kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters);
#elif MAX_THREADS
			int thread_vectors = (nvectors + (D_THREADS - 1)) / D_THREADS;
			kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters, nvectors, thread_vectors);
#elif COALESCE
			kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_membership, ndims, nclusters, nvectors);
#endif /* membership type */
#elif GPU_SUM
#if ONE_VECTOR
			kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters);
#elif MAX_THREADS
			int thread_vectors = (nvectors + (D_THREADS - 1)) / D_THREADS;
			kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters, nvectors, thread_vectors);
#elif COALESCE
			kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
					d_clusters_sums, d_clusters_members, ndims, nclusters, nvectors);
#endif /* membership type */
#endif /* sum type */
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "kmeans_kernel error %s\n", cudaGetErrorString(err));
			return -1;
		}
		cudaDeviceSynchronize();

#if CPU_SUM
		err = cudaMemcpy(h_membership, d_membership, nvectors * sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy h_membership error %s\n", cudaGetErrorString(err));
			return -1;
		}
		cpu_sum_clusters(h_data, h_membership, h_clusters_members,
				h_clusters_sums, nvectors, ndims, nclusters);
#elif GPU_SUM
		err = cudaMemcpy(h_clusters_sums, d_clusters_sums, nclusters*ndims * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy h_clusters_sum error %s\n", cudaGetErrorString(err));
			return -1;
		}	

		err = cudaMemcpy(h_clusters_members, d_clusters_members, nclusters * sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy h_clusters_members error %s\n", cudaGetErrorString(err));
			return -1;
		}
#endif

		for (int i = 0; i < nclusters; i++)
			for (int j = 0; j < ndims; j++)
				h_clusters[i * ndims + j] = h_clusters_sums[i * ndims + j] / h_clusters_members[i];

		clock_gettime(CLOCK_MONOTONIC, &end);
		printf("iter(%d) = %luns\n", iter, time_diff(start, end));
	}

	cudaFree(d_clusters_sums);
	cudaFree(d_clusters_members);

	/*for (int i = 0; i < nclusters; i++) {*/
	/*    printf("clusters[%d]:\n", i);*/
	/*    printf("%f", h_clusters[i * ndims + 0]);*/
	/*    for (int j = 1; j < ndims; j++)*/
	/*        printf(", %f", h_clusters[i * ndims + j]);*/
	/*    printf("\n");*/
	/*}*/
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
int
setup_data(float **h_data, float **d_data, float **h_clusters, float **d_clusters,
		int **h_membership, int **d_membership, int **h_clusters_members, int **d_clusters_members,
		float **h_clusters_sums, float **d_clusters_sums, long nvectors, int ndims, int nclusters,
		const char *infile)
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

	for (int i = 0; i < nclusters; i++)
		for (int j = 0; j < ndims; j++)
			(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];

	err = cudaMalloc(d_clusters, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters error %s\n", cudaGetErrorString(err));
		return -1;
	}

#if CPU_SUM
	*h_membership = (int *)malloc(nvectors * sizeof(int));
	if (*h_membership == NULL) {
		fprintf(stderr, "malloc h_membership failed\n");
		return -1;
	}
	memset(*h_membership, 0, nvectors * sizeof(int)); 
	err = cudaMalloc(d_membership, nvectors * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_membership error %s\n", cudaGetErrorString(err));
		exit(2);
	}
	err = cudaMemset(*d_membership, 0, nvectors * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemset d_membership error %s\n", cudaGetErrorString(err));
		exit(2);
	}
#elif GPU_SUM
	err = cudaMalloc(d_clusters_members, nclusters * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_members error %s\n", cudaGetErrorString(err));
		exit(2);
	}
	err = cudaMemset(*d_clusters_members, 0, nclusters * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemset d_clusters_members error %s\n", cudaGetErrorString(err));
		exit(2);
	}
	err = cudaMalloc(d_clusters_sums, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_clusters_sums error %s\n", cudaGetErrorString(err));
		exit(2);
	}
	err = cudaMemset(*d_clusters_sums, 0, nclusters * ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemset d_clusters_sums error %s\n", cudaGetErrorString(err));
		exit(2);
	}
#endif

	*h_clusters_members = (int *)malloc(nclusters * sizeof(int));
	if (*h_clusters_members == NULL) {
		fprintf(stderr, "malloc h_clusters_members failed\n");
		return -1;
	}
	memset(*h_clusters_members, 0, nclusters * sizeof(int)); 

	*h_clusters_sums = (float *)malloc(nclusters * ndims * sizeof(float));
	if (*h_clusters_sums == NULL) {
		fprintf(stderr, "malloc h_clusters_sums failed\n");
		return -1;
	}
	memset(*h_clusters_sums, 0, nclusters * ndims * sizeof(float)); 

	return 0;
}

int
main(int argc, char *argv[])
{
	if (argc != 6) {
		printf("usage: ./kmeans <infile> <vectors> <dimensions> <clusters> <iterations>\n");
		return 1;
	}

	char *infile  = argv[1];
	// need to be careful with large sizes nvectors * ndims can overflow a signed int
	long nvectors  = strtol(argv[2], NULL, 10);
	int  ndims     = atoi(argv[3]);
	int  nclusters = atoi(argv[4]);
	int  niters    = atoi(argv[5]);
	struct timespec run_start, run_end;
	struct timespec start, end;
	float *h_data, *d_data, *h_clusters, *d_clusters, *h_clusters_sums, *d_clusters_sums;
	int *h_membership, *d_membership, *h_clusters_members, *d_clusters_members;

	clock_gettime(CLOCK_MONOTONIC, &run_start);

	int err = setup_data(&h_data, &d_data, &h_clusters, &d_clusters,
			&h_membership, &d_membership, &h_clusters_members, &d_clusters_members,
			&h_clusters_sums, &d_clusters_sums, nvectors, ndims, nclusters, infile);
	if (err)
		return err;

	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("setup = %luns\n", time_diff(run_start, end));
	clock_gettime(CLOCK_MONOTONIC, &start);

	err = run_kmeans(h_data, d_data, h_clusters, d_clusters, h_membership, d_membership,
			h_clusters_members, d_clusters_members, h_clusters_sums, d_clusters_sums,
			nvectors, ndims, nclusters, niters);
	if (err)
		return err;

	clock_gettime(CLOCK_MONOTONIC, &run_end);
	printf("kmeans = %luns\n", time_diff(start, run_end));
	printf("runtime = %luns\n", time_diff(run_start, run_end));

	/*free(h_data);*/
	/*cudaFree(d_data);*/
	/*free(h_clusters);*/
	/*cudaFree(d_clusters);*/
	/*free(h_membership);*/
	/*cudaFree(d_membership);*/
	/*free(h_clusters_members);*/
	/*free(h_clusters_sums);*/

	cudaDeviceReset();
	return err;
}

