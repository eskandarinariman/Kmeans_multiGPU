#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <thrust/sort.h>

extern "C" {
#include "kmeans.h"
}

using namespace std;


//template<typename float>
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


__global__ void deviceReduceKernel(float *in, float* out, int N, int nclusters, int ndims, int clusters_sort_place,int j_dim, int i_clst)
{
	float sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i +clusters_sort_place*ndims + N*j_dim];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
	
}




void deviceReduce(float *in, float* out, int N, int nclusters, int ndims, int clusters_sort_place,int j_dim, int i_clst)
{
  int threads = 512;
  int blocks = min((N + threads - 1) / threads, 1024);

  deviceReduceKernel<<<blocks, threads>>>(in, out, N, nclusters, ndims, clusters_sort_place, j_dim, i_clst);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks, nclusters, ndims, 0, 0,  0);
}
/*
 * data       [nvectors  * ndims]
 * clusters   [nclusters * ndims]
 * membership [nvectors]
 */
__inline__ __device__ void
vector_dist(unsigned int vector, const float *data, const float *clusters,
		int *membership, int ndims, int nclusters, float *clusters_sums, int * d_clusters_members)
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
	
	atomicAdd(&d_clusters_members[index], 1);
	//__syncthreads();
	//atomic add
	for (int j = 0; j < ndims; j++) {
		//__syncthreads();
			float val = data[vector * ndims + j] ;
			
			atomicAdd(&clusters_sums[index*ndims+ j], val);
		}
}

__global__ void
kmeans_one_vector(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters, float *clusters_sums, int * d_clusters_members)
{
	unsigned int vector = blockIdx.x * blockDim.x + threadIdx.x;
	vector_dist(vector, data, clusters, membership, ndims, nclusters, clusters_sums, d_clusters_members);
}

__global__ void
kmeans_max_threads(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters, int nvectors, int thread_vectors, float * clusters_sums, int * d_clusters_members)
{
	unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * thread_vectors;
	unsigned int end   = start + thread_vectors;
	for (int vector = start; vector < end; vector++) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, membership, ndims, nclusters, clusters_sums, d_clusters_members);
	}
}

__global__ void
kmeans_coalesce(const float *data, const float *clusters, int *membership,
		int ndims, int nclusters, int nvectors, int threads, float *clusters_sums, int * d_clusters_members)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int vector = tid; vector < nvectors; vector+=threads) {
		if (vector < nvectors)
			vector_dist(vector, data, clusters, membership, ndims, nclusters, clusters_sums, d_clusters_members);
	}
}

void
GPU_sum_clusters(const float *data, const int *membership, int *clusters_members,
		float *clusters_sums, long nvectors, int ndims, int nclusters)
{
	//N might be smaller for clusters
	//int N= nvectors* ndims; 
	int threads = 512;
	//int blocks = min((N + threads - 1) / threads, 1024);

	cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
	
	
	
	
	for (int i = 0; i < nvectors; i++) {
		int cluster = membership[i];

		//int data_num= clusters_members[cluster];

		//for (int j = 0; j < ndims; j++)
		//clusters_sums[cluster * ndims + j] += data[i * ndims + j];
		//for (int j = 0; j < ndims; j++)
		//h_data_clusters[cluster][data_num* ndims + j]= data[i * ndims + j];
		clusters_members[cluster]++;		
	}

	
 printf("after \n");
	float *h_data_clusters = (float *)malloc(nvectors*ndims * sizeof(float));
 printf("after 2\n");
	
	 printf("after 3\n");
	 
	int *clusters_sort = (int *)malloc(nclusters * sizeof(int));
	
	for (int i = 0; i < nclusters; i++) {
		if(i==0)
		clusters_sort[i]=0;
	else 
		clusters_sort[i]=clusters_sort[i-1]+clusters_members[i-1];
	
	printf("cluster_sort %d \n", clusters_sort[i]);
	}
	
	//reset number of pixels in clusters_members
	/*for (int i = 0; i < nclusters; i++) {
		clusters_members[i]=0;
	}*/
	int *clusters_members_count= (int *) malloc (nclusters * sizeof(int));
	
	for (int i=0; i<nclusters; i++)
	{
		clusters_members_count[i]=0;
	}
	
	//storing members of distinct clusters in h_data_clusters
	     for (int i = 0; i < nvectors; i++) {
	         int cluster = membership[i];
			 int data_count= clusters_members[cluster];
	         int data_num= clusters_members_count[cluster];
			//printf("data_num %d \n", data_num);
			
			int clusters_sort_place= clusters_sort[cluster];
				// printf("clusters_sort_place %d \n", clusters_sort_place);
			
	         for (int j = 0; j < ndims; j++){
				 
				//printf("check index %d \n", clusters_sort_place*ndims + j*data_count +data_num);
				 h_data_clusters[clusters_sort_place*ndims + j*data_count +data_num]= data[i * ndims + j];
			 }
	            
	         clusters_members_count[cluster]++;		
	     }
	 
	printf("after h_data_cluster\n");
	
	float *d_data_clusters = NULL;
	
	cudaError_t err = cudaMalloc(&d_data_clusters, nvectors*ndims * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamalloc d_data_clusters_ptrs error %s\n", cudaGetErrorString(err));
		exit(1);
	}
printf("before \n");

	
	
	err = cudaMemcpy(d_data_clusters, h_data_clusters, nvectors*ndims * sizeof(float),cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudamemcpy h_data_clusters error %s\n", cudaGetErrorString(err));
		exit(1);
	}
	
	float elapsedTime_GPU, clock=0;
	
	float *d_out = NULL;
	float *h_out; 
cudaEventRecord(start_GPU, 0);
	for (int i_clst = 0; i_clst < nclusters; i_clst++) {
		
		int N = clusters_members[i_clst];
		int blocks = min((N + threads - 1) / threads, 1024);
		int clusters_sort_place= clusters_sort[i_clst];
		printf("cluster members %d ", N);
		
		
		
		for(int j_dim=0; j_dim<ndims; j_dim++)
		{
			err = cudaMalloc(&d_out, (blocks)*sizeof(float));
			if (err != cudaSuccess) {
				cerr << "Error in allocating device output Array!" << endl;
				cout << "Error is: " << cudaGetErrorString(err) << endl;
		
			}
			h_out = (float *)malloc(blocks * sizeof(float));
			deviceReduce(d_data_clusters, d_out, N, nclusters, ndims, clusters_sort_place, j_dim, i_clst);
			err = cudaMemcpy(h_out, d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);
			//printf("result %f \n", h_out[0]);
			
			clusters_sums[i_clst * ndims + j_dim] = h_out[0];
			
			free(h_out);
			cudaFree(d_out);
		}
		
		cudaEventRecord(stop_GPU, 0);
		cudaEventSynchronize(stop_GPU);
		cudaEventElapsedTime(&clock, start_GPU, stop_GPU);
		elapsedTime_GPU += clock;
		
	}
	
	
	
   

	 printf("average time elapsed of GPU:  %.3f\n", elapsedTime_GPU);
	
	printf("after 6\n");
	//transfer back to host  (HOw??????????????)
	free(h_data_clusters);
cudaFree(d_data_clusters);
	free(clusters_sort);
	free(clusters_members_count);
	cudaDeviceSynchronize();
	
}



/*
 * [hd]_data          [nvectors  * ndims]
 * [hd]_clusters      [nclusters * ndims]
 * [hd]_membership    [nvectors]
 * h_clusters_members [nclusters]
 * h_clusters_sums    [nclusters * ndims]
 */
int
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
	//int grid_blocks = 16;
	//int block_threads = 128;
	int threads = grid_blocks * block_threads;
	assert(threads == 2048);
#if MAX_THREADS
	int thread_vectors = (nvectors + (threads - 1))/threads;
#endif
#endif

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	for (int i = 0; i < niters; i++) {
		cudaError_t err = cudaMemcpy(d_clusters, h_clusters, nclusters * ndims * sizeof(float),
				cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy d_clusters error %s\n", cudaGetErrorString(err));
			return -1;
		}
		
		float* d_clusters_sums=NULL;
		 err = cudaMalloc(&d_clusters_sums, nclusters * ndims * sizeof(float));
		 int thread_init= (ndims*nclusters)/256;
				initKernel_float<<<256,thread_init >>>(d_clusters_sums,0, nclusters*ndims * sizeof(float));
				
				
				
				
				int* d_clusters_members=NULL;
		 err = cudaMalloc(&d_clusters_members, nclusters  * sizeof(int));
		  thread_init= (nclusters)/2;
				initKernel_int<<<2,thread_init >>>(d_clusters_members,0, nclusters * sizeof(int));

#ifdef ONE_VECTOR
		kmeans_one_vector<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters, d_clusters_sums, d_clusters_members);
#elif MAX_THREADS
		kmeans_max_threads<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters, nvectors, thread_vectors, d_clusters_sums, d_clusters_members);
#elif COALESCE
		kmeans_coalesce<<<grid_blocks, block_threads>>>(d_data, d_clusters,
				d_membership, ndims, nclusters, nvectors, threads, d_clusters_sums, d_clusters_members);
#endif
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "kmeans_kernel error %s\n", cudaGetErrorString(err));
			return -1;
		}
		cudaDeviceSynchronize();

		//no need
		/*err = cudaMemcpy(h_membership, d_membership, nvectors * sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "cudamemcpy h_membership error %s\n", cudaGetErrorString(err));
			return -1;
		}*/
		
		//cudaFree((float *)d_data);

#if CPU_SUM
		//cpu_sum_clusters(h_data, h_membership, h_clusters_members,
				//h_clusters_sums, nvectors, ndims, nclusters);
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
		
		/*for (int i = 0; i < nvectors; i++) {
		int cluster = h_membership[i];
		h_clusters_members[cluster]++;
		//for (int j = 0; j < ndims; j++)
			//h_clusters_sums[cluster * ndims + j] += h_data[i * ndims + j];
	}*/
	
		cudaFree(d_clusters_sums);
		cudaFree(d_clusters_members);
		//free(h_clusters_sums);
				
#elif GPU_SUM
		GPU_sum_clusters(h_data, h_membership, h_clusters_members,
				h_clusters_sums, nvectors, ndims, nclusters);
#endif

		for (int i = 0; i < nclusters; i++){
			//printf("num of cluster memebrs %d \n", h_clusters_members[i]);
			for (int j = 0; j < ndims; j++){
				h_clusters[i * ndims + j] = h_clusters_sums[i * ndims + j] / h_clusters_members[i];
				
				
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &end);

	for (int i = 0; i < nclusters; i++) {
		printf("clusters[%d]:\n", i);
		printf("%f", h_clusters[i * ndims + 0]);
		for (int j = 1; j < ndims; j++)
			printf(", %f", h_clusters[i * ndims + j]);
		printf("\n");
	}

	printf("runtime = %luns\n", time_diff(start, end));
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
	cudaEvent_t start_CPU, stop_CPU;
    cudaEventCreate(&start_CPU);
    cudaEventCreate(&stop_CPU);
	cudaEventRecord(start_CPU, 0);
	
	for (int i = 0; i < nvectors; i++) {
		int cluster = membership[i];
		clusters_members[cluster]++;
		for (int j = 0; j < ndims; j++)
			clusters_sums[cluster * ndims + j] += data[i * ndims + j];
	}
	
	cudaEventRecord(stop_CPU, 0);
	
	float elapsedTime_CPU;
    cudaEventElapsedTime(&elapsedTime_CPU, start_CPU, stop_CPU);

	 printf("average time elapsed of CPU: %f\n", elapsedTime_CPU);
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

	for (int i = 0; i < nclusters; i++)
		for (int j = 0; j < ndims; j++)
			(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];

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

	float *h_data, *d_data, *h_clusters, *d_clusters, *h_clusters_sums;
	int *h_membership, *d_membership, *h_clusters_members;
	int err = setup_data(&h_data, &d_data, &h_clusters, &d_clusters,
			&h_membership, &d_membership, &h_clusters_members,
			&h_clusters_sums, nvectors, ndims, nclusters, infile);
	if (err)
		return err;

	printf("setup complete running kmeans...\n");

	err = run_kmeans(h_data, d_data, h_clusters, d_clusters, h_membership,
			d_membership, h_clusters_members, h_clusters_sums, nvectors, ndims,
			nclusters, niters);
	if (err)
		return err;

	free(h_data);
	//for gpu reduction u need to fix it
	cudaFree(d_data);
	free(h_clusters);
	cudaFree(d_clusters);
	free(h_membership);
	cudaFree(d_membership);
	free(h_clusters_members);
	free(h_clusters_sums);

	cudaDeviceReset();
	return err;
}

