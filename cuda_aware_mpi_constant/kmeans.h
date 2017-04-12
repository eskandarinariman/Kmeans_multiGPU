#ifndef KMEANS_H
#define KMEANS_H

#include <stdint.h>
#include <time.h>

#define FLT_MAX 3.40282347e+38

/* util functions */
uint64_t time_diff(struct timespec start, struct timespec end);
int read_data(float **data, ssize_t size, const char *filename);


/* kmeans functions */
int device_setup_data(float **h_data, float **d_data,
		int **d_membership, long nvectors, int ndims, int nclusters);

int host_setup_data(float **h_data, float **h_clusters, int **h_membership, int **h_clusters_members,
		float **h_clusters_sums, float ** h_clusters_global_sums, int ** h_clusters_global_members,
		long nvectors, int ndims, int nclusters, const char *infile, int world_rank);

int run_kmeans(const float *h_data, const float *d_data, float *h_clusters,
		int *h_membership, int *d_membership,
		int *h_clusters_members, float *h_clusters_sums, long nvectors,
		int ndims, int nclusters, int niters);

#endif /* KMEANS_H */
