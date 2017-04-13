// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// An intro MPI hello world program that uses MPI_Init, MPI_Comm_size,
// MPI_Comm_rank, MPI_Finalize, and MPI_Get_processor_name.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
/*#include <stdint.h>*/
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "kmeans.h"

char* new_filename(char * infile,int i);

int
main(int argc, char** argv)
{
	if (argc != 6) {
		printf("usage: ./kmeans <infile> <vectors> <dimensions> <clusters> <iterations>\n");
		return 1;
	}

	int i, j, iter;
	char *infile   = argv[1];
	// need to be careful with large sizes nvectors * ndims can overflow a signed int
	long nvectors  = strtol(argv[2], NULL, 10);
	int  ndims     = atoi(argv[3]);
	int  nclusters = atoi(argv[4]);
	int  niters    = atoi(argv[5]);

	float *h_data, *d_data, *h_clusters, *h_clusters_local_sums, *h_clusters_global_sums;
	int *h_membership, *d_membership, *h_clusters_local_members,*h_clusters_global_members;

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


	int err = host_setup_data(&h_data, &h_clusters, &h_membership, &h_clusters_local_members,
			&h_clusters_local_sums,&h_clusters_global_sums, &h_clusters_global_members,
			nvectors, ndims, nclusters, infile, world_rank);
	if (err)
		return err;

#ifdef REMOTE_DATA
	if(world_rank == 0){
		for(i = 1; i < world_size ;i++){
			int error_code = MPI_Send(h_data,nvectors * ndims , MPI_FLOAT,i,i,MPI_COMM_WORLD);
			if (error_code != MPI_SUCCESS) {

   				char error_string[BUFSIZ];
   				int length_of_error_string;

   				MPI_Error_string(error_code, error_string, &length_of_error_string);
   				fprintf(stderr, "%3d: %s\n", world_rank, error_string);
			}
			int errr = read_data(&h_data, nvectors * ndims * sizeof(float), new_filename(infile,i));
			if (errr) {
				fprintf(stderr, "read_data error: %d\n", errr);
				return 1;
			}
		}
	}
	else{
		int error_code = MPI_Recv(h_data, nvectors * ndims, MPI_FLOAT,0,world_rank,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		if (error_code != MPI_SUCCESS) {
   			char error_string[BUFSIZ];
   			int length_of_error_string;
   			MPI_Error_string(error_code, error_string, &length_of_error_string);
   			fprintf(stderr, "%3d: %s\n", world_rank, error_string);
		}
	}
#endif



//***********************************************************************************

	err = device_setup_data(&h_data, &d_data,
			&d_membership, nvectors, ndims, nclusters);
	if (err)
		return err;

	printf("setup complete running kmeans...\n");

	// Initialize the MPI environment. The two arguments to MPI Init are not
	// currently used by MPI implementations, but are there in case future
	// implementations might need the arguments.

	for(iter = 0; iter < niters; iter++) {
		struct timespec iter_start, iter_end;
		clock_gettime(CLOCK_MONOTONIC, &iter_start);

		MPI_Bcast(h_clusters, nclusters *ndims, MPI_FLOAT, 0, MPI_COMM_WORLD);

		err = run_kmeans(h_data, d_data, h_clusters, h_membership, d_membership,
				h_clusters_local_members, h_clusters_local_sums, nvectors, ndims, nclusters, niters);
		if (err)
			return err;

		MPI_Reduce(h_clusters_local_sums, h_clusters_global_sums,nclusters * ndims, MPI_FLOAT,
				MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(h_clusters_local_members, h_clusters_global_members,nclusters, MPI_FLOAT,
				MPI_SUM, 0, MPI_COMM_WORLD);

		if(world_rank == 0) {
			for (i = 0; i < nclusters; i++)
				for (j = 0; j < ndims; j++)
					h_clusters[i * ndims + j] = h_clusters_global_sums[i * ndims + j] / h_clusters_global_members[i];
		}

		clock_gettime(CLOCK_MONOTONIC, &iter_end);
		printf("iter(%d) = %luns\n", iter, time_diff(iter_start, iter_end));
	}

	// if(world_rank == 0) {
	// 	for (i = 0; i < nclusters; i++) {
	// 		printf("clusters[%d]:\n", i);
	// 		printf("%f", h_clusters[i * ndims + 0]);
	// 		for (j = 1; j < ndims; j++)
	// 			printf(", %f", h_clusters[i * ndims + j]);
	// 		printf("\n");
	// 	}
	// }

	clock_gettime(CLOCK_MONOTONIC, &end);
	printf("total runtime = %luns\n", time_diff(start, end));

	MPI_Finalize();
	return 0;
}

char* new_filename(char * infile,int i){
	size_t len = strlen(infile);
	char *str2 = malloc(len + 1 + 1 );
	strcpy(str2, infile);
    str2[len] = i + '0';
    str2[len + 1] = '\0';
    return str2;
}