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
#include <time.h>
#include <assert.h>
#include "kmeans.h"

// int run_kmeans(const float *h_data, const float *d_data, float *h_clusters,
//     float *d_clusters, int *h_membership, int *d_membership,
//     int *h_clusters_members, float *h_clusters_sums, long nvectors,
//     int ndims, int nclusters, int niters);

// void
// cpu_sum_clusters(const float *data, const int *membership, int *clusters_members,
//     float *clusters_sums, long nvectors, int ndims, int nclusters);


// int
// setup_data(float **h_data, float **d_data, float **h_clusters, float **d_clusters,
//     int **h_membership, int **d_membership, int **h_clusters_members,
//     float **h_clusters_sums, long nvectors, int ndims, int nclusters, const char *infile);



int main(int argc, char** argv) {


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

  int i;
  float *h_data, *d_data, *h_clusters, *d_clusters, *h_clusters_sums;
  int *h_membership, *d_membership, *h_clusters_members;

  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // should be .cu file
  int err = setup_data(&h_data, &d_data, &h_clusters, &d_clusters,
      &h_membership, &d_membership, &h_clusters_members,
      &h_clusters_sums, nvectors, ndims, nclusters, infile);
  if (err)
    return err;

  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
 

  printf("setup complete running kmeans...\n");

 
  err = run_kmeans(h_data, d_data, h_clusters, d_clusters, h_membership,
      d_membership, h_clusters_members, h_clusters_sums, nvectors, ndims,
      nclusters, niters);
  if (err)
    return err;


  // int num_of_ints = 5;
  // int * s_buf = (int *) malloc(num_of_ints);
  // for(i = 0 ; i < num_of_ints ;i++){
  //   s_buf[i] = i;
  // }


  // if(world_rank == 0){
  //   // printf("rank 0 sends buffer to other ranks...\n");
  //   // MPI_Send(s_buf,num_of_ints,MPI_INT,1,0,MPI_COMM_WORLD);
  //   //  MPI_Send(s_buf,num_of_ints,MPI_INT,2,1,MPI_COMM_WORLD);
  //   //   MPI_Send(s_buf,num_of_ints,MPI_INT,3,2,MPI_COMM_WORLD);

  //   kernel_call(world_rank);

  // }
  // else if(world_rank == 1){
  //   // int * r_buf = (int *) malloc(num_of_ints);
  //   // MPI_Recv(r_buf,num_of_ints,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  //   // printf(" from rank 1 %d\n",r_buf[world_rank]);
  //   kernel_call(world_rank);
  // }
  // else if(world_rank == 2){
  //   // int * r_buf = (int *) malloc(num_of_ints);
  //   // MPI_Recv(r_buf,num_of_ints,MPI_INT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  //   // printf(" from rank 2 %d\n",r_buf[world_rank]);
  //   kernel_call(world_rank);
  // }
  // else if(world_rank == 3){
  //   // int * r_buf = (int *) malloc(num_of_ints);
  //   // MPI_Recv(r_buf,num_of_ints,MPI_INT,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  //   // printf(" from rank 3 %d\n",r_buf[world_rank]);
  //   kernel_call(world_rank);
  // }




  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
    printf("****************************\nKernel is running on processor %s, rank %d out of %d processors\n",
        processor_name, world_rank, world_size);

  // should be on .cu file
  // err = free_data(&h_data, &d_data, &h_clusters, &d_clusters,
  //     &h_membership, &d_membership, &h_clusters_members,
  //     &h_clusters_sums);
  // if (err)
  //   return err;

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}


