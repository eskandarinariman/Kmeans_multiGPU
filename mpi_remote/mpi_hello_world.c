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


int
run_kmeans(const float *h_data, float *h_clusters,
  int *h_membership, int *h_clusters_members, 
  float *h_clusters_sums, long nvectors,
  int ndims, int nclusters, int niters)
{
  int d,i,j;

  for(d = 0 ; d < nvectors ;d++){
    //printf("data % d\n",d);
    int index = -1;
    float min_dist = FLT_MAX;
    for (i = 0; i < nclusters; i++) {
      float dist = 0.0;

      for (j = 0; j < ndims; j++) {
        float diff = h_data[d * ndims + j] - h_clusters[i * ndims + j];
        dist += diff * diff;
      }

      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
    h_membership[d] = index;
    h_clusters_members[index]++;
    for (j = 0; j < ndims; j++)
      h_clusters_sums[index * ndims + j] += h_data[d * ndims + j];
  }
  return 0;
}

int 
host_setup_data(float **h_data, float **h_clusters,
  int **h_membership, int **h_clusters_members,
  float **h_clusters_sums, float ** h_clusters_global_sums, 
  int ** h_clusters_global_members, long nvectors, int ndims,
  int nclusters, const char *infile, int world_rank)
{

  int i,j;

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

  *h_clusters = (float *)malloc(nclusters * ndims * sizeof(float));
  if (*h_clusters == NULL) {
    fprintf(stderr, "malloc h_clusters failed\n");
    return 1;
  }

  if(world_rank == 0){
    for (i = 0; i < nclusters; i++)
      for (j = 0; j < ndims; j++)
        (*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];
    }

    *h_membership = (int *)malloc(nvectors * sizeof(int));
    if (*h_membership == NULL) {
      fprintf(stderr, "malloc h_membership failed\n");
      return -1;
    }

    for (i = 0; i < nclusters; i++)
      (*h_membership)[i] = 0;


    *h_clusters_members = (int *)malloc(nclusters * sizeof(int));
    if (*h_clusters_members == NULL) {
      fprintf(stderr, "malloc h_clusters_members failed\n");
      return -1;
    }

    for (i = 0; i < nclusters; i++)
      (*h_clusters_members)[i] = 0;

    if(world_rank == 0){

      *h_clusters_global_members = (int *)malloc(nclusters * sizeof(int));
      if (*h_clusters_global_members == NULL) {
        fprintf(stderr, "malloc h_clusters_members failed\n");
        return -1;
      }
  
      for (i = 0; i < nclusters; i++)
        (*h_clusters_global_members)[i] = 0;
    }

    *h_clusters_sums = (float *)malloc(nclusters * ndims * sizeof(float));
    if (*h_clusters_sums == NULL) {
      fprintf(stderr, "malloc h_clusters_sums failed\n");
      return -1;
    }

    for (i = 0; i < nclusters; i++)
      (*h_clusters_sums)[i] = 0;


    if(world_rank == 0){
      *h_clusters_global_sums = (float *)malloc(nclusters * ndims * sizeof(float));
      if (*h_clusters_global_sums == NULL) {
        fprintf(stderr, "malloc h_clusters_sums failed\n");
        return -1;
      }
  
      for (i = 0; i < nclusters; i++)
        (*h_clusters_global_sums)[i] = 0;
    }

    return 0;
  }




  int main(int argc, char** argv) {


    if (argc != 6) {
      printf("usage: ./kmeans <infile> <vectors> <dimensions> <clusters> <iterations>\n");
      return 1;
    }
    int i,j,itr;
    char *infile  = argv[1];
  // need to be careful with large sizes nvectors * ndims can overflow a signed int
    long nvectors  = strtol(argv[2], NULL, 10);
    int  ndims     = atoi(argv[3]);
    int  nclusters = atoi(argv[4]);
    int  niters    = atoi(argv[5]);

    float *h_data, *d_data, *h_clusters, *d_clusters, *h_clusters_local_sums, *h_clusters_global_sums;
    int *h_membership, *d_membership, *h_clusters_local_members,*h_clusters_global_members;

    MPI_Init(NULL, NULL);

  // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // should be .cu file
    int err = host_setup_data(&h_data, &h_clusters,
      &h_membership, &h_clusters_local_members,
      &h_clusters_local_sums,&h_clusters_global_sums, &h_clusters_global_members, nvectors, ndims, nclusters, infile, world_rank);
    if (err)
      return err;


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

  // if(world_rank == 0){ 
  //   printf("befor Bcast ... rank : %d \n",world_rank);
  //   for (i = 0; i < nclusters; i++)
  //     for (j = 0; j < ndims; j++)
  //       printf("%f ",h_clusters[i*ndims+j]);
  // }

  // printf(" after Bcast ... rank : %d \n",world_rank);
  //   for (i = 0; i < nclusters; i++)
  //     for (j = 0; j < ndims; j++)
  //       printf("%f ",h_clusters[i*ndims+j]);

    // int err = setup_data(&h_data, &d_data, &h_clusters, &d_clusters,
    //   &h_membership, &d_membership, &h_clusters_members,
    //   &h_clusters_sums, nvectors, ndims, nclusters, infile);
    // if (err)
    //   return err;

  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.

  for(itr = 0 ; itr < niters ;itr++){    
    MPI_Bcast(h_clusters, nclusters *ndims, MPI_FLOAT, 0, MPI_COMM_WORLD);

    printf("setup complete running kmeans...\n");


// for (i = 0; i < nclusters; i++)
//    for (j = 0; j < ndims; j++)
//      //(*h_clusters)[i * ndims + j] = (*h_data)[i * ndims + j];
//      printf("%f ",(h_clusters)[i * ndims + j]);


    err = run_kmeans(h_data, h_clusters, h_membership,
      h_clusters_local_members, h_clusters_local_sums, 
      nvectors, ndims, nclusters, niters);
    if (err)
      return err;



    MPI_Reduce(h_clusters_local_sums, h_clusters_global_sums,nclusters * ndims, MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(h_clusters_local_members, h_clusters_global_members,nclusters, MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);

    if(world_rank == 0){
      for (i = 0; i < nclusters; i++)
        for (j = 0; j < ndims; j++)
          h_clusters[i * ndims + j] = h_clusters_global_sums[i * ndims + j] / h_clusters_global_members[i];
    }

  }

  if(world_rank == 0){
    for (i = 0; i < nclusters; i++) {
      printf("clusters[%d]:\n", i);
      printf("%f", h_clusters[i * ndims + 0]);
      for (j = 1; j < ndims; j++)
        printf(", %f", h_clusters[i * ndims + j]);
      printf("\n");
    }
  }
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
  //  printf("****************************\nKernel is running on processor %s, rank %d out of %d processors\n",
  //      processor_name, world_rank, world_size);

  // should be on .cu file
  // err = free_data(&h_data, &d_data, &h_clusters, &d_clusters,
  //     &h_membership, &d_membership, &h_clusters_members,
  //     &h_clusters_sums);
  // if (err)
  //   return err;

  // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
  }

  char* new_filename(char * infile,int i){
  size_t len = strlen(infile);
  char *str2 = malloc(len + 1 + 1 );
  strcpy(str2, infile);
    str2[len] = i + '0';
    str2[len + 1] = '\0';
    return str2;
}



