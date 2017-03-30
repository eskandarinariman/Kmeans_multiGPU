/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         mpi_read.c                                                */
/*   Description:  This program reads data points from a file using MPI-IO   */
/*                 that implements a simple k-means clustering algorithm     */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include "kmeans.h"

#include <mpi.h>
#include "kmeans.h"

#define DIMS 256
#define VECTORS 16


/*---< mpi_read() >----------------------------------------------------------*/
int read_data(float data[VECTORS][DIMS], char *filename)
{
    int infile;
    long long bytes_left = VECTORS*DIMS*sizeof(float);
    ssize_t offset = 0;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1){
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return 1;
    }
    while (bytes_left > 0) {
        printf("%d\n", bytes_left);
        // cast to char * since offset in bytes not float
        ssize_t bytes = read(infile, ((char *)data) + offset, bytes_left);
        printf("%d\n", bytes);
        if (bytes == -1) {
            perror("read");
            return 1;
        }
        offset     += bytes;
        bytes_left -= bytes;
    }
    close(infile);
    return bytes_left;
}