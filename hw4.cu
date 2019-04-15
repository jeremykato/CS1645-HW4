#include "stdio.h"
#include "stdlib.h"
#include <math.h>

#define FUNCTION(x) ((4.0) / (1.0 + (pow(x, 2.0))))
#define DOMAIN_START 0.0
#define DOMAIN_END 1.0


__global__ void trap_appx(int iters, int max_iters, double dx, double *global_sum) {
  // Get our global thread ID
  __shared__ double shared_sum;
  shared_sum = 0;
  double local_sum = 0.0;
  int id = (blockIdx.x * blockDim.x + threadIdx.x) * iters;
  double x = id * dx;
  double p1 = 0.0;
  double p2 = FUNCTION(x);
  __syncthreads();

  int i;
  for (i = id; (i < (id + iters)) && (i < max_iters); i++) {
    x += dx;
    p1 = p2;
    p2 = FUNCTION(x);
    local_sum += ((p2 + p1) / 2) * dx;
  }

  //printf("%d's local sum is %2.6f\n", id, local_sum);
  atomicAdd(&shared_sum, local_sum);
  __syncthreads();
  //printf("%d's shared sum is %2.6f\n", id, shared_sum);

  if (threadIdx.x == 0) {
    atomicAdd(global_sum, shared_sum);
  }
  __syncthreads();

}

int main(int argc, char *argv[]) {
  
  if (argc != 2) {
    printf("Usage: ./hw4 <Number of Threads>\n");
    return -1;
  }

  int total_threads = atoi(argv[1]);
  int grid_size = total_threads / 256;
  if (total_threads % 256 > 0) {
    total_threads++;
  }

  int trapezoids = 1000000;
  int iters_per_thread = trapezoids / total_threads;
  if (trapezoids % total_threads > 0) {
    iters_per_thread++;
  }

  double incr = (DOMAIN_END - DOMAIN_START) / ((double) trapezoids);

  if (total_threads < 1) {
    printf("Error: number of threads must be greater than zero.\n");
    return -1;
  }

  double *h_sum = (double *) calloc(1, sizeof(double));
  double *d_sum = NULL;
  cudaMalloc(&d_sum, (size_t) sizeof(double));
  cudaMemcpy(d_sum, h_sum, sizeof(double), cudaMemcpyHostToDevice);

  // Cuda time
  float time;
  cudaEvent_t before, after;
  cudaEventCreate(&before);
  cudaEventCreate(&after);

  cudaEventRecord(before);

  trap_appx <<< grid_size, 256 >>> (iters_per_thread, trapezoids, incr, d_sum);
  cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaEventRecord(after);
  cudaEventElapsedTime(&time, before, after);

  printf("Final result is: %2.8f\n", *h_sum);
  printf("Time taken: %2.6f\n", time);


  return 0;
}


