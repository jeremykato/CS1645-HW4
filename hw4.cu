#include "stdio.h"
#include "stdlib.h"
#include <math.h>

#define FUNCTION(x) ((4.0) / (1.0 + (pow(x, 2.0))))
#define DOMAIN_START 0.0
#define DOMAIN_END 1.0


__global__ void trap_appx(int trapezoids, double dx, double *global_sum) {
  // Get our global thread ID
  __shared__ double shared_sum;
  shared_sum = 0;
  double local_sum = 0.0;
  int id = (blockIdx.x * blockDim.x + threadIdx.x);

  if (id >= trapezoids) {
    return;
  }
  double x = id * dx;
  
  local_sum += ( FUNCTION(x) + FUNCTION((x + dx))) * 0.5 * dx;

  atomicAdd(&shared_sum, local_sum);
  __syncthreads();

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

  int threads = atoi(argv[1]);
  if (threads > 1024) {
    printf("Error: max of 1024 threads.\n");
    return -1;
  }
  else if (threads < 1) {
    printf("Error: min of 1 thread\n");
    return -1;
  }

  int trapezoids = 1000000;
  int total_blocks = trapezoids / threads;
  if (trapezoids % threads > 0) {
    total_blocks++;
  }

  double incr = (DOMAIN_END - DOMAIN_START) / ((double) trapezoids);


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

  trap_appx <<< total_blocks, threads >>> (trapezoids, incr, d_sum);
  cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaEventRecord(after);
  cudaEventElapsedTime(&time, before, after);

  printf("Final result is: %2.8f\n", *h_sum);
  printf("Time taken: %2.6f\n", time);


  return 0;
}


