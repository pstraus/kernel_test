#include <iostream>
#include <math.h>
#include "add.cuh"
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y, float *out)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}
