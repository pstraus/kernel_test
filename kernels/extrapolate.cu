#include <add.cuh>
// Kernel function to add the elements of two arrays
__global__ 
void propagate(int n, float *x, float *y, float *out)
{
  for (int i = 0; i < n; i++)
    out[i] = x[i] + y[i];
}
