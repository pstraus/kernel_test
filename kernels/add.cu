#include <add.cuh>
// Kernel function to add the elements of two arrays
__global__ void add(int N, float *x, float *y, float *out)
{
  int i = threadIdx.x;

  //prevent bad access.
  if(i < N)
  {
    out[i] = x[i] + y[i];
  }
}
