#include <add.cuh>

// Kernel function to add the elements of two arrays
__global__ void kernel_test::add(int N, float *x, float *y, float *out)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  // printf("%i\n",i);

  // prevent bad access.
  if (i < N)
  {
    out[i] = x[i] + y[i];
  }

  return;
}
