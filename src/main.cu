#include <add.cuh>

int main()
{
 int N = 1000000;
  float *x, *y, *out;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));
  cudaMallocManaged(&out, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  int numBlocks = N/256;
  numBlocks++; //Integer division would potentially underestimate number of required threads.

  //Actually add them in parallel
  add<<<numBlocks, 256>>>(N, x, y, out);



  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  float minError = 3.0f;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(out[i] - 3.0f));
    minError = fmin(minError, fabs(out[i] - 3.0f));
  }
  std::cout << "\tMax error: " << maxError << std::endl << "\tMin error: " << minError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(out);

  return 0;
}