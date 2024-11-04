#include <add.cuh>
#include <CudaManagedArray.h>

int main()
{
  int N = 1e6;

  cmem::CudaManagedArray<float> x(N);
  cmem::CudaManagedArray<float> y(N);
  cmem::CudaManagedArray<float> out(N);

  // initialize x and y arrays from the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  int numThreads = 256;
  int numBlocks = N/numThreads;
  numBlocks++; //Integer division would potentially underestimate number of required threads.

  std::cout << "numThreads: " << numThreads <<std::endl << "numBlocks: " << numBlocks << std::endl;


  //Actually add them in parallel
  dim3 threadPerBlock(numThreads, 1, 1);
  kernel_test::add<<<numBlocks, threadPerBlock>>>(N, x.get(), y.get(), out.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = -1.0f;
  float minError = 5.0f;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(out[i] - 3.0f));
    minError = fmin(minError, fabs(out[i] - 3.0f));
  }
  std::cout << "\tMax error: " << maxError << std::endl << "\tMin error: " << minError << std::endl;

  return 0;
}