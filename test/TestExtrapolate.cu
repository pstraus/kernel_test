#include <extrapolate.cuh>
#include <CudaManagedArray.h>
#include <Particle.h>

#include <gtest/gtest.h>

TEST(TestExtrapolate, identity_single)
{
  int N = 1e3;

  cmem::CudaManagedArray<filters::Particle<3>> p(N);
  cmem::CudaManagedArray<Eigen::Matrix<float, 3, 3>> F(1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  F[0] << 1, 0, 0,
      0, 1, 0,
      0, 0, 1;

  for (int i = 0; i < N; i++)
  {
    p[i].getState() << 1, 2, 3;
  }

  int numThreads = 256;
  int numBlocks = N / numThreads;
  numBlocks++;
  math::se::extrapolate<<<numBlocks, numThreads>>>(N, p.get(), F.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Make sure the particle is still valid
  for (int i = 0; i < N; i++)
  {
    EXPECT_EQ(p[i].getState()[0], 1) << "Expected: 1, got: " << p[i].getState()[0];
    EXPECT_EQ(p[i].getState()[1], 2) << "Expected: 2, got: " << p[i].getState()[1];
    EXPECT_EQ(p[i].getState()[2], 3) << "Expected: 3, got: " << p[i].getState()[2];
  }
}

TEST(TestExtrapolate, not_identity)
{
  int N = 1e4;

  cmem::CudaManagedArray<filters::Particle<3>> p(N);
  cmem::CudaManagedArray<Eigen::Matrix<float, 3, 3>> F(1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  F[0] << 2, 0, 0,
      0, 3, 0,
      0, 0, 4;

  for (int i = 0; i < N; i++)
  {
    p[i].getState() << 1, 2, 3;
  }

  int numThreads = 256;
  int numBlocks = N / numThreads;
  numBlocks++;
  math::se::extrapolate<<<numBlocks, numThreads>>>(N, p.get(), F.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Make sure the particle is still valid
  for (int i = 0; i < N; i++)
  {
    EXPECT_EQ(p[i].getState()[0], 2) << "Expected: 1, got: " << p[i].getState()[0] * 2;
    EXPECT_EQ(p[i].getState()[1], 6) << "Expected: 2, got: " << p[i].getState()[1] * 3;
    EXPECT_EQ(p[i].getState()[2], 12) << "Expected: 3, got: " << p[i].getState()[2] * 4;
  }
}