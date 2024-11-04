#include <extrapolate.cuh>
#include <CudaManagedArray.h>
#include <Particle.h>

#include <gtest/gtest.h>

TEST(TestExtrapolate, identity_single)
{
  int N = 1e6;

  cmem::CudaManagedArray<filters::Particle<3>> p(N);
  cmem::CudaManagedArray<Eigen::Matrix<float, 3, 3>> F(1);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  F[0] << 1, 0, 0,
      0, 1, 0,
      0, 0, 1;

  p[0].getState() << 1, 2, 3;

  int numThreads = 256;
  int numBlocks = N / numThreads;
  numBlocks++;
  math::se::extrapolate<<<numBlocks, numThreads>>>(N, p.get(), F.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Make sure the particle is still valid
  EXPECT_EQ(p[0].getState()[0], 1) << "Expected: 1, got: " << p[0].getState()[0];
  EXPECT_EQ(p[0].getState()[1], 2) << "Expected: 2, got: " << p[0].getState()[1];
  EXPECT_EQ(p[0].getState()[2], 3) << "Expected: 3, got: " << p[0].getState()[2];
}