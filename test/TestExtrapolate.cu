#include <extrapolate.cuh>
#include <CudaManagedArray.h>
#include <Particle.h>

#include <gtest/gtest.h>

TEST(TestExtrapolate, identity_single)
{
  int N = 1;

  cmem::CudaManagedArray<filters::Particle<3>> p(N);
  cmem::CudaManagedArray<Eigen::Matrix<float, 3, 3>> F(1);

  F[0] << 1, 0, 0,
      0, 1, 0,
      0, 0, 1;

  p[0].getState() << 1, 2, 3;

  math::se::extrapolate<<<1, 1>>>(N, p.get(), F.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Make sure the particle is still valid
  EXPECT_EQ(p[0].getState()[0], 1);
  EXPECT_EQ(p[0].getState()[1], 2);
  EXPECT_EQ(p[0].getState()[2], 3);
}