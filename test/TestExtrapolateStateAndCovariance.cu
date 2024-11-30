#include <extrapolateStateAndCovariance.cuh>
#include <CudaManagedArray.h>
#include <Particle.h>

#include <gtest/gtest.h>

TEST(extrapolateStateAndCovariance, identity_single)
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
    p[i].getCovariance() << 2, 0, 0,
                            0, 3, 0,
                            0, 0, 4;
  }

  int numThreads = 256;
  int numBlocks = N / numThreads;
  numBlocks++;
  math::se::extrapolateStateAndCovariance<<<numBlocks, numThreads>>>(N, p.get(), F.get());

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Make sure the particle is still valid
  for (int i = 0; i < N; i++)
  {
    EXPECT_FLOAT_EQ(p[i].getState()[0], 1) << "Expected: 1, got: " << p[i].getState()[0];
    EXPECT_FLOAT_EQ(p[i].getState()[1], 2) << "Expected: 2, got: " << p[i].getState()[1];
    EXPECT_FLOAT_EQ(p[i].getState()[2], 3) << "Expected: 3, got: " << p[i].getState()[2];

    //verify covariance
    for(int row= 0; row < 3; row++)
    {
      for(int col = 0; col < 3; col++)
      {
        if(row == col)
        {
          if(row == 0)
          {
          EXPECT_FLOAT_EQ(p[i].getCovariance()(row, col), 2.0);
          }
          else if(row ==1)
          {
          EXPECT_FLOAT_EQ(p[i].getCovariance()(row, col), 3.0);
          }
          else if(row ==2)
          {
          EXPECT_FLOAT_EQ(p[i].getCovariance()(row, col), 4.0);
          }
        }
        else
        {
          EXPECT_FLOAT_EQ(p[i].getCovariance()(row, col), 0.0);
        }
      }
    }
  }


}

TEST(extrapolateStateAndCovariance, not_identity)
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
  math::se::extrapolateStateAndCovariance<<<numBlocks, numThreads>>>(N, p.get(), F.get());

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