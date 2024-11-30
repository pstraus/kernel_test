#include <extrapolate.cuh>
#include <CudaManagedArray.h>
#include <LabelledParticle.h>

#include <gtest/gtest.h>

TEST(TestLabelledParticle, CreateArray)
{
  int N = 1e3;

  EXPECT_NO_THROW(cmem::CudaManagedArray<filters::LabelledParticle<3, 3>> p(N));
}

// TEST(TestLabelledParticle, getPrimaryLabel)
// {

// }