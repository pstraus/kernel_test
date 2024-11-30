#pragma once

#include <iostream>
#include <math.h>
#include <Particle.h>
#include <Eigen/Core>

// Kernel function to add the elements of two arrays
namespace math::se
{
    template <int N>
    __global__ void extrapolate(int n, filters::Particle<N> *p, Eigen::Matrix<float, N, N> *F)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < n)
        {
            p[i].getState() = *F * p[i].getState();
        }
    }
}
