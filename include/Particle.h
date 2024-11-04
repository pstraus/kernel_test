#pragma once
#include <Eigen/Core>

namespace filters
{
    template <int N>
    class Particle
    {
    public:
        __host__ Particle() = default;
        __host__ Particle(const Eigen::Vector<float, N> &state);
        __host__ __device__ Eigen::Vector<float, N> &getState();

    private:
        Eigen::Vector<float, N> mState;
    };
    template <int N>
    __host__ Particle<N>::Particle(const Eigen::Vector<float, N> &state) : mState{state} {
                                                                               // Everthing done by initializer
                                                                           };

    template <int N>
    __host__ __device__ Eigen::Vector<float, N> &Particle<N>::getState()
    {
        return this->mState;
    };

}
