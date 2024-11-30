#pragma once
#include <Eigen/Core>

namespace filters
{
    template <int N>
    class Particle
    {
    public:
        __host__ Particle() = default;
        __host__ Particle(const Eigen::Vector<float, N>& state, const Eigen::Matrix<float, N, N>& covariance);
        __host__ __device__ Eigen::Vector<float, N>& getState();
        __host__ __device__ Eigen::Matrix<float, N, N>& getCovariance();

    protected:
        Eigen::Vector<float, N> mState;
        Eigen::Matrix<float, N, N> mCovariance;
    };
    template <int N>
    __host__ Particle<N>::Particle(const Eigen::Vector<float, N> &state, const Eigen::Matrix<float, N, N>& covariance) : mState{state}, mCovariance(covariance) {
                                                                               // Everthing done by initializer
                                                                           };

    template <int N>
    __host__ __device__ Eigen::Vector<float, N>& Particle<N>::getState()
    {
        return this->mState;
    };

        template <int N>
    __host__ __device__ Eigen::Matrix<float, N, N>& Particle<N>::getCovariance()
    {
        return this->mCovariance;
    };

}
