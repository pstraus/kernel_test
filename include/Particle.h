#pragma once
#include <Eigen/Core>

namespace filters
{
    template <int N>
    class Particle
    {
    public:
        Particle() = default;
        Particle(const Eigen::Vector<float, N> &state);
        Eigen::Vector<float, N> &getState();

    private:
        Eigen::Vector<float, N> mState;
    };
    template <int N>
    Particle<N>::Particle(const Eigen::Vector<float, N> &state) : mState{state} {
                                                                      // Everthing done by initializer
                                                                  };

    template <int N>
    Eigen::Vector<float, N> &Particle<N>::getState()
    {
        return this->mState;
    };

}
