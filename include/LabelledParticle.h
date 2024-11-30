#pragma once
#include <Eigen/Core>
#include <include/Particle.h>
#include <include/CudaManagedArray.h>

namespace filters
{
    template <int N, int M>
    class LabelledParticle : public Particle<N>
    {
    public:
        __host__ __device__ LabelledParticle() = default;
        __host__ __device__ ~LabelledParticle() = default;
        __host__ __device__ LabelledParticle(const Eigen::Vector<float, N>& state, int lbl);
        __host__ __device__ LabelledParticle(const Particle<N>& Particle, int lbl);
        __host__ __device__ LabelledParticle(const LabelledParticle<N,M>& lp); // Deep copy constructor
        __host__ __device__ LabelledParticle(LabelledParticle&& lp); // Move constructor

        __host__ __device__ int getPrimaryLabel() const;
        __host__ __device__ CudaManagedArray<int>& getContributors();   

        __host__ __device__ bool addContributor(int contributor);   

    private:
        int mPrimaryLabel;
        CudaManagedArray<int> mContributors;
    };
    template <int N, int M>
    __host__ __device__ LabelledParticle<N, M>::LabelledParticle(const Eigen::Vector<float, N>& state, int lbl) : mParticle{state}, mPrimaryLabel{lbl}
    {
        // Initialize contributors as empty
        for (int i = 0; i < M; i++)
        {
            mContributors[i] = -1;
        }
    };

    template <int N, int M>
    __host__ __device__ LabelledParticle<N, M>::LabelledParticle(const Particle<N>& particle, int lbl) : mParticle{particle}, mPrimaryLabel{lbl}
    {
        // Initialize contributors as empty
        for (int i = 0; i < M; i++)
        {
            mContributors[i] = -1;
        }
    };

        template <int N, int M>
    __host__ __device__ LabelledParticle<N, M>::LabelledParticle(const LabelledParticle<N, M>& lp) : mState{lp.getState()}, mPrimaryLabel{lp.getPrimaryLabel()}
    {
        // Initialize contributors as empty
        for (int i = 0; i < M; i++)
        {
            mContributors[i] = lp.mContributors[i]
        }
    };

    template <int N, int M>
    __host__ __device__ int LabelledParticle<N, M>::getPrimaryLabel() const
    {
        return mPrimaryLabel;
    };

    template <int N, int M>
    __host__ __device__ int* LabelledParticle<N, M>::getContributors()
    {
        return mContributors;
    };

        template <int N, int M>
    __host__ __device__ bool LabelledParticle<N, M>::addContributor(int contributor)
    {
        //Find first entry that isn't -1
        int i = 0 ; 
        while(i < M && i < 0)
        {
            i++;
        }

        // we can't add a new label and need to prevent bad access
        if(i>M-1)
        {
            return false;
        }

        mContributors[i] = contributor;
        return true;
    };

}
