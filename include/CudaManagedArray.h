#pragma once
#include <stdexcept>
#include <sstream>

namespace cmem
{
    template <typename T>
    class CudaManagedArray
    {
    public:
        CudaManagedArray() = delete;
        explicit CudaManagedArray(int N);
        ~CudaManagedArray();

        CudaManagedArray(const CudaManagedArray &) = delete;

        T &operator[](int i);

        T *get();

        int size();

        // TODO: move / copy constructors.

    private:
        T *ptr;
        int N;
    };

    template <typename T>
    CudaManagedArray<T>::CudaManagedArray(int N) : N{N}
    {
        cudaMallocManaged(&ptr, N * sizeof(T));
    }

    template <typename T>
    CudaManagedArray<T>::~CudaManagedArray()
    {
        cudaFree(ptr);
    }

    template <typename T>
    T &CudaManagedArray<T>::operator[](int i)
    {
        if (i >= N)
        {
            std::stringstream error;
            error << "Error!  Attempted to access index: " << i << " which is out of bounds of object with size: " << N << "!" << std::endl;
            throw(std::out_of_range(error.str()));
        }
        return this->ptr[i];
    }

    template <typename T>
    T *CudaManagedArray<T>::get()
    {
        return this->ptr;
    }

    template <typename T>
    int CudaManagedArray<T>::size()
    {
        return this->N;
    }
}
