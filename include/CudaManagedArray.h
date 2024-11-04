#pragma once

namespace cmem
{
template <typename T>
class CudaManagedArray
{
    public: 
        CudaManagedArray() = delete;
        explicit CudaManagedArray(int N);
        ~CudaManagedArray();

        CudaManagedArray(const CudaManagedArray& ) = delete;

        T& operator[](int i);

        T* get();

        //TODO: move / copy constructors.

    private:
        T* ptr;

};


template <typename T>
CudaManagedArray<T>::CudaManagedArray(int N)
{
    cudaMallocManaged(&ptr,   N * sizeof(T));
}

template <typename T>
CudaManagedArray<T>::~CudaManagedArray()
{
    cudaFree(ptr);
}

template <typename T>
T& CudaManagedArray<T>::operator[](int i)
{
    return this->ptr[i];
}

template <typename T>
T* CudaManagedArray<T>::get()
{
    return this->ptr;
}
}
