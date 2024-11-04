cmake_minimum_required(VERSION 3.28.3)
project(kernel_test LANGUAGES CXX CUDA)

find_package(Eigen3 CONFIG REQUIRED)

set(CUDA_ARCHITECTURES 75)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)

add_subdirectory(kernels)
add_subdirectory(src)

