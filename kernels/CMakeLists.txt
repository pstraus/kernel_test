cmake_minimum_required(VERSION 3.28.3)
project(kernels LANGUAGES CXX CUDA)

add_library(kernels STATIC add.cu)

target_compile_features(kernels PUBLIC cuda_std_20)

set_target_properties(kernels
                        PROPERTIES
                            CUDA_RUNTIME_LIBRARY SHARED
                            CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(kernels PUBLIC cuda)


