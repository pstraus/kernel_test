project(kernels LANGUAGES CXX CUDA)

add_library(kernels STATIC add.cu add.cuh)

include_directories(kernels PUBLIC 
                    "${CMAKE_SOURCE_DIR}/kernels/"
                    "${CMAKE_SOURCE_DIR}/include/")

target_compile_features(kernels PUBLIC cuda_std_20)

set_target_properties(kernels
                        PROPERTIES
                            CUDA_RUNTIME_LIBRARY Shared
                            CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(kernels 
                        PUBLIC Eigen3::Eigen)


