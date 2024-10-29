cmake_minimum_required(VERSION 3.28.3)
project(kernel_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED)
add_subdirectory(kernels)

#find_package(CUDA)

add_executable(main)
target_sources(main
               PRIVATE "src/main.cpp"
)

target_include_directories(main 
                            PRIVATE "${CMAKE_SOURCE_DIR}/include"
                            PRIVATE"${CMAKE_SOURCE_DIR}/kernels"
)

