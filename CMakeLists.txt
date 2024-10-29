cmake_minimum_required(VERSION 3.28.3)
project(particle_filter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD REQUIRED)
add_subdirector(kernels)

find_package(CUDA)

add_executable(main)
target_sources(main
               PRIVATE "src/main.cpp"
)

target_include_directories(main 
                            "${CMAKE_SOURCE_DIR}/include
                            "${CMAKE_SOURCE_DIR}/kernels
)

