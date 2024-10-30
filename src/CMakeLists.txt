project(kernel_test LANGUAGES CXX CUDA)

#find_package(CUDA)

add_executable(main)

target_link_libraries(main
                        PRIVATE kernels
)


target_sources(main
               PRIVATE "main.cpp"
)

include_directories(main 
                            PRIVATE "${CMAKE_SOURCE_DIR}/include"
                            PRIVATE "${CMAKE_SOURCE_DIR}/kernels"
)

