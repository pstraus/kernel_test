project(kernel_test_main)


add_executable(main main.cu)

target_include_directories(main 
                            PRIVATE "${CMAKE_SOURCE_DIR}/include/"
                            PRIVATE "${CMAKE_SOURCE_DIR}/kernels/"
)

target_link_libraries(main PRIVATE kernels)


