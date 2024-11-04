project(test)

enable_testing()
include(GoogleTest)

find_package(Eigen3 REQUIRED)

add_executable(test)

target_include_directories(test 
                            PRIVATE "${CMAKE_SOURCE_DIR}/include/"
                            PRIVATE "${CMAKE_SOURCE_DIR}/kernels/"
)
target_sources(test
                PUBLIC
                    "TestExtrapolate.cu")

target_link_libraries(test 
                            PRIVATE kernels 
                            PRIVATE Eigen3::Eigen 
                            PRIVATE gtest gtrest_main pthread)


