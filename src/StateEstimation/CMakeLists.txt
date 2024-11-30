project(StateEstimation)

add_library(StateEstimation StateEstimation.cpp)

target_include_directories(StateEstimation 
                            PRIVATE "${CMAKE_SOURCE_DIR}/include/"
)

target_link_libraries(StateEstimation PUBLIC Eigen3::Eigen)


