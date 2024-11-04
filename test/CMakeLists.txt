project(unitTests)

enable_testing()
find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

find_package(Eigen3 REQUIRED)

add_executable(unitTests)

target_include_directories(unitTests 
                            PUBLIC "${CMAKE_SOURCE_DIR}/include/"
                            PUBLIC "${CMAKE_SOURCE_DIR}/kernels/"
)
target_sources(unitTests
                PUBLIC
                    "TestExtrapolate.cu")

target_link_libraries(unitTests 
                            PRIVATE kernels 
                            PRIVATE Eigen3::Eigen 
                            PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)


gtest_discover_tests(unitTests)