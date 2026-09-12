cmake_minimum_required(VERSION 3.18)
project(FlashGANN CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)
enable_testing()

add_executable(test_turbo unittest/test_turbo.cc)
target_include_directories(test_turbo PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(test_turbo PRIVATE -O3 -march=native)
target_link_libraries(test_turbo PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME test_turbo COMMAND test_turbo)
