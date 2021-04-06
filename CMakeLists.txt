cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
project(autodiff_jacobian_with_tuple)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

# External dependenties
find_package(Eigen3 REQUIRED)
find_package(xsimd REQUIRED)
if( ${xsimd_FOUND} )
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}")
endif()

# Modified autodiff, to work with xsimd
include_directories(dependencies)

add_executable(simd_autodiff_benchmark main.cpp)

target_link_libraries(simd_autodiff_benchmark Eigen3::Eigen)
target_link_libraries(simd_autodiff_benchmark xsimd)
