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

include_directories(${xsimd_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Modified autodiff, to work with xsimd
include_directories(${CMAKE_SOURCE_DIR}/dependencies/autodiff)

add_executable(autodiff_jacobian_with_tuple main.cpp)

install(TARGETS autodiff_jacobian_with_tuple RUNTIME DESTINATION bin)
