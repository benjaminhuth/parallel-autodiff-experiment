cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
project(autodiff_jacobian_with_tuple)

add_executable(autodiff_jacobian_with_tuple main.cpp)

install(TARGETS autodiff_jacobian_with_tuple RUNTIME DESTINATION bin)
