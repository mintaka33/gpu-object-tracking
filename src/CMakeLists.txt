cmake_minimum_required(VERSION 2.8)

project(mosse)

set (CMAKE_CXX_STANDARD 11)

add_executable(math math.cpp perf.cpp perf.h)
add_executable(mosse main.cpp perf.cpp perf.h mosse.cpp mosse.h)