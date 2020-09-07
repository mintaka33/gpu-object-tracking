cmake_minimum_required(VERSION 2.8)

project(math)
set (SOURCES_ math.cpp)
set (CMAKE_CXX_STANDARD 11)

add_executable(math ${SOURCES_})