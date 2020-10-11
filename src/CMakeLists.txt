cmake_minimum_required(VERSION 2.8)

project(mosse)

set (CMAKE_CXX_STANDARD 11)

add_compile_definitions(USE_OPENCV)

set (OpenCVPathDebug   "E:/Code/opencv/build2/debug/install")
set (OpenCVPathRelease "E:/Code/opencv/build2/release/install")

find_package(OpenCV 4.0 REQUIRED PATHS ${OpenCVPathDebug})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test test.cpp perf.cpp perf.h math.cpp math.h)
add_executable(mosse main.cpp perf.cpp perf.h mosse.cpp mosse.h math.cpp math.h)

target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(mosse ${OpenCV_LIBS})