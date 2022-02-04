cmake_minimum_required(VERSION 3.0)

project(gpu_math)

set(OCL_INCLUDE_DIR $ENV{INTELOCLSDKROOT}include)
set(OCL_LIB_DIR $ENV{INTELOCLSDKROOT}lib/x64)

include_directories(${OCL_INCLUDE_DIR})
link_directories(${OCL_LIB_DIR})

add_executable(gpu_math gpu/host.cpp util.cpp util.h perf.h perf.cpp gpu/vkFFT.h)
target_link_libraries(gpu_math OpenCL)

set_target_properties(gpu_math PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/$<0:>)

# file(GLOB OPENCL_KERNEL_FILES "${CMAKE_CURRENT_LIST_DIR}/gpu/*.cl")
# file(COPY ${OPENCL_KERNEL_FILES} DESTINATION ${PROJECT_BINARY_DIR}/bin)

project(mosse)

set (CMAKE_CXX_STANDARD 11)

option(WITH_OPENCV "WithOpenCV" ON)
if(WITH_OPENCV)
    add_compile_definitions(USE_OPENCV)
    find_package(OpenCV 4.0 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_executable(test test.cpp perf.cpp math.cpp util.cpp perf.h math.h util.h)
add_executable(mosse main.cpp perf.cpp mosse.cpp math.cpp util.cpp perf.h mosse.h math.h util.h)

set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/$<0:>)
set_target_properties(mosse PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/$<0:>)

if(WITH_OPENCV)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(mosse ${OpenCV_LIBS})
endif()

