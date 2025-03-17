cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_COMPILER icpx)

project(labs CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)

add_subdirectory(sample)
add_subdirectory(rng)
add_subdirectory(tasks)
