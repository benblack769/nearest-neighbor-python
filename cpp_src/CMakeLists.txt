cmake_minimum_required (VERSION 3.5)
project (nn-python C CXX)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR})

add_library(main SHARED
    ${CPP_SOURCE_DIR}/barnes.cpp
    ${CPP_SOURCE_DIR}/python-interface.cpp
)
add_executable(maintest  ${CPP_SOURCE_DIR}/test_barnes.cpp)

target_compile_options(main PRIVATE -Wall -Wextra -pedantic)

set_property(TARGET maintest PROPERTY CXX_STANDARD 14)
set_property(TARGET maintest PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(maintest main)
