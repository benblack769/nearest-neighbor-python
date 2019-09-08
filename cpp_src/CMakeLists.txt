cmake_minimum_required (VERSION 3.5)
project (nn-python C CXX)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR})

add_library(main SHARED
    ${CPP_SOURCE_DIR}/graph_nn.cpp
    ${CPP_SOURCE_DIR}/python-interface.cpp
    ${CPP_SOURCE_DIR}/cnpy.cpp
)
add_executable(maintest  ${CPP_SOURCE_DIR}/test_graph_nn.cpp) #${CPP_SOURCE_DIR}/graph_nn.cpp )
add_executable(cnpy_example ${CPP_SOURCE_DIR}/cnpy.cpp ${CPP_SOURCE_DIR}/cnpy_example.cpp)

set_property(TARGET maintest PROPERTY CXX_STANDARD 14)
set_property(TARGET maintest PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET cnpy_example PROPERTY CXX_STANDARD 14)
set_property(TARGET cnpy_example PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(maintest main)
