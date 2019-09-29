cmake_minimum_required (VERSION 3.5)
project (nn-python C CXX)
include(CheckIncludeFile)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(CPP_SOURCE_DIR ${PROJECT_SOURCE_DIR})


CHECK_INCLUDE_FILE(immintrin.h HAS_INTRIN)

add_library(fast_dot_prod STATIC ${CPP_SOURCE_DIR}/fast_dot_prod.cpp)
if(HAS_INTRIN)
    target_compile_options(fast_dot_prod PRIVATE -O3 -mavx -fPIC)
    target_compile_definitions(fast_dot_prod PRIVATE USE_INTRIN)
else()
    target_compile_options(fast_dot_prod PRIVATE -O3 -fPIC)
endif()

add_library(main SHARED
    ${CPP_SOURCE_DIR}/graph_nn.cpp
    ${CPP_SOURCE_DIR}/python-interface.cpp
    ${CPP_SOURCE_DIR}/data_accessor.cpp
    ${CPP_SOURCE_DIR}/global_ranker.cpp
    ${CPP_SOURCE_DIR}/local_ranker.cpp
)
target_compile_options(main PRIVATE -Wall -Wextra)


add_executable(maintest ${CPP_SOURCE_DIR}/test_graph_nn.cpp ${CPP_SOURCE_DIR}/cnpy.cpp) #${CPP_SOURCE_DIR}/graph_nn.cpp )
add_executable(cnpy_example ${CPP_SOURCE_DIR}/cnpy.cpp ${CPP_SOURCE_DIR}/cnpy_example.cpp)

set_property(TARGET fast_dot_prod PROPERTY CXX_STANDARD 14)
set_property(TARGET fast_dot_prod PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET maintest PROPERTY CXX_STANDARD 14)
set_property(TARGET maintest PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET cnpy_example PROPERTY CXX_STANDARD 14)
set_property(TARGET cnpy_example PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(main fast_dot_prod)
target_link_libraries(maintest main)
