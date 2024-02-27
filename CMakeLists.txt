cmake_minimum_required(VERSION 3.26)
project(grupal_main2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(fmt)
find_package(SFML)

add_executable(grupal_main2 main.cu
        utils.h utils.cpp)

target_link_libraries(grupal_main2 fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(grupal_main2 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(grupal_main2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
