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

add_executable(examen main_cuda.cpp
        kernel.cu)

target_link_libraries(examen fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(examen PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(examen PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(grises main_cuda_grises.cpp
        kernel_grises.cu)

target_link_libraries(grises fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(grises PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(grises PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(suma main_cuda_suma.cpp
        kernel_suma.cu)

target_link_libraries(suma fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(suma PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(suma PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(espejo main_cuda_espejo.cpp
        kernel_espejo.cu)

target_link_libraries(espejo fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(espejo PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(espejo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(rotar main_cuda_rotar.cpp
        kernel_rotar.cu)

target_link_libraries(rotar fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(rotar PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(rotar PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(bordes main_cuda_bordes.cpp
        kernel_bordes.cu)

target_link_libraries(bordes fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(bordes PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(bordes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(mediana main_cuda_mediana.cpp
        kernel_mediana.cu)

target_link_libraries(mediana fmt::fmt sfml::sfml ${CUDART_LIBRARY})

target_include_directories(mediana PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(mediana PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


