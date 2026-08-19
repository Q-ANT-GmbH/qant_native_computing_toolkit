# from https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top

cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(qant-native-computing-toolkit-memtests)


# for xtensor
get_filename_component(CONDA_PREFIX_ENV "$ENV{CONDA_PREFIX}" ABSOLUTE)
# for dlpack
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

function(add_test_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..   # for dlpack_utils.h
            ${PROJECT_ROOT}/external/dlpack/include
            ${CONDA_PREFIX_ENV}/include
    )
    target_link_libraries(${target_name} PRIVATE ${TOOLKIT})
    target_compile_features(${target_name} PUBLIC cxx_std_23)
endfunction()

add_test_executable(test_memory_safety test_memory_safety.cpp)
add_test_executable(test_memory_safety_i16 test_memory_safety_i16.cpp)
add_test_executable(test_memory_safety_mat test_memory_safety_mat.cpp)
