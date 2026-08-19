# from https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top

cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(qant-native-computing-toolkit-unittests)

# for xtensor and catch 2
get_filename_component(CONDA_PREFIX_ENV "$ENV{CONDA_PREFIX}" ABSOLUTE)

find_package(Catch2 3.4.0 REQUIRED)
find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain ${TOOLKIT})
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/..)  # for dlpack_utils.h
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR}/../../external/dlpack/include)
target_include_directories(tests PRIVATE ${CONDA_PREFIX_ENV}/include)

add_compile_options(-Werror)
target_compile_features(tests PUBLIC cxx_std_23)

include(CTest)
include(Catch)
catch_discover_tests(tests)
