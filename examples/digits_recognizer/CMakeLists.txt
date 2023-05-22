cmake_minimum_required(VERSION 3.14)
project(mlp-digits-recognizer LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(mlp REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mlp-digits-recognizer)
target_sources(mlp-digits-recognizer PRIVATE ${sources})
target_link_libraries(mlp-digits-recognizer PRIVATE mlp::mlp)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mlp-digits-recognizer mlp::mlp)
endif()