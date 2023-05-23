cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

project(mlp
        VERSION 0.0.1
        DESCRIPTION "Simple Multilayer Perceptron implementation"
        LANGUAGES CXX)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------
include(cmake/utils.cmake)
include(GNUInstallDirs)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# MLP_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(MLP_TESTS "Build mlp tests" OFF)
option(MLP_BUILD_EXAMPLES "Build mlp examples" OFF)
option(MLP_BUILD_DOCS "Build mlp documentation" OFF)
option(MLP_INSTALL "Generate target for installing mlp" ${is_top_level})
set_if_undefined(MLP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/mlp" CACHE STRING
        "Install path for mlp package-related CMake files")

if (DEFINED MLP_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MLP_SHARED_LIBS})
endif ()

if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(mlp include/mlp/mlp.h)
add_library(mlp::mlp ALIAS mlp)

#----------------------------------------------------------------------------------------------------------------------
# NeuralNetwork dependencies
#----------------------------------------------------------------------------------------------------------------------

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
target_link_libraries(mlp Eigen3::Eigen)

include_directories(./EigenRand)


#----------------------------------------------------------------------------------------------------------------------
# mlp sources
#----------------------------------------------------------------------------------------------------------------------

include(GenerateExportHeader)
set(export_file_name "export_shared.h")

if (NOT BUILD_SHARED_LIBS)
    set(export_file_name "export_static.h")
endif ()

generate_export_header(mlp EXPORT_FILE_NAME include/mlp/${export_file_name})

set(sources
        include/mlp/
        include/mlp/mlp.h
        include/mlp/mlp.cpp
        src/linear_layer.h
        src/linear_layer.cpp
        src/loss_func.h
        src/loss_func.cpp
        src/non_linear_layer.h
        src/non_linear_layer.cpp
        )
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# mlp target
#----------------------------------------------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

target_sources(mlp PRIVATE ${sources})
target_compile_definitions(mlp PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:MLP_STATIC_DEFINE>")

target_include_directories(mlp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(mlp PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION})

#----------------------------------------------------------------------------------------------------------------------
# other targets
#----------------------------------------------------------------------------------------------------------------------

if (MLP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (MLP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (MLP_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs include)
endif ()