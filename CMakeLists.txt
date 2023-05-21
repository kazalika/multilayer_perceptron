cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

project(NeuralNetwork
        VERSION 0.0.1
        DESCRIPTION "Multilayer Perceptron Simple Implementation"
        LANGUAGES CXX)


include(cmake/utils.cmake)
include(GNUInstallDirs)

set(sources
        src/mlp.h
        src/mlp.cpp
        src/non_linear_layer.h
        src/non_linear_layer.cpp
        src/linear_layer.h
        src/linear_layer.cpp
        src/loss_func.h
        )
