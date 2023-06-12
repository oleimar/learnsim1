cmake_minimum_required(VERSION 3.10)

# The LearnSim program runs learning simulations
# Copyright (C) 2023  Olof Leimar
# See Readme.md for copyright notice

# set the project name
project(LearnSim VERSION 1.0 LANGUAGES CXX C)

# set c++ standard
set(CMAKE_CXX_STANDARD 17)

# we need HighFive to handle hdf5 files
find_package(HighFive CONFIG REQUIRED)

# add the executable
add_executable(LearnSim LS.cpp LearnSim.cpp hdf5code.cpp Utils.cpp)

# link target
target_link_libraries(LearnSim PUBLIC HighFive)
