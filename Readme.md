
# learnsim1: C++ code for learning simulations with fixed and flexible learning rates


## Overview

This repository contains C++ code and input files.
The executable program `LearnSim`, built from this code, will run learning simulations with either fixed (Rescorla-Wagner) or flexible (Autostep) learning rates.
The program was used to produce the simulation results for the paper "Flexibility of learning in complex worlds" by Olof Leimar, Andrés Quiñones, and Redouan Bshary.


## System requirements

This program has been compiled and run on a Linux server with Ubuntu 22.04 LTS.
The C++ compiler was g++ version 11.3.0, provided by Ubuntu, with compiler flags for c++17, and `cmake` (<https://cmake.org/>) was used to build the program.
The instructions below should work for many Linux distributions, as well as for macOS, using the Apple supplied Clang version of g++.

The program reads input parameters from TOML files (<https://github.com/toml-lang/toml>), using the open source `cpptoml.h` header file (<https://github.com/skystrife/cpptoml>), which is included in this repository.

The program writes the simulation learning data to a HDF5 file (<https://www.hdfgroup.org/>), which is an open source binary file format.
The program uses the open source HighFive library (<https://github.com/BlueBrain/HighFive>) to write to such files.
These pieces of software need to be installed in order for `cmake` to successfully build the program.


## Installation guide

Install the repository from Github to a local computer.
There is a single directory `learnsim1` for source code and executable, a subdirectory `Data` where input data and simulation data files are kept, and a subdirectory `build` used by `cmake` for files generated during building, including the executable `LearnSim`.


## Building the program

The CMake build system is used.
If it does not exist, create a build subdirectory in the project folder (`mkdir build`) and make it the current directory (`cd build`).
If desired, for a build from scratch, delete any previous content (`rm -rf *`).
Run CMake from the build directory. For a release build:
```
cmake -D CMAKE_BUILD_TYPE=Release ../
```
and for a debug build replace Release with Debug.
If this succeeds, i.e. if the `CMakeLists.txt` file in the project folder is processed without problems, build the program:
```
cmake --build . --config Release
```
and for a debug build replace Release with Debug. This should produce an executable in the `build` directory.


## Running

Make the Data directory current.
Assuming that the executable is called `LearnSim` and with an input file called `RunRW1.toml`, corresponding to Rescorla-Wagner for case 1 in Figure 1 of the paper, run the program as
```
../build/LearnSim RunRW1.toml
```
which also reads information about the stimulus dimensions and compound stimuli from the `tsv` file `CompStim1.tsv`.

### Description of the learning simulations

There is an input file, for instance `RunRW1.toml`, which typically simulates 1,000 plus 1,000 trials, outputting the learning data to, e.g., the HDF5 file `RunRW1.h5`. For the cases shown in Figures 1 to 3 of the paper, the input files `RunRW1.toml`, `RunRW2.toml`, `RunAS1.toml`, and `RunAS2.toml` for Rescorla-Wagner and Autostep were used. The files `CompStim1.tsv` and `CompStim2.tsv`, given in the respective input files, contain information about the stimuli (see also Table 1 in the paper).


## License

The `LearnSim` program runs learning simulations with fixed and flexible learning rates.

Copyright (C) 2023  Olof Leimar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

