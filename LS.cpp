#include "LearnSim.hpp"
#include <iostream>

// The LearnSim program runs learning simulations
// Copyright (C) 2023  Olof Leimar
// See Readme.md for copyright notice

int main(int argc, char* argv[])
{
    // Open input file and read indata
    InpData lsid(argv[1]);
    if (!lsid.OK) {
        std::cout << "Input failed!" << "\n";
        return -1;
    }
    // Run the iteration
    LearnSim ls(lsid);
    ls.Run();
    return 0;
}
