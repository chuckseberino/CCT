//-----------------------------------------------------------------------------
// Copyright 2016 Chuck Seberino
//
// This file is part of CCT.
//
// CCT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CCT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CCT.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------
#include "hardwareTest.hpp"

#include <string>

void usage()
{
    CCT_ERROR("Usage: \n\thardwareTest [-t <numThreads>][-j <numJobs>][-b <blockSize>][-s <numStreams>]\n\n");
}

int main(int argc, char** argv)
{
    int numThreads = 1;
    HardwareTest::numStreams = 4;
    HardwareTest::numJobs = 100;
    HardwareTest::threadBlock = 1000;

    // Perform basic command line parsing
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < argc; ++i)
    {
        if (args[i] == "-t") numThreads = std::stoi(args[++i]);
        else if (args[i] == "-j") HardwareTest::numJobs = std::stoi(args[++i]);
        else if (args[i] == "-b") HardwareTest::threadBlock = std::stoi(args[++i]);
        else if (args[i] == "-s") HardwareTest::numStreams = std::stoi(args[++i]);
        else
        {
            CCT_ERROR("Unknown cmd-line argument: " << args[i]);
            usage();
            return 1;
        }
    }
    CCT_INFO("Running with the following options:\n\t"
        << numThreads << " thread(s), each with "
        << HardwareTest::numJobs << " jobs using "
        << HardwareTest::threadBlock << " threads per block over "
        << HardwareTest::numStreams << " stream(s).\n");

    CCT::TestComposer<HardwareTest> tc(numThreads);
    return 0;
}
