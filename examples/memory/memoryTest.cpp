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
#include "memoryTest.hpp"

#include <string>

void usage()
{
    CCT_ERROR("Usage: \n\tmemoryTest [-t <numThreads>][-j <numJobs>][-s <numStreams>][-pin <0|1>][-um <0|1>\n\n");
}

int main(int argc, char** argv)
{
    int numThreads = 1;
    MemoryTest::numStreams = 4;
    MemoryTest::numJobs = 100;
    MemoryTest::usePinnedMemory = true;
    MemoryTest::useUnifiedMemory = true;

    // Perform basic command line parsing
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < argc; ++i)
    {
        if (args[i] == "-t") numThreads = std::stoi(args[++i]);
        else if (args[i] == "-j") MemoryTest::numJobs = std::stoi(args[++i]);
        else if (args[i] == "-s") MemoryTest::numStreams = std::stoi(args[++i]);
        else if (args[i] == "-pin") MemoryTest::usePinnedMemory = bool(1 == std::stoi(args[++i]));
        else if (args[i] == "-um") MemoryTest::useUnifiedMemory = bool(1 == std::stoi(args[++i]));
        else
        {
            CCT_ERROR("Unknown cmd-line argument: " << args[i]);
            usage();
            return 1;
        }
    }
    CCT_INFO("Running with the following options:\n\t"
        << numThreads << " thread(s), each with "
        << MemoryTest::numJobs << " jobs over "
        << MemoryTest::numStreams << " stream(s) using "
        << (MemoryTest::useUnifiedMemory ? "Unified" :
            MemoryTest::usePinnedMemory ? "Pinned" : "Pageable") << " memory.\n");

    CCT::TestComposer<MemoryTest> tc(numThreads);
    return 0;
}
