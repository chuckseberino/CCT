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
#include "scalingTest.hpp"

#include <string>

void usage()
{
    CCT_ERROR("Usage: \n\tscalingTest [-c <numCpuThreads>]\n\n");
}

int main(int argc, char** argv)
{
    int numCpuThreads = 1;

    // Perform basic command line parsing
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < argc; ++i)
    {
        if (args[i] == "-c") numCpuThreads = std::stoi(args[++i]);
        else if (args[i] == "-j") ScalingTest::numJobs = std::stoi(args[++i]);
        else
        {
            CCT_ERROR("Unknown cmd-line argument: " << args[i]);
            usage();
            return 1;
        }
    }


    for (int i=1; i<=4; ++i)
    {
        ScalingTest::numStreams = i;
        for (int step=0; step<=10; ++step)
        {
            ScalingTest::numThreads = 128 << step;
            CCT_INFO("Running with the following options:\n\t"
                << numCpuThreads << " CPU thread(s), each with "
                << ScalingTest::numThreads << " thread(s), each with "
                << ScalingTest::numStreams << " stream(s), each with "
                << ScalingTest::numJobs << " jobs.\n");
            CCT::TestComposer<ScalingTest> tc(numCpuThreads);
        }
    }
    return 0;
}
