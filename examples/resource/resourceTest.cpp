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
#include "resourceTest.hpp"

#include <string>

void usage()
{
    CCT_ERROR("Usage: \n\tresourceTest [-t <numThreads>][-j <numJobs>][-s <numStreams>]\n\n");
}

int main(int argc, char** argv)
{
    int numThreads = 1;
    ResourceTest::numStreams = 1;
    ResourceTest::numJobs = 10;

    // Perform basic command line parsing
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < argc; ++i)
    {
        if (args[i] == "-t") numThreads = std::stoi(args[++i]);
        else if (args[i] == "-j") ResourceTest::numJobs = std::stoi(args[++i]);
        else if (args[i] == "-s") ResourceTest::numStreams = std::stoi(args[++i]);
        else
        {
            CCT_ERROR("Unknown cmd-line argument: " << args[i]);
            usage();
            return 1;
        }
    }
    CCT_INFO("Running with the following options:\n\t"
        << numThreads << " thread(s), each with "
        << ResourceTest::numJobs << " jobs over "
        << ResourceTest::numStreams << " stream(s).\n");

    CCT::TestComposer<ResourceTest> tc(numThreads);
    return 0;
}
