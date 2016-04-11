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
#include "streamsyncTest.hpp"

#include <string>

void usage()
{
    CCT_ERROR("Usage: \n\tstreamsyncTest [-t <numThreads>][-j <numJobs>][-s 0|1]\n\n");
}

int main(int argc, char** argv)
{
    int numThreads = 1;
    StreamsyncTest::numJobs = 20;

    // Perform basic command line parsing
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < argc; ++i)
    {
        if (args[i] == "-t") numThreads = std::stoi(args[++i]);
        else if (args[i] == "-j") StreamsyncTest::numJobs = std::stoi(args[++i]);
        else if (args[i] == "-s") StreamsyncTest::numStreams = std::stoi(args[++i]) == 1;
        else
        {
            CCT_ERROR("Unknown cmd-line argument: " << args[i]);
            usage();
            return 1;
        }
    }
    CCT_INFO("Running with the following options:\n\t"
        << (StreamsyncTest::syncStreams ? "SYNCED " : "UNSYNCED ")
        << numThreads << " thread(s), each with "
        << StreamsyncTest::numJobs << " jobs.\n");

    CCT::TestComposer<StreamsyncTest> tc(numThreads);
    return 0;
}
