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
#pragma once

#include <examples/TestClass.hpp>

//-----------------------------------------------------------------------------
class MemoryTest : public CCT::TestClass
{
public:
    explicit MemoryTest(int index);
    virtual void run();

    static int numStreams;         //!< How many streams to divide work into
    static bool usePinnedMemory;   //!< Whether to use pinned memory for test
    static bool useUnifiedMemory;  //!< Whether to use unified memory for test
    static int numJobs;            //!< Number of total work units

    const size_t LENGTH = 1000000; //!< Number of elements to work on
    /// Adjust this value up or down to modify the amount of time spent in kernel.
    /// Used to make kernel somewhat equivalent length of time relative to copy.
    const int GPU_DELAY = 100;

    /// Persistent memory to use for paged (non-pinned) test
    std::vector<short> paged_input;
    std::vector<float> paged_output;

    /// host and device pointers
    std::vector<short*> h_inputs, d_inputs;
    std::vector<float*> h_outputs, d_outputs;
};
