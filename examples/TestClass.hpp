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

#include <cct/gpu/CudaHelper.hpp>
#include <cct/gpu/CudaGpu.hpp>

#include <thread>
#include <vector>

namespace CCT {

//-----------------------------------------------------------------------------
/// Base class for testing GPU kernels. It provides access to a GPU handle
/// via gpuPtr.
class TestClass
{
public:
    /// Constructor with index to GPU handle. All memory allocation should be
    /// done here.
    explicit TestClass(int index)
        : gpuPtr(Helper::getGpu(index))
    {}

    /// Override this method to supply content to the test.
    virtual void run()
    {}

protected:
    GpuPtr gpuPtr; //!< Handle to GPU object
};


//-----------------------------------------------------------------------------
/// High level test class that handles spawning a given number of testing
/// threads. After creating the test objects (and allocating memory for them),
/// it will call the TestClass::run() method on each one to start them off, and
/// then wait for them to finish before cleaning up.
template <typename T>
class TestComposer
{
public:
    /// Constructor that specifies how many threads (TestClass objects) to use.
    explicit TestComposer(int numThreads)
    {
        // Start all threads and perform memory allocation, but wait to process
        for (int i = 0; i < numThreads; ++i)
        {
            tests.push_back(new T(i));
            threads.push_back(std::thread(&TestClass::run, tests[i]));
        }
    }

    /// Destructor that will clean up GPU memory and resources.
    ~TestComposer()
    {
        // Wait for all threads to finish
        for (int i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
            delete tests[i];
        }

        Helper::destroy();
    }


private:
    std::vector<std::thread> threads; //!< Threads to spawn
    std::vector<TestClass*> tests;    //!< One TestClass object per thread 
};

} // namespace CCT
