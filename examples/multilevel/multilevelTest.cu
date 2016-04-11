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
#include "multilevelTest.hpp"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

using namespace CCT;

int MultilevelTest::numStreams = 1;
int MultilevelTest::numJobs = 100;
bool MultilevelTest::useCustomPolicy = true;

//-----------------------------------------------------------------------------
MultilevelTest::MultilevelTest(int index)
    : TestClass(index)
{
    h_inputs.resize(numStreams);
    d_inputs.resize(numStreams);
    h_outputs.resize(numStreams);
    d_outputs.resize(numStreams);

    // Create memory for each stream case
    for (int ii = 0; ii < numStreams; ++ii)
    {
        gpuPtr->alloc(h_inputs[ii], LENGTH, CPU);
        gpuPtr->alloc(h_outputs[ii], LENGTH, CPU);

        gpuPtr->alloc(d_inputs[ii], LENGTH, GPU);
        gpuPtr->alloc(d_outputs[ii], LENGTH, GPU);
    }
}

//-----------------------------------------------------------------------------
void MultilevelTest::run()
{
    // Perform copy and compute test
    gpuPtr->timerStart(0); // Pick first stream to use as timer. 
    int numIterations = IDIVUP(numJobs, numStreams);
    for (int n = 0; n < numIterations; ++n)
    {
        for (int ii = 0; ii < numStreams; ++ii)
        {
            gpuPtr->copy(d_inputs[ii], h_inputs[ii], LENGTH, GPU, ii);
            thrust::device_ptr<short> pBuf = thrust::device_pointer_cast(d_inputs[ii]);
            // Using a custom policy means it will try and reuse existing
            // memory on subsequent invocations instead of creating and
            // deleting memory every time.
            if (useCustomPolicy)
            {
                thrust::sort(gpuPtr->thrustPolicy(ii), pBuf, pBuf + LENGTH);
            }
            else
            {
                thrust::sort(thrust::cuda::par.on(gpuPtr->stream(ii)), pBuf, pBuf + LENGTH);
            }
            gpuPtr->copy(h_outputs[ii], d_inputs[ii], LENGTH, CPU, ii);
        }
    }
    // Flush all outstanding work to ensure valid timing value.
    gpuPtr->timerStop(0);
    gpuPtr->deviceSynchronize();
    CCT_INFO("Total time = " << gpuPtr->timerElapsed(0) << "ms");
}
