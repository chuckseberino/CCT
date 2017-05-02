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

using namespace CCT;

int ScalingTest::numStreams = 2;
int ScalingTest::numJobs = 100;
int ScalingTest::numThreads = 32;
size_t ScalingTest::LENGTH = 32 * 1024 * 1024;

//-----------------------------------------------------------------------------
// Basic column sum of 'length' columns over 'N' rows.
__global__ void sumKernel(const float* input, float* output, int N, int length)
{
    CCT_KERNEL_1D(index, length);
    float sum = 0.f; // Assumes no overflow

    input += index;
    for (int i=0; i<N; ++i)
    {
        sum += input[0];
        input += length;
    }
    output[index] = sum;
}

//-----------------------------------------------------------------------------
ScalingTest::ScalingTest(int index)
    : TestClass(index)
{
    d_inputs.resize(numStreams);
    d_outputs.resize(numStreams);

    // Create memory for each stream case
    for (int ii = 0; ii < numStreams; ++ii)
    {
        gpuPtr->alloc(d_inputs[ii], LENGTH, GPU);
        gpuPtr->alloc(d_outputs[ii], numThreads, GPU);
    }
}

//-----------------------------------------------------------------------------
void ScalingTest::run()
{
    int numIterations = IDIVUP(numJobs, numStreams);
    int blockSize = std::min(numThreads, 1024);
    int gridSize = IDIVUP(numThreads, blockSize);
    gpuPtr->timerStart(0); // Pick first stream to use as timer. 
    for (int n = 0; n < numIterations; ++n)
    {
        for (int ii = 0; ii < numStreams; ++ii)
        {
            sumKernel<<<gridSize, blockSize, 0, gpuPtr->stream(ii)>>>(
                d_inputs[ii], d_outputs[ii], LENGTH/numThreads, numThreads);
        }
    }
    // Flush all outstanding work to ensure valid timing value.
    gpuPtr->timerStop(0);
    gpuPtr->deviceSynchronize();
    CCT_INFO("Total time = " << gpuPtr->timerElapsed(0) << "ms");
}
