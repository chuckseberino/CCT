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

using namespace CCT;

int HardwareTest::numStreams = 4;
int HardwareTest::threadBlock = 1000;
int HardwareTest::numJobs = 100;

//-----------------------------------------------------------------------------
__global__ void kernel(const short* input, size_t length, float* output, int delay)
{
    // Create global index and perform early return boundary check. 
    CCT_KERNEL_1D(index, length);

    // Perform some compute work for an arbitrary period of time.
    float scale = 3.2f;
    for (int i = 0; i<delay; ++i) scale += log(scale);

    output[index] = input[index] * scale;
}

//-----------------------------------------------------------------------------
HardwareTest::HardwareTest(int index)
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
void HardwareTest::run()
{
    dim3 blockSize(threadBlock);
    dim3 gridSize(IDIVUP(LENGTH, blockSize.x));

    // Perform copy and compute test
    gpuPtr->timerStart(0); // Pick first stream to use as timer. 
    int numIterations = IDIVUP(numJobs, numStreams);
    for (int n = 0; n < numIterations; ++n)
    {
        for (int ii = 0; ii < numStreams; ++ii)
        {
            gpuPtr->copy(d_inputs[ii], h_inputs[ii], LENGTH, GPU, ii);
            kernel<<<gridSize, blockSize, 0, gpuPtr->stream(ii)>>>(d_inputs[ii], LENGTH, d_outputs[ii], GPU_DELAY);
            gpuPtr->copy(h_outputs[ii], d_outputs[ii], LENGTH, CPU, ii);
        }
    }
    // Flush all outstanding work to ensure valid timing value.
    gpuPtr->timerStop(0);
    gpuPtr->deviceSynchronize();
    CCT_INFO("Total time = " << gpuPtr->timerElapsed(0) << "ms");
}
