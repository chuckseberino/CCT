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

using namespace CCT;

int ResourceTest::numStreams = 1;
int ResourceTest::numJobs = 100;

//-----------------------------------------------------------------------------
// Performs a transpose of the data using shared memory. Assumes square matrix
// of side 'length'. Each thread processes a single input column.
__global__ void bigKernel(short* input, short* output, size_t length)
{
    // Create global index and perform early return boundary check. 
    CCT_KERNEL_1D(index, length);

    // This kernel allocates a substantial amount of shared memory, which
    // makes it the limiting factor for occupancy.
    extern __shared__ short shared_space[];

    // Transpose input data into shared space.
    for (int i=0; i<length; ++i)
    {
        shared_space[index*length + i] = input[i*length + index];
    }

    // Copy shared space into output data.
    for (int i=0; i<length; ++i)
    {
        output[i*length + index] = shared_space[i*length + index];
    }
}

//-----------------------------------------------------------------------------
__global__ void smallKernel(short* input, short* output, size_t length)
{
    // Create global index and perform early return boundary check. 
    CCT_KERNEL_1D(index, length);

    // Perform some compute work for an arbitrary period of time.
    float scale = 3.2f;
    for (int i = 0; i<length; ++i) scale += log(scale);
}

//-----------------------------------------------------------------------------
ResourceTest::ResourceTest(int index)
    : TestClass(index)
{
    h_inputs.resize(numStreams);
    d_inputs.resize(numStreams);
    h_outputs.resize(numStreams);
    d_outputs.resize(numStreams);

    // Create memory for each stream case
    for (int ii = 0; ii < numStreams; ++ii)
    {
        gpuPtr->alloc(h_inputs[ii], LENGTH*LENGTH, CPU);
        gpuPtr->alloc(h_outputs[ii], LENGTH*LENGTH, CPU);

        gpuPtr->alloc(d_inputs[ii], LENGTH*LENGTH, GPU);
        gpuPtr->alloc(d_outputs[ii], LENGTH*LENGTH, GPU);
    }
}

//-----------------------------------------------------------------------------
void ResourceTest::run()
{
    // Perform copy and compute test
    gpuPtr->timerStart(0); // Pick first stream to use as timer. 
    int numIterations = IDIVUP(numJobs, numStreams);

    size_t LENGTH_2 = LENGTH / 32;

    for (int n = 0; n < numIterations; ++n)
    {
        for (int ii = 0; ii < numStreams; ++ii)
        {
            // Set this kernel to use all shared memory with 25% occupancy.
            bigKernel<<<1, 256, 49152, gpuPtr->stream(ii)>>>(d_inputs[ii], d_outputs[ii], LENGTH_2);
            // This kernel uses small thread blocks (32), which also limit occupancy.
            smallKernel<<<LENGTH_2, LENGTH_2, 0, gpuPtr->stream(numStreams+ii)>>>(d_inputs[ii], d_outputs[ii], LENGTH_2);
        }
    }
    // Flush all outstanding work to ensure valid timing value.
    gpuPtr->timerStop(0);
    gpuPtr->deviceSynchronize();
    CCT_INFO("Total time = " << gpuPtr->timerElapsed(0) << "ms");
}
