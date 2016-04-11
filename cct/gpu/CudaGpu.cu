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
#include "CudaGpu.hpp"
#include "CudaDevice.hpp"

#include <cuda_runtime.h>
#include <thrust/system/cuda/execution_policy.h>

namespace CCT {

typedef long long int c64_t;

//-----------------------------------------------------------------------------
__global__ void kernelSleep(c64_t sleepTicks, c64_t* d_diff=nullptr)
{
    c64_t start = clock64();
    c64_t offsetTicks = 0;
    while (offsetTicks < sleepTicks)
    {
        c64_t end = clock64();
        offsetTicks = end - start;
    }

    if (d_diff) *d_diff = offsetTicks;
}

//-----------------------------------------------------------------------------
void Gpu::sleep(int milliseconds, size_t streamIndex /*=MainStream*/)
{
    c64_t sleepTicks = milliseconds * device->clockRate();
    kernelSleep<<<1, 1, 0, stream(streamIndex)>>>(sleepTicks);
    CHECK_KERNEL("kernelSleep");
}


//-----------------------------------------------------------------------------
ThrustPolicy Gpu::thrustPolicy(size_t streamIndex /*=~0*/)
{
    // Keep index limited to valid stream.
    if (streamIndex > MainStream) streamIndex = MainStream;
    return ThrustPolicy(thrustCache[streamIndex]).on(stream(streamIndex));
}

} // namespace CCT
