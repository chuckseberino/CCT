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
#include "CudaHelper.hpp"
#include "CudaDevice.hpp"
#include "CudaGpu.hpp"

#include <cstdlib> // for atexit

namespace CCT {

// Map a DeviceID to an implementation handle
typedef std::map<int, GpuPtr> GpuPtrMap;
typedef std::map<int, DevicePtr> DevicelPtrMap;
typedef std::lock_guard<std::mutex> scoped_lock; // imitate boost

static bool atExitRegistered = false;
static int maxThreadsPerGPU = 7;
static size_t numStreams = 4;
static GpuPtrMap gpuPtrMap;
static DevicelPtrMap devPtrMap;
std::mutex Helper::mutex;

//-----------------------------------------------------------------------------
GpuPtr Helper::getGpu(int index)
{
    // Check it against selection criteria.
    pair_int gpuBlock = selectDeviceAndBlock(index);
    if (-1 == gpuBlock.first) return GpuPtr();

    scoped_lock lock(mutex);

    // First check to see if this GPU device has been initialized.
    DevicePtr device;
    DevicelPtrMap::iterator dit = devPtrMap.find(gpuBlock.first);
    if (dit == devPtrMap.end())
    {
        device.reset(new Device(gpuBlock.first));
        // Double check to make sure our device is OK
        if (-1 == device->ID()) return GpuPtr();
        devPtrMap.insert(std::make_pair(gpuBlock.first, device));
    }
    else
    {
        device = dit->second;
        // Make sure we activate this per-thread.
        device->setDevice();
    }

    // Now that we have a device, get the block implementation handle.
    GpuPtr impl;
    GpuPtrMap::iterator it = gpuPtrMap.find(index);
    if (it == gpuPtrMap.end())
    {
        impl.reset(new Gpu(device));
        gpuPtrMap.insert(std::make_pair(index, impl));
    }
    else
    {
        impl = it->second;
    }

    if (!atExitRegistered)
    {
        atexit(Helper::destroy);
        atExitRegistered = true;
    }
    return impl;
}


//-----------------------------------------------------------------------------
// Determine the combination of GPU device and block within the GPU based upon
// a unique thread index. If the index is larger than the available blocks,
// then it returns -1.
pair_int Helper::selectDeviceAndBlock(int index)
{
    int numDevices = Device::numDevices();
    if (!numDevices || index >= maxThreadsPerGPU*numDevices)
    {
        return pair_int(-1, -1);
    }

    // Alternate between devices, then by block
    return pair_int(index % numDevices, index / numDevices);
}


//-----------------------------------------------------------------------------
void Helper::destroy()
{
    scoped_lock lock(mutex);
    gpuPtrMap.clear();
    devPtrMap.clear();
}


//-----------------------------------------------------------------------------
void Helper::setMaxThreadsPerGPU(int num)
{
    if (num > 0) maxThreadsPerGPU = num;
}

} // namespace CCT
