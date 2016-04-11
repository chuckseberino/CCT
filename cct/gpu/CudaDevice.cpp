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
#include "CudaDevice.hpp"
#include "CudaGpu.hpp"

#include <cuda_runtime.h>

namespace CCT {

Device::DeviceMap Device::deviceMap;

//-----------------------------------------------------------------------------
Device::Device(int _deviceID)
    : deviceID(-1)
    , clockRateInKHz(0)
{
    if (!numDevices()) // No CUDA capable devices
    {
        return;
    }
    DeviceMap::const_iterator it = deviceMap.find(_deviceID);
    if (it != deviceMap.end())
    {
        deviceID = it->second;
        setDevice();

        cudaDeviceProp deviceProp;
        CCT_CHECK_GPU(cudaGetDeviceProperties(&deviceProp, deviceID));
        clockRateInKHz = deviceProp.clockRate;
    }
    else
    {
        CCT_WARN("Improper GPU device");
    }
}


//-----------------------------------------------------------------------------
Device::~Device()
{
    // Device never initialized or no CUDA hardware exists
    if (-1 == deviceID)
    {
        return;
    }

    CCT_DEBUG("Shutting down GPU...");
    setDevice();
    CCT_CHECK_GPU(cudaDeviceReset());
    CCT_DEBUG("Done");
}


//-----------------------------------------------------------------------------
int Device::numDevices()
{
    int devMapSize = int(deviceMap.size());
    if (!devMapSize)
    {
        // Go through and check for supported devices.
        int deviceCount = 0;
        if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
        {
            return devMapSize;
        }

        for (int ii = 0; ii < deviceCount; ++ii)
        {
            cudaDeviceProp deviceProp;
            CCT_CHECK_GPU(cudaGetDeviceProperties(&deviceProp, ii));
#if 0
            if (getenv("CCT_USE_ANY_GPU"))
            {
                CCT_INFO("Bypassing GPU capability check.");
            }
            else
            {
                // NOTE: Only allow compute capability 3.5 and above!
                if (deviceProp.computeMode == cudaComputeModeProhibited
                    || (deviceProp.major < 3)
                    || (deviceProp.major == 3 && deviceProp.minor < 5))
                {
                    continue;
                }
            }
#endif
            CCT_INFO("Found \"" << deviceProp.name << "(Device " << ii
                << ")\" with compute capability " << deviceProp.major << "."
                << deviceProp.minor);

            deviceMap.emplace(devMapSize++, ii);
        }
    }
    return devMapSize;
}


//-----------------------------------------------------------------------------
void Device::setDevice()
{
    CCT_CHECK_GPU(cudaSetDevice(deviceID));
}

} // namespace CCT

