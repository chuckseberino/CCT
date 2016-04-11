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
//
/// @file CudaHelper.hpp
/// @class CCT::Helper
/// @brief Opaque wrapper around GPU implementation details.
//
//-----------------------------------------------------------------------------
#pragma once

#include <cct/Config.hpp>

#include <mutex>

#ifdef gpu_EXPORTS
#   define LIBGPU_DECL CCT_EXPORT
#else
#   define LIBGPU_DECL CCT_IMPORT
#endif

namespace CCT {

/// Tuple of device/block
typedef std::pair<int, int> pair_int;

class Helper
{
public:
    /// Retrieve a handle to the CUDA implementation handle. The input is a
    /// unique integer representing a round-robin style selection of available
    /// GPUs. It will first cycle through each device, to try and load balance
    /// work amonst all devices. Max value is numDevices * MaxThreadsPerGPU.
    /// @param[in] index Arbitrary index value
    /// @return GpuPtr Handle to GPU internals or NULL on failure.
    static LIBGPU_DECL GpuPtr getGpu(int index);

    /// Performs a selection from the available GPUs on which one to use
    /// as well as which block based upon the input index.
    /// @param[in] index Arbitrary index value for pseudo-uniqueness
    /// @return DeviceID Unique device and block identifier selected, or -1 on failure
    static LIBGPU_DECL pair_int selectDeviceAndBlock(int index);

    /// Clean up Handle.
    static LIBGPU_DECL void destroy();

    /// Specify a maximum number of threads to be spawned per GPU device.
    /// @param[in] num Number of threads allowed per GPU
    static LIBGPU_DECL void setMaxThreadsPerGPU(int num);

private:
    static std::mutex mutex; //!< Serialize access
};

} // namespace CCT
