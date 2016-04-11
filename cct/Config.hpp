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

#include <iostream>
#include <memory>

//-----------------------------------------------------------------------------
// Stub out printing to just the basics for now.
#define CCT_LOG(s,x) std::cout << s << x << std::endl
#define CCT_ERROR(x) CCT_LOG("Error: ", x)
#define CCT_WARN(x)  CCT_LOG("Warn:  ", x)
#define CCT_INFO(x)  CCT_LOG("Info:  ", x)
#define CCT_DEBUG(x) CCT_LOG("Debug: ", x)
#define CCT_TRACE(x) CCT_LOG("Trace: ", x)

/// Perform an integer division round up
constexpr unsigned int IDIVUP(unsigned int x, unsigned int d)
{
    return (x+d-1) / d;
}

/// Up front thread index check for a 1-D kernel
#define CCT_KERNEL_1D(var, length) \
   int var = blockIdx.x*blockDim.x + threadIdx.x; \
   if (var >= (length)) return

namespace CCT {
    /// Forward declare GPU handle
    class Gpu; typedef std::shared_ptr<Gpu> GpuPtr;
}


//-----------------------------------------------------------------------------
// Define platform/compiler type
#if defined(_MSC_VER)
#   define CCT_MSVC 1
#elif defined(__APPLE__)
#   define CCT_DARWIN 1
#   define CCT_GCC 1 // For now also set
#elif defined(__GNUC__)
#   define CCT_GCC 1
#else
#   error "Unsupported compiler"
#endif

//-----------------------------------------------------------------------------
// Declaration specification for Windows DLL generation.
#if defined(CCT_MSVC)
#   define CCT_EXPORT __declspec(dllexport)
#   define CCT_IMPORT __declspec(dllimport)
#else
#   define CCT_EXPORT
#   define CCT_IMPORT
#endif

