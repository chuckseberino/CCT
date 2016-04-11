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

#include <algorithm>
#include <cuda_runtime.h>

#ifdef CCT_USE_GPU_PROFILING
#   include <nvToolsExt.h>

#   if defined(CCT_MSVC)
#      include <Windows.h>
void CCT::EventTracer::setThreadName(const char* name)
{ nvtxNameOsThread(GetCurrentThreadId(), name); }
#   elif defined(CCT_DARWIN)
void CCT::EventTracer::setThreadName(const char* name)
{ uint64_t tid; pthread_threadid_np(NULL, &tid); nvtxNameOsThread(tid, name); }
#   else // CCT_GCC
void CCT::EventTracer::setThreadName(const char* name)
{ nvtxNameOsThread(pthread_self(), name); }
#   endif

CCT::EventTracer::EventTracer(const char* name, uint32_t color)
{
    if (color)
    {
        nvtxEventAttributes_t eventAttrib = { 0 };
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;
        nvtxRangePushEx(&eventAttrib);
    }
    else nvtxRangePushA(name);
}
CCT::EventTracer::~EventTracer()
{
    nvtxRangePop();
}
#endif

namespace CCT {

const size_t Gpu::MainStream;

//-----------------------------------------------------------------------------
Gpu::Gpu(DevicePtr device)
    : device(device)
    , thrustCache(NumStreams, ThrustAllocator(this))
    , streams(NumStreams)
{
    //-------------------------------------------------------------------------
    // Create independent streams of operation and sync/timing events
    //-------------------------------------------------------------------------
    for (int ii=0; ii<NumStreams; ++ii)
    {
        CCT_CHECK_GPU(cudaStreamCreateWithFlags(&streams[ii], cudaStreamNonBlocking));
    }

    for (int ii=0; ii<NumEvents; ++ii)
    {
        CUevent_st* eventStart, *eventStop;
        CCT_CHECK_GPU(cudaEventCreate(&eventStart));
        CCT_CHECK_GPU(cudaEventCreate(&eventStop));
        events.push_back(std::make_pair(eventStart, eventStop));
    }
}


//-----------------------------------------------------------------------------
Gpu::~Gpu()
{
    // Device never initialized or no CUDA hardware exists
    if (!device) return;

    thrustCache.clear();

    // Make sure and flush remaining work before exiting.
    for (int ii = 0; ii < streams.size(); ++ii)
    {
        synchronize(ii);
    }

    // Clean up shared references to memory.
    memPtr.clear();

    for (auto& s : streams)
    {
        CCT_CHECK_GPU(cudaStreamDestroy(s));
    }
    for (auto& e : events)
    {
        CCT_CHECK_GPU(cudaEventDestroy(e.first));
        CCT_CHECK_GPU(cudaEventDestroy(e.second));
    }
    CCT_TRACE("GPU Worker on device " << device->ID() << " complete");
}


//-----------------------------------------------------------------------------
void* Gpu::allocate(size_t size, MemType type)
{
    void *ptr = nullptr;
    if (GPU == type)
    {
        CCT_CHECK_GPU(cudaMalloc(&ptr, size));
        if (ptr) memPtr.push_back(SharedMem(ptr, cudaFree));
    }
    else if (UM == type)
    {
#if defined(CCT_DARWIN) // No support for UM on OSX
        CCT_ERROR("OS X Doesn't support Unified Memory");
#endif
        CCT_CHECK_GPU(cudaMallocManaged(&ptr, size));
        if (ptr) memPtr.push_back(SharedMem(ptr, cudaFree));
    }
    else // CPU
    {
        CCT_CHECK_GPU(cudaMallocHost(&ptr, size));
        if (ptr) memPtr.push_back(SharedMem(ptr, cudaFreeHost));
    }
    return ptr;
}


//-----------------------------------------------------------------------------
void Gpu::free(const void* ptr)
{
    for (MemoryVectorPtr::iterator it=memPtr.begin(); it!=memPtr.end(); ++it)
    {
        if (ptr == it->get())
        {
            memPtr.erase(it);
            return;
        }
    }
    // else
    CCT_WARN("Attempt to free unknown data: " << ptr);
}


//-----------------------------------------------------------------------------
void Gpu::implcopy(void* to, const void* from, size_t size, MemType type, size_t index)
{
    // Provide pass-through for Unified Memory transfers. Until CUDA 8 comes
    // out, we can't explicitly copy or even hint at movement. If source and
    // destination are the same, just no-op. Otherwise perform the copy as
    // normal.
    if (to == from) return; // Let CUDA migrate data internally.

    cudaMemcpyKind kind;
    switch(type)
    {
    case GPU: kind = cudaMemcpyHostToDevice; break;
    case CPU: kind = cudaMemcpyDeviceToHost; break;
    case DEV: kind = cudaMemcpyDeviceToDevice; break;
    default: return;
    }
    CCT_CHECK_GPU(cudaMemcpyAsync(to, from, size, kind, stream(index)));
}


//-----------------------------------------------------------------------------
void Gpu::implset(void* to, int value, size_t size, size_t index)
{
    CCT_CHECK_GPU(cudaMemsetAsync(to, value, size, stream(index)));
}


//-----------------------------------------------------------------------------
cudaStream_t Gpu::stream(size_t index /*=MainStream*/)
{
    index = std::min(index, MainStream);
    return streams[index];
}


//-----------------------------------------------------------------------------
void Gpu::streamWait(size_t streamIndex, size_t eventIndex)
{
    eventIndex = std::min(eventIndex, NumEvents-1);
    CCT_CHECK_GPU(cudaStreamWaitEvent(stream(streamIndex), events[eventIndex].second, 0));
}


//-----------------------------------------------------------------------------
void Gpu::timerStart(size_t eventIndex, size_t streamIndex /*=~0*/)
{
    eventIndex = std::min(eventIndex, NumEvents-1);
    streamIndex = std::min(eventIndex, MainStream);
    CCT_CHECK_GPU(cudaEventRecord(events[eventIndex].first, stream(streamIndex)));
}


//-----------------------------------------------------------------------------
void Gpu::timerStop(size_t eventIndex, size_t streamIndex /*=~0*/)
{
    eventIndex = std::min(eventIndex, NumEvents-1);
    streamIndex = std::min(eventIndex, MainStream);
    CCT_CHECK_GPU(cudaEventRecord(events[eventIndex].second, stream(streamIndex)));
}


//-----------------------------------------------------------------------------
float Gpu::timerElapsed(size_t eventIndex)
{
    float timeEvent = 0.f;
    // Make sure our stop event has finished.
    eventIndex = std::min(eventIndex, NumEvents-1);
    streamWait(eventIndex, eventIndex);

    CCT_CHECK_GPU(cudaEventElapsedTime(&timeEvent, events[eventIndex].first, events[eventIndex].second));
    return timeEvent;
}


//-----------------------------------------------------------------------------
void Gpu::synchronize(size_t streamIndex /*=0*/)
{
    CCT_CHECK_GPU(cudaStreamSynchronize(stream(streamIndex)));
}


//-----------------------------------------------------------------------------
void Gpu::deviceSynchronize()
{
    CCT_CHECK_GPU(cudaDeviceSynchronize());
}


//-----------------------------------------------------------------------------
void Gpu::getLastError(char const* const func, const char* const file, int const line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        CCT_ERROR("CUDA error at " << file << ":" << line << " rc = "
            << err << "(" << cudaGetErrorString(err) << ") " << func);
    }
}


//-----------------------------------------------------------------------------
ThrustAllocator::ptr_type ThrustAllocator::allocate(std::ptrdiff_t size)
{
    ptr_type result = NULL;
    FreeBlockMap::iterator freeBlock = freeBlockMap.find(size);
    if (freeBlock != freeBlockMap.end())
    {
        result = freeBlock->second;
        freeBlockMap.erase(freeBlock);
    }
    else
    {
        gpuPtr->alloc(result, size, GPU);
    }

    allocatedBlockMap.emplace(result, size);
    return result;
}


//-----------------------------------------------------------------------------
void ThrustAllocator::deallocate(ptr_type ptr, size_t)
{
    // Move memory from allocated to free map
    AllocatedBlockMap::iterator it = allocatedBlockMap.find(ptr);
    if (it == allocatedBlockMap.end())
    {
        CCT_ERROR("Attempt to remove unknown memory");
    }
    else
    {
        freeBlockMap.emplace(it->second, it->first);
        allocatedBlockMap.erase(it);
    }
}

} // namespace CCT
