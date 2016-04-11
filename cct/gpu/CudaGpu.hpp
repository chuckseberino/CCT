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
/// @file CudaGpu.hpp
/// @class CCT::Gpu
/// @brief NVIDIA CUDA implementation details for managing GPU resources.
//
//-----------------------------------------------------------------------------
#pragma once

#include <cct/Config.hpp>

#include <map>
#include <vector>

#ifdef CCT_MSVC
#   pragma warning(push)
#   pragma warning(disable:4251) // Disable DLL-interface warning on STL types
#endif

#ifdef gpu_EXPORTS
#   define LIBGPU_DECL CCT_EXPORT
#else
#   define LIBGPU_DECL CCT_IMPORT
#endif

struct CUstream_st;
struct CUevent_st;

// Forward declare some thrust-based types
namespace thrust {
namespace detail { template <typename, template <typename> class> struct execute_with_allocator; }
namespace system { namespace cuda { namespace detail { template <typename> class execute_on_stream_base; }}}
}

namespace CCT {

class ThrustAllocator;
typedef std::vector<ThrustAllocator> ThrustAllocatorVector;
typedef thrust::detail::execute_with_allocator<ThrustAllocator, thrust::system::cuda::detail::execute_on_stream_base> ThrustPolicy;
class Device;
typedef std::shared_ptr<Device> DevicePtr;

/// Denote the type of memory to allocate or copy
enum MemType {
    CPU, //!< Host memory
    GPU, //!< Device memory
    UM,  //!< Unified memory
    DEV  //!< Denotes device-to-device copy
};

class LIBGPU_DECL Gpu
{
public:
    /// Constructor.
    /// @param[in] device Specify device to use
    explicit Gpu(DevicePtr device);

    /// Destructor. Releases all CUDA-created memory properly.
    ~Gpu();

    /// Create memory for use. Can either be host(CPU), device(GPU), or unified
    /// memory(UM) allocated. Memory created in this fashion will be cleaned up
    /// automatically on class destruction. CPU or UM based memory will be
    /// pinned on the host to allow for ansynchronous transfers.
    /// @param[in] var Variable to assign memory pointer to
    /// @param[in] numElements Size of elements to create (not in bytes)
    /// @param[in] type Which type of memory to allocate
    template <typename T> void alloc(T*& var, size_t numElements, MemType type)
    { var = reinterpret_cast<T*>(allocate(numElements*sizeof(T), type)); }

    /// Removes memory that was previous created from alloc.
    /// @param[in] ptr Pointer to memory to free
    void free(const void* ptr);

    /// Copies data from one location to another. This method is aware of the
    /// data types being copied, so the numElements input doesn't include any
    /// element size, such as sizeof(int) added to it.
    /// @param[out] to Destination location
    /// @param[in] from Source location
    /// @param[in] numElements Size of elements to copy (not in bytes)
    /// @param[in] type Memory copy type (destination)
    /// @param[in] index Stream index to use
    template <typename T> void copy(T* to, const T* from, size_t numElements, MemType type, size_t index=MainStream)
    { implcopy((void*)to, (const void*)from, numElements*sizeof(T), type, index); }

    /// Sets data in GPU memory. This method is aware of the data types being
    /// copied, so the numElements input doesn't include any element size, such
    /// as sizeof(int) added to it.
    /// @param[out] to Destination location
    /// @param[in] value Value to set
    /// @param[in] numElements Size of elements to set (not in bytes)
    /// @param[in] index Stream index to use
    template <typename T> void set(T* to, int value, size_t numElements, size_t index=MainStream)
    { implset((void*)to, value, numElements*sizeof(T), index); }

    /// Retrieve a handle to a GPU stream.
    /// @param[in] index Stream index to retrieve.
    /// @return cudaStream_t Handle to stream for kernel invocation
    CUstream_st* stream(size_t index=MainStream);

    /// Block until the event for this stream index has completed.
    /// @param[in] streamIndex Stream index to retrieve
    /// @param[in] eventIndex Event index to wait on
    void streamWait(size_t streamIndex, size_t eventIndex);

    /// Event and timing start marker.
    /// @param[in] eventIndex Event index to use
    /// @param[in] streamIndex Stream index to use. Defaults to min(MainStream, eventIndex)
    void timerStart(size_t eventIndex, size_t streamIndex=~0);

    /// Event and timing stop marker.
    /// @param[in] eventIndex Event index to use
    /// @param[in] streamIndex Stream index to use. Defaults to min(MainStream, eventIndex)
    void timerStop(size_t eventIndex, size_t streamIndex=~0);

    /// Elapsed time between start and stop events.
    /// @param[in] eventIndex Event index to use
    /// @return float Time elapsed since timerStart() in milliseconds
    float timerElapsed(size_t eventIndex);

    /// Synchronize (block) until stream is flushed.
    /// @param[in] streamIndex Stream index to use
    void synchronize(size_t streamIndex = MainStream);

    /// Synchronize all GPU work on the entire device. Typically only required
    /// when trying to flush work from all streams.
    void deviceSynchronize();

    /// Pause the CUDA stream for a specified period of time. Typically only
    /// used when testing thread synchronization for potential race conditions.
    /// @param[in] milliseconds Amount of time to wait
    /// @param[in] streamIndex Stream index to use
    void sleep(int milliseconds, size_t streamIndex=MainStream);

    /// Checks GPU status and prints out an error code if there is an error.
    /// @param[in] func Description of location
    /// @param[in] file File name
    /// @param[in] line Line number
    static void getLastError(char const* const func, const char* const file, int const line);

    /// Provide convenience method to select a thrust execution policy based
    /// upon stream. This will launch the given thrust kernel on the given
    /// stream as well as use a per-stream cached allocator to re-use memory.
    /// @param[in] streamIndex Stream index to use
    /// @return ThrustPolicy Execution policy to pass to thrust functions
    ThrustPolicy thrustPolicy(size_t streamIndex = 0);

private:
    typedef std::shared_ptr<void> SharedMem;
    typedef std::vector<SharedMem> MemoryVectorPtr;
    typedef std::vector<std::pair<CUevent_st*,CUevent_st*> > EventPairVector;
    typedef std::vector<CUstream_st*> StreamPtr;
    friend class Device;
    friend class ThrustAllocator;

    /// @copydoc alloc(T*&,size_t,MemType)
    /// @return void* Allocated memory
    void* allocate(size_t size, MemType type);

    /// @copydoc copy(T*,const T*,size_t,MemType,size_t)
    void implcopy(void* to, const void* from, size_t size, MemType type, size_t index);

    /// @copydoc set(T*,int,size_t,size_t)
    void implset(void* to, int value, size_t size, size_t index);

    static const size_t NumStreams = 5;            //!< Streams per handle
    static const size_t MainStream = NumStreams-1; //!< Default if none set
    static const size_t NumEvents  = NumStreams;   //!< One event per stream

    DevicePtr device;                  //!< Handle to GPU device attached to
    EventPairVector events;            //!< Event recording start/stop markers
    StreamPtr streams;                 //!< Handle to GPU Streams
    MemoryVectorPtr memPtr;            //!< Vector to allocated memory
    ThrustAllocatorVector thrustCache; //!< Reusable temp memory for thrust
};


//-----------------------------------------------------------------------------
/// ThrustAllocator attempts to re-use existing temporaries.
class LIBGPU_DECL ThrustAllocator
{
public:
    explicit ThrustAllocator(Gpu* gpuPtr) : gpuPtr(gpuPtr) {}

    /// Provide required interface for thrust usage
    typedef char value_type;
    typedef value_type* ptr_type;
    ptr_type allocate(std::ptrdiff_t size);
    void deallocate(ptr_type ptr, size_t);

private:
    typedef std::multimap<std::ptrdiff_t, ptr_type> FreeBlockMap;
    typedef std::map<ptr_type, std::ptrdiff_t>      AllocatedBlockMap;

    Gpu* gpuPtr;                         //!< Handle to GPU details
    FreeBlockMap freeBlockMap;           //!< List of available blocks
    AllocatedBlockMap allocatedBlockMap; //!< List of used blocks
};


#ifdef CCT_MSVC
#   pragma warning(pop)
#endif

#define CHECK_KERNEL(val)  CCT::Gpu::getLastError(#val, __FILE__, __LINE__)
#define CCT_CHECK_GPU(val) do \
    { if (val) { CHECK_KERNEL(val); exit(1); }  } while(0)

// Provide support for tracing CPU code
#ifdef CCT_USE_GPU_PROFILING
#   define GPU_TRACE_FUNCTION   CCT::EventTracer _trace(__FUNCTION__)
#   define GPU_TRACE_COLOR(f,c) CCT::EventTracer _trace(f, c)
#   define GPU_TRACE_THREAD(s)  CCT::EventTracer::setThreadName(s)
struct EventTracer
{
    LIBGPU_DECL explicit EventTracer(const char* name, uint32_t color = 0);
    LIBGPU_DECL static void setThreadName(const char* name);
    LIBGPU_DECL ~EventTracer();
};
#else
#   define GPU_TRACE_FUNCTION
#   define GPU_TRACE_COLOR(f,c)
#   define GPU_TRACE_THREAD(s)
#endif

} // namespace CCT
