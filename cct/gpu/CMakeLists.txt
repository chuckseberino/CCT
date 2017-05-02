# Copyright (C) 2016 Chuck Seberino
#
# This file is part of CCT.
#
# CCT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CCT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CCT.  If not, see <http://www.gnu.org/licenses/>.

set(CUDA_NVCC_FLAGS_DEBUG "-G -g" CACHE STRING "" FORCE)

# Default Windows flag(s)
set(CCT_NVCC_FLAGS -Xcompiler -wd4267)  #Thrust warning

# Set C++11 support in CMake
set (CMAKE_CXX_STANDARD 11)

# Which are overwritten on Linux
if (CMAKE_HOST_UNIX)
    set(CCT_NVCC_FLAGS --std=c++11)
endif (CMAKE_HOST_UNIX)

set(CUDA_NVCC_FLAGS ${CCT_NVCC_FLAGS}
    -Xcudafe --diag_suppress=field_without_dll_interface  #DLL warning
    -gencode arch=compute_30,code=sm_30 #Support for Macbook
    -gencode arch=compute_35,code=sm_35 #Support for Kepler (K40/K80)
    -gencode arch=compute_52,code=sm_52 #Support for Maxwell (Titan X, M6000)
    -gencode arch=compute_61,code=sm_61 #Support for Pascal (P6000, P100)
    CACHE STRING "" FORCE
)

file(GLOB src *.cpp *.hpp *.cu)
cct_cuda_add_library(gpu SHARED ${src})

include_directories(${NVTOOLSEXT_INCLUDE})

# Add static linking on linux
if (CMAKE_HOST_UNIX)
    target_link_libraries(gpu ${CUDA_LIBOS_LIBRARY})
    if (NOT CMAKE_HOST_APPLE)
        target_link_libraries(gpu dl pthread rt)
    endif ()
endif (CMAKE_HOST_UNIX)

if (CCT_USE_GPU_PROFILING)
    target_link_libraries(gpu ${NVTOOLSEXT_LIBRARY})
endif (CCT_USE_GPU_PROFILING)

install(TARGETS gpu DESTINATION bin)
