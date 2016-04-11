# Copyright 2016 Chuck Seberino
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

#------------------------------------------------------------------------------
# CUDA configuration
#------------------------------------------------------------------------------
if (NOT MSVC14) # CUDA 7.5 doesn't yet MSVC2015
    find_package(CUDA)
endif () 

if (CUDA_FOUND)
    # Statically link CUDA libraries in so we can just rely on a runtime check
    # and allow the same binary to run on systems without GPU support.
    #
    # FIXME: There is no direct way to do this in CMake right now, so simply
    # replace the .so libs with their _static.a equivalents.
    if (NOT CMAKE_HOST_APPLE AND CUDA_USE_STATIC_CUDA_RUNTIME)
        set(lext ".so")
        if (CMAKE_HOST_APPLE)
            set(lext ".dylib")
        endif ()
        # Additional lib to finish static linking. Get path from cudart.
        string(REPLACE "cudart${lext}" "culibos.a" CUDA_LIBOS_LIBRARY ${CUDA_CUDART_LIBRARY})
        set(CUDA_LIBOS_LIBRARY ${CUDA_LIBOS_LIBRARY} CACHE FILEPATH "")
        mark_as_advanced(CUDA_LIBOS_LIBRARY)

        foreach (clib cublas;CUDART;cufft;curand;cupti;cusolver;cusparse;nppc;nppi;npps)
            if (CUDA_${clib}_LIBRARY)
                string(REPLACE "${lext}" "_static.a" CUDA_${clib}_LIBRARY ${CUDA_${clib}_LIBRARY})
            endif ()
        endforeach (clib)

        # cuda_add_ needs this variable to be updated too.
        set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
    endif()

    # Add support for nv Tools Extension
    find_path(NVTOOLSEXT_INCLUDE nvToolsExt.h
        PATHS
        /usr/local/cuda/include
        $ENV{NVTOOLSEXT_PATH}/include
    )
    find_library(NVTOOLSEXT_LIBRARY
        NAMES libnvToolsExt.so nvToolsExt64_1.lib libnvToolsExt.dylib
        NAMES_PER_DIR
        PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
        $ENV{NVTOOLSEXT_PATH}/lib/x64
    )
    mark_as_advanced(NVTOOLSEXT_INCLUDE NVTOOLSEXT_LIBRARY)
endif (CUDA_FOUND)


#------------------------------------------------------------------------------
# Doxygen configuration
#------------------------------------------------------------------------------
find_package(Doxygen)
