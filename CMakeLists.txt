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

cmake_minimum_required(VERSION 3.0)
project(CCT)

#------------------------------------------------------------------------------
# Global settings
#------------------------------------------------------------------------------

# Place all binaries in a common location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if (CMAKE_HOST_WIN32)
    # Remove MinSizeRel and RelWithDebInfo configurations.
    set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
    # Needed to allow solution folders in Visual Studio.
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif ()

# Make sure that any 3rd party stuff is found at runtime
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add a cmake path to our custom scripts
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CMakeMacros)
# 3rd-party configuration is done here
find_package(3rdParty)

# Don't turn profiling on by default when using Windows, since it requires
# additional work by the user to have the following in the path:
if (CMAKE_HOST_WIN32)
    option(CCT_USE_GPU_PROFILING
        "Enable CPU-based profiling markers (requires C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64\\nvToolsExt64_1.dll in path)" OFF)
else ()
    option(CCT_USE_GPU_PROFILING "Enable CPU-based profiling markers" ON)
endif ()

#------------------------------------------------------------------------------
# Compiler Preprocessor definitions
#------------------------------------------------------------------------------
if (CMAKE_HOST_WIN32)
    add_definitions(
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_WARNINGS
        -DNOMINMAX
    )
else ()
    add_definitions(
        -Werror
        -Wno-unused-variable
        -fno-omit-frame-pointer
        -fno-strict-aliasing
        -std=c++11
    )
    if (CMAKE_HOST_APPLE)
        add_definitions(
            -arch x86_64
            -stdlib=libc++
            -Wno-deprecated-register
        )
    endif ()
endif ()

if (CUDA_FOUND AND CCT_USE_GPU_PROFILING)
    add_definitions(-DCCT_USE_GPU_PROFILING)
endif ()


#------------------------------------------------------------------------------
# Compiler include directories
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR})


#------------------------------------------------------------------------------
# Project directories
#------------------------------------------------------------------------------
add_subdirectory(cct)
add_subdirectory(examples)
add_subdirectory(docs)
