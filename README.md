Chuck's CUDA Toolkit (CCT)
------------------------

This repository is a small collection of CUDA wrappers and code created to test
out functionality. Some of the features of this code include:

* Support for multiple GPUs, multiple CPU threads, and multiple streams.
* Construction of a GPU object that encapsulates memory allocation and destruction.
* Multi-stream support including synchronization.
* Support for a custom Thrust execution policy to enable memory reuse.
* Wrapper for Unified Memory. (Linux or Windows only)
* Simplification of CPU profiling via NVTX extension.

Building
--------

CCT requires CMake (3.0+) to configure and CUDA 7.5. It has been tested with the
following OSes:

* Windows 7 (x64) with Visual Studio 2013 Update 5
* OSX El Capitan (10.11.3) with XCode 7.2.1
* Ubuntu 14.04 x64 with GCC 9.4.1

The code currently targets compute capability 3.0, 3.5, and 5.2, which should
cover most GPUs within the last few years.
