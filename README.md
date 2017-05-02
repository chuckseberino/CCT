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

CCT requires CMake (3.0+) to configure and CUDA 8.0. It has been tested with the
following OSes:

* Windows 7 (x64) with Visual Studio 2013 Update 5
* OSX El Capitan (10.11.3) with XCode 7.2.1
* Ubuntu 16.04 x64 with GCC 5.4.0

The code currently targets compute capability 3.0, 3.5, 5.2, and 6.1 which should
cover most GPUs within the last few years.
