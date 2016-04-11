hardwareTest

Provides a basic test for running a kernel adjusting the blocksize. The kernel
performs rudimentary computation to keep the device busy for a specified period of
time so that it can be examined using NVIDIA Visual Profiler (NVVP).

Usage: 
    hardwareTest [-t <numThreads>][-j <numJobs>][-b <blockSize>][-s <numStreams>]
