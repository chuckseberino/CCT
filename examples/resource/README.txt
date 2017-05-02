resourceTest

Example code that demonstrates running two overlapping kernels since they don't
interfere with each others limiting resource. One kernel uses up all shared memory
and is limited in occupancy (25%). The other kernel performs compute using a small
block size, also limiting occupancy. But together these kernels achieve 100% occupancy
by running concurrently in separate streams.

Usage: 
    resourceTest [-t <numThreads>][-j <numJobs>][-s <numStreams>]
