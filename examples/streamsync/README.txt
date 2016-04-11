streamsyncTest

Stream synchronization example with one main stream and 4 worker streams. The code
contains a conditional line which checks the status of the 'syncStreams' boolean
and if it is false, will not synchronize stream 3. This unsynchronized behavior can
be seen in the NVIDIA Visual Profiler to examine the behavior. Streams are synced
by default, and can be turned off with '-s 0'.

Usage: 
    streamsyncTest [-t <numThreads>][-j <numJobs>][-s <0|1>]
