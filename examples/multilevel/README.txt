multilevelTest

Example code exercising a custom thrust execution policy that enables reusing of
temporary memory for subsequent invocations of thrust. Memory is allocated on first
use on a per-stream basis. Custom policy in enabled by default, but can be set
explicitly with the '-m' flag.

Usage: 
    multilevelTest [-t <numThreads>][-j <numJobs>][-s <numStreams>][-m <0|1>]
