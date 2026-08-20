My solution to the 1brc challenge in C++.

Completes in ~450ms (not including unmap cost) on a dual socket Skylake-X (32 cores, 64 threads,
frequency locked at 2.1GHz), with ~124 GiB DDR4 spread across two NUMA nodes. Uses AVX2 only.