My solution to the 1brc challenge in C++.

Completes in ~480ms on a dual socket Skylake-X server (32 cores), with ~124 GiB DDR4 spread across
two NUMA nodes. This beats the [fastest](https://github.com/lehuyduc/1brc-simd/blob/main/main.cpp) known solution (AFAIK) on Skylake which completes in
~530ms on the same machine. For comparison, the fastest Java solution completes in ~2.5 seconds on my
test server, which is about 5.2x slower. The solution only uses AVX2 which seems to be the standard
across many 1brc solutions.