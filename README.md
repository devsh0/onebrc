My solution to the 1brc challenge in C++.

Completes in ~480ms on a dual socket Skylake-X server (32 cores), with ~124 GiB DDR4 spread across
two NUMA nodes. This beats the [fastest](https://github.com/lehuyduc/1brc-simd/blob/main/main.cpp) known solution (AFAIK) on Skylake which completes in
~530ms on the same machine. For comparison, the fastest Java solution completes in ~2.5 seconds on my
test server, which is about 5.2x slower.

The biggest bottleneck seems to be entry lookup in the hash table; Once an entry is added, we need to
update its record when we come across a line that lists the same city name. This lookup is what's slow.
I am fairly confident an open-addressed hash table would shave off a few tens of MS, but that's yet to
be tested. The IPC sits at 1.44 which means ideally, we can make it ~2.5x faster if all memory stalls are
eliminated. Overall, I think the best solution on this machine would have a runtime in the range of 150ms
to 200ms. The solution only uses AVX2 which seems to be the standard across many 1brc solutions.