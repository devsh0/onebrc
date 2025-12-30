My solution to the 1brc challenge in C++.

**Completes in ~750ms on a dual socket Skylake-X server (32 cores, 64 threads), with ~124 GiB DDR4 spread across
two NUMA nodes. This gets very close to the [fastest](https://github.com/lehuyduc/1brc-simd/blob/main/main.cpp) known
solution (AFAIK) which completes in ~600ms on the same machine. For comparison, the fastest Java solution completes
in ~2.5 seconds on my test server, which is about 3x slower.**

I have only tested this on the default dataset. Claims above may not hold for the 10k dataset.

There's still room to squeeze a lot more performance out of this. A few things I was too lazy to optimize:

- Single stream AVX2 for the main loop. Skylake has AVX-512, but the goal was to see how close we can get
to the fastest solution without AVX-512 because most 1brc solutions limit themselves to only using AVX2.
Also, the single SIMD stream (which isn't exactly a "stream" in the traditional sense, just a few SIMD
instructions between many scalar ops) is underutilizing the EUs; SKX has two load (ports 2 and 3) and
3 vector compute units (ports 0, 1, and 5). At the very least, sustaining wo vector operations per cycle
in the steady state should be possible, likely more given the instructions I am using. That said, one
of the goals was to keep the solution as readable as possible. I may come back to optimize this further,
and that could be months in the future. It better be readable then.

- City names mixed with data + bucketed hash table. This is not a problem for the default dataset because most
buckets are never touched so the hot buckets are always resident in L1d. If the dataset changes and we start
touching more buckets or map more entries to the same bucket, this will become a bottleneck. If we don't
increase the bucket count, the hashtable grows vertically which means we'll spend more time searching in
buckets. If we increase the number of buckets, the hash table grows horizontally and would likely miss L1d
very frequently.

- Lots of cross-NUMA traffic. This was a bit of a surprise. Workers are pinned to specific CPUs and default NUMA
policy in Linux should allocate physical memory for a given CPU in the local NUMA node. But I observed \>100 MB
cross-NUMA traffic in the main loop. I have no idea where that is coming from. In the merge stage these
transactions are expected, but not in the parse loop.
