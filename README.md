My solution to the 1brc challenge in C++.

**Completes in ~630ms on a dual socket Skylake-X server (32 cores, 64 threads), with ~124 GiB DDR4 spread across
two NUMA nodes. This gets very close to the [fastest](https://github.com/lehuyduc/1brc-simd/blob/main/main.cpp) known
solution (AFAIK) which completes in ~600ms on the same machine. For comparison, the fastest Java solution completes
in ~2.5 seconds on my test server, which is about 4x slower.**

There's still room to squeeze a lot more performance out of this. A few things I was too lazy to optimize:

- Single stream AVX2 for the main loop. Skylake has AVX-512, but the goal was to see how close we can get
to the fastest solution without AVX-512 because most 1brc solutions limit themselves to only using AVX2.
Also, the single SIMD stream (which isn't exactly a "stream" in the traditional sense, just a few SIMD
instructions between many scalar ops) is underutilizing the EUs; SKX has two load (ports 2 and 3) and
3 vector compute units (ports 0, 1, and 5). At the very least, sustaining wo vector operations per cycle
in the steady state should be possible, likely more given the instructions I am using. That said, one
of the goals was to keep the solution as readable as possible. I may come back to optimize this further,
and that could be months in the future. It better be readable then.

- City names mixed with data + bucketed hash table. This is only a tiny problem for the default dataset because most
buckets are never touched so the hot buckets are always resident in L1d. If the dataset changes and we start
touching more buckets or map more entries to the same bucket, this will become a bottleneck.

- ...quiet a few others.
