A fourth two-socket box on the arena path, CUDA this time, with one more failure mode for the per-node placement. The change is up as a draft: #5436.

### Machine

- 2x Xeon E5-2696 v3 (Haswell, 18 cores each, AVX2), 125.7 GiB DDR3-1600 (8x 16 GB, one DIMM per channel), 2 NUMA nodes.
- 2x Tesla P100 PCIe 16 GB, one per socket (GPU0 PCIe 3.0 x8, GPU1 x16), both capped at 150 W. Ubuntu 26.04, kernel 7.0.0-43, driver 580.178.04, CUDA 12.8, `iommu=pt`.
- Qwen3.8-Flash-Next, Unsloth layout (bitlamas Q4_K_XL-IQ4NL), native pack, 63.5 GiB arena, layer split 0-23 / 24-47.

### The registration can undo the placement

Placing each slice's halves before `cudaHostRegister` (the direction petski3 described above) worked right after the touch: `move_pages` sampling showed 100% of both slices on their nodes. After the registration, only 68-79% of node 1's half was still on node 1 (three engine starts: 3233, 2984 and 2783 of 4096 sampled pages); how much moved varied from start to start.

The cause is in the kernel, not the engine. Node 1 has 7329 CMA-type pageblocks (14 GiB, `/proc/pagetypeinfo`, although `CmaTotal` is 0). An anonymous page in one of them cannot stay under a long-term pin, so `cudaHostRegister`'s pin migrates it, to the page's own node only while that node has free memory. After its 31.6 GiB half, node 1 has none, so the pages go to node 0. A 30-line test outside the engine reproduces it: map 63.5 GiB, touch the halves per node, register: 1.14 M pages migrated and 14.5% of node 1's half ended on node 0. A box without CMA-type pageblocks (petski3's stayed at 100%) does not see this, which may be why the same placement measures differently from box to box.

The fix in #5436: each touching thread takes an io_uring fixed-buffer registration (also a long-term pin) of every chunk right after touching it, and releases it at once. Those pages migrate then, while their node still has room. With it, 94-100% of each half stays local after the registration and the load. `mlock2(MLOCK_ONFAULT)` keeps reclaim off the fresh pages until the registration, and one `mbind` keeps NUMA balancing off the arena. All of it needs `ulimit -l unlimited` (`LimitMEMLOCK=infinity`); without it the placement runs as before and the log says so.

### Numbers (same binary, interleaved, a fresh server per run)

- Stock setup config with and without the switch, ANNAANNA, decode tok/s medians of 4: code 38.42 -> 40.25, prose 30.18 -> 31.60, chat 27.62 -> 29.93, 15K-token prompt 31.48 -> 34.35 (+4.7 to +9.1%, all 4 pairs faster in every case). Prompt reading unchanged.
- The CPU pool's time per decode window (`STRATA_DECODE_TIMING`, stock config, one run each): 28.6 -> 14.0 ms. With every expert on the CPU (the `quant_quality.py` yardstick): 39 -> 26 s per 1000-token segment.
- Output: the yardstick's teacher-forced log-probabilities are byte-identical with and without the switch.

### On the GPU side (relevant to #2242 and to petski3's note on host-to-device copies)

With the IOMMU translating (`DMA-FQ`), a pinned copy to GPU1 from the far node ran at 10.29 GB/s against 11.70 from its own node; with `iommu=pt` it is 12.40 against 12.42 GB/s, also for 3 MiB blobs. I also tried whole layers on each GPU's node (layers 0-23 on node 0, 24-47 on node 1), so that every expert a GPU copies in is local: prompt reading did not change (482.7 vs 481.0 tok/s at 32K) and decode lost 11-28%, because the pool then reads half its rows remotely. On this box the CPU-side placement is what matters.

### Not run

HIP and SYCL builds of #5436 (no ROCm or oneAPI here), a hugetlb arena, more than two nodes, other CPUs. If the 6226R, Z840 or ML350 owners can run the branch, the placement line at start (`STRATA_ARENA_NUMA: row halves placed ... after the registration and the load: ...`) shows whether their registration moves pages too.

Measured and prepared with Claude Code on this box; the numbers are from its logs.
