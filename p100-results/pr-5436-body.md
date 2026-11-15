**Problem and result.** On a two-socket Linux machine the CPU pool splits itself by NUMA node (0.2.2), but when the
experts come from the arena (the default source whenever RAM holds every expert) their memory is not placed per node:
the engine logs `NUMA split on, but this model's experts come from the arena path: memory is not placed per node (known
limitation)`. `cudaHostRegister` faults the whole arena in from one thread, so it lands on that thread's node and the
other node's slice of the pool reads every row across the socket link. Measured on 2x Xeon E5-2696 v3 (DDR3-1600, one
DIMM per channel) with 2x Tesla P100 16 GB, Unsloth-layout Qwen3.8-Flash-Next (bitlamas Q4_K_XL-IQ4NL, 63.5 GiB of
experts): the CPU pool took 28.6 ms per decode window for ~351 expert-blob reads (~34 GB/s; one socket alone streams
45.9 GB/s).

This PR adds an opt-in `STRATA_ARENA_NUMA=1` (Linux) that places each expert's row halves (`numa_slice_ranges`, the
pool's slices) on their slice's node before the registration. With it the CPU pool takes 14.0 ms per window (same config, one run each), the output
is unchanged (teacher-forced log-probabilities byte-identical), and the placement takes ~2 s at start. Default: nothing
changes except one log line (the known-limitation line names the switch).

**Related discussion.** #3708 (the arena-path limitation on three two-socket boxes). There petski3 described and
measured the same direction on gfx906 (a hook before `cudaHostRegister`, 4 KiB pages, node-pinned touchers,
`move_pages` checks; +29% decode on their box) and offered a PR "if this direction is wanted". This branch arrived at
the same design independently (both follow the file path's `NumaComplementPlacement`; I found their comment only when
searching before this PR), on CUDA, and adds what a fourth box needed: a long-term pin of each chunk before it is
touched (without it 21-32% of one node's half migrated away during the registration) and a PLE table read half per node
(without it a cold start could leave one node too little room for its half). #3730 keeps the automatic pool off on the arena path when
memory is not placed; with this switch the placement can be checked and the split kept. Happy to fold this into
petski3's PR instead if they open one.

**Approach.**
- A `pretouch` hook on `PinnedArena`'s bounds constructor (default empty) runs after the reservation and before the
  registration. `ArenaExpertSource::open` sets it when `STRATA_ARENA_NUMA=1` and the arena is a private,
  whole-registered arena of a native pack (otherwise it logs why not).
- The touch: `MADV_NOHUGEPAGE` first (a 2 MiB THP page holds about four row halves of both slices; with THP the halves
  landed ~50/50, measured), then 8 threads per node, each bound with `set_mempolicy(MPOL_BIND)` to its node (the node's
  file cache is reclaimed instead of spilling; `MPOL_PREFERRED` when the node cannot hold its slice), write one byte per
  page of their node's ranges.
- **Long-term pins from the touching threads, before the touch.** After the touch was 100% local, sampling after the
  registration still found 21-32% of node 1's half on node 0 (863-1313 of 4096 sampled pages, three starts). Reproduced
  outside the engine: at 63.5 GiB, `cudaHostRegister` alone migrated 1.14 M pages. Node 1 has 7329 CMA-type pageblocks
  (14 GiB; `CmaTotal: 0`); an anonymous page there cannot stay under a long-term pin, and the migration takes a free page
  of the other node before it frees file cache on its own. So each touching thread takes an io_uring fixed-buffer
  registration (a FOLL_LONGTERM pin) of each chunk and releases it at once. Revision 2 takes it **before** the chunk is
  touched: the pin's own faults then allocate the pages outside CMA-type memory under the thread's `MPOL_BIND` (the
  node's file cache is reclaimed), and nothing migrates. (The first revision pinned right after the touch: the pages
  migrated while the node had room, which held in most starts but left 96.8% of a half local in a cold start of the
  stock config and 81.7% in one of the author's tuned config, see Validation.) `STRATA_ARENA_NUMA_PREPIN=after` is the first revision's order, `=0` no pre-pin.
  `mlock2(MLOCK_ONFAULT)` keeps reclaim off the fresh pages until the registration; one `mbind(MPOL_BIND, both nodes)`
  keeps NUMA balancing off the VMA. Both need `RLIMIT_MEMLOCK` (`ulimit -l unlimited`, systemd `LimitMEMLOCK=infinity`);
  without it the placement runs unprotected and the log says so.
- **The room check** (`MPOL_BIND` only when the node can hold its slice, else `MPOL_PREFERRED`) leaves out the node's
  CMA-type memory (`/proc/zoneinfo` `cma`) and its locked pages: a long-term pin can use neither, and under `MPOL_BIND` a
  pin with no room has nowhere else to go. The start line prints it (`room 45.0 / 42.7 GiB`).
- **A PLE table locked in RAM is read half per node** (`src/kernels/ngram.cpp`, only with `STRATA_ARENA_NUMA=1` and only
  when the table is mmap-locked). The table is locked before the arena is filled; read from a cold file after one
  `MADV_WILLNEED` from the main thread, all of it (26.8 GiB here, 7.0 M pages) landed in that thread's node, which then had
  31.4 GiB of room for a 31.9 GiB half. Now 8 threads per node fault in one half each (`MPOL_PREFERRED`, `MADV_WILLNEED`
  per 64 MiB piece), and pages an earlier reader left in the file cache on the wrong node are moved (`move_pages`; 3.5 M
  pages in 1.7 s on a warm start). The line `PLE table per NUMA node ...` reports it. `STRATA_PLE_NUMA=0` keeps the plain
  fault-in.
- `move_pages` samples each slice's pages before the registration and after the load; the line
  `STRATA_ARENA_NUMA: row halves placed before the registration: ...` reports both. After the pool is built,
  generate.cpp checks the placement's nodes against the pool's slices.
- Linux only; the io_uring and `mlock2` parts compile out when the headers lack them. Nothing changes where a page's
  contents come from or what is computed: only which node a page is on.

**Validation** (Ubuntu 26.04, Linux 7.0.0-43, CUDA 12.8.93, gcc-14, `-DSTRATA_ENABLE_CUDA=ON
-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60`, 2x E5-2696 v3, 2x Tesla P100 PCIe 16 GB, driver 580.178.04)

Passed, revision 2 (this branch's head):
- CUDA build of `strata`.
- Placement from a cold file cache (`sync; echo 3 > /proc/sys/vm/drop_caches` before every start, as after a boot), the
  stock setup config with `STRATA_ARENA_NUMA=1`, revision 1 (R, commit 2850a55f) and revision 2 (N) interleaved RNNR, a
  fresh server per block. Pages of each slice on its node after the registration and the load, of 4096 sampled:

| block | slice 0 | slice 1 | placement time | load (cold) |
|---|---|---|---|---|
| R1 | **3963** | 4096 | 10.8 s | 260 s |
| N1 | 4096 | 4096 | 1.7 s | 244 s |
| N2 | 4096 | 4096 | 1.7 s | 244 s |
| R2 | 4096 | 4094 | 7.6 s | 252 s |

  On the author's tuned config (the same engine change there) revision 1's order also gave 3346/4096 and 4075/4096 and,
  after a real cold boot, 88.3%; revision 2 4096/4096 on both nodes in every start (two cold, one warm).
- Decode is unchanged by revision 2 (the hot path is the same): time per decode step N/R, decode_ms / (generated -
  accepted), code 1.004, prose 1.009, L8k 0.982, geomean 0.998 (medians in tok/s N / R: 39.82 / 40.23, 30.67 / 31.77, 33.38 /
  33.69; they move with each run's draft acceptance).
- Output: `tools/quant_quality.py run` (as below, CPU experts, FP16 K/V) with revision 2 and `STRATA_ARENA_NUMA=1`: the
  LOGPOS file is byte-identical to the run without the switch (md5 `989d6f4fd04641f8bfa4bb96f9446b0e`); 25.3-29.4 s per
  1000-token segment (revision 1: 25.2-26.0 s).

Passed, revision 1 (the measurements below are of the first revision; revision 2 changes only the start):
- CUDA build of `strata`.
- Output: `tools/quant_quality.py run` (5 x 1000 tokens of `data/quant_quality/v1`, CPU experts, FP16 K/V) with and
  without `STRATA_ARENA_NUMA=1`: the two LOGPOS files are byte-identical (md5 `989d6f4fd04641f8bfa4bb96f9446b0e`).
  The same with GPU experts (`--gpu-experts auto`) together with `STRATA_DOWN_WS=1 STRATA_SPEC_COST=1`: byte-identical
  to the run without them.
- Placement (`move_pages`, 4096 samples per slice) after the registration and the load: slice 0 4096/4096 on node 0,
  slice 1 3896/4096 on node 1 (without the pre-pin: 2783-3233/4096).
- Speed, the CPU path alone (the yardstick above, every expert on the CPU): 39.4 / 39.1 / 37.6 / 38.8 / 39.5 s per
  1000-token segment without, 26.0 / 25.9 / 25.2 / 25.9 / 25.8 s with.
- Speed, 2-GPU layer split decode (`STRATA_DECODE_TIMING=1`, one code prompt, 512 tokens): CPU pool 28.6 -> 14.0 ms per
  window (the stock config, one run each). End to end, the stock setup config with and without the switch, ANNAANNA, a fresh server per block:
  decode tok/s, median [min-max]; prompt reading unchanged (L8k, 15.4K tokens: 308.3 vs 308.2 tok/s). The GPUs are the
  bottleneck of this config, so the CPU's gain shows only in part:

| case | without (4 runs) | with `STRATA_ARENA_NUMA=1` (4 runs) | median gain | with > without |
|---|---|---|---|---|
| code | 38.42 [36.66-38.66] | 40.25 [40.09-40.86] | +4.8% | 4/4 |
| prose | 30.18 [29.65-30.62] | 31.60 [31.18-31.92] | +4.7% | 4/4 |
| chat | 27.62 [23.48-28.29] | 29.93 [29.62-32.16] | +8.4% | 4/4 |
| L8k | 31.48 [31.30-31.54] | 34.35 [33.58-34.93] | +9.1% | 4/4 |

- Without `RLIMIT_MEMLOCK` the engine starts, places, and logs `NOT locked (RLIMIT_MEMLOCK)` / `long-term pre-pin
  unavailable`.

Failed: none observed in the final form. Iterations that did not hold the placement: THP left on, an `mbind` of the
VMA alone, `mlock2` alone (above); the PLE table halved with the touch-then-pin order (slice 1 3559/4096, a warm start);
the pin before the touch without halving the PLE table (slice 0 3240/4096 from a cold cache: the room check found
31.4 GiB on node 0 for 31.9 GiB and chose `MPOL_PREFERRED`).

Not run:
- HIP and SYCL builds (`expert_source.cpp`, `pinned.cu`, `ngram.cpp` are in them): no ROCm or oneAPI here. Hence a draft.
- Windows (the code is under `__linux__`).
- Another two-socket machine, a machine without CMA-type pageblocks, more than two NUMA nodes (the switch then logs
  "off").
- The canonical (Q2_0 `experts.bin`) arena: only native packs are placed, as the pool only splits native layers.
- A PLE table read through the direct reader (`--ple-io direct`) or not locked: not placed (only the mmap-locked table).

**Limits.** Exactly two NUMA nodes with CPUs; a private arena registered whole (not pin by need, not a shared arena,
not the deferred Windows registration). The placement costs ~2 s per start at 63.5 GiB (1.7-1.8 s; revision 1's
touch-then-pin took 7.6-10.8 s). A layer-affine variant (whole
layers on each GPU's socket, so every RAM-to-GPU copy is socket-local) was measured and is not included: with
`iommu=pt` the copies were equally fast either way, and decode lost 11-28% to the pool's remote reads.


🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01N976PznRue6hXdFtmrbJNj
