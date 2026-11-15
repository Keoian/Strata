# OPTLOG: what was tried, measured, kept and rejected

Every entry: what changed, the measurement (with what it ran on), the decision. Hypotheses are marked as such.
Raw logs and scripts are in `p100-results/`.

## Phase 1: audit and environment (2026-11-06)

### Hardware facts measured
- GPUs: GPU0 `04:00.0` on NUMA node 0, GPU1 `81:00.0` on NUMA node 1; `nvidia-smi topo -m`: `SYS` between them
  (no P2P path besides PCIe + QPI). Both capped at 150 W by `p100-power-cap.service` (RUNBOOK section 2).
- **GPU0 negotiates PCIe 3.0 x8** (max x16); GPU1 x16. Cause found: GPU0's root port `00:02.2` is x8 only (CPU0
  port 2 bifurcated x8x8, `00:02.0` empty); CPU0's x16 port `00:03.0` holds the Radeon HD 7450. **Plan (owner):** swap
  the P100 and the Radeon after this effort, once a new P100 cooler is printed (the stock shroud hits the RAM in that
  slot). This whole effort therefore runs with GPU0 at x8; see RUNBOOK section 1 for the after-swap checklist.
- RAM: 8x 16 GB DDR3-1600 RDIMM, one per channel. MemTotal 121.5 GiB; node0 64,055 MB, node1 60,348 MB. The gap is
  the kdump `crashkernel` reservation (4.25 GiB, 4 GiB of it at 0x1f7f000000, in node 1).
- IOMMU: every group `DMA-FQ` (translating). Strata's docs (DETAILS.md "Two sockets and NUMA", #3201) measured pinned
  copies 7.7-9.2 GB/s translated vs 12.3 GB/s with `iommu=pt`.

### Pinned copy bandwidth, IOMMU DMA-FQ (`p100-results/phase1-audit/pcie-bw-iommu-dmafq.txt`)
`pcie_bw <gpu> 256 20` under `numactl --membind=N --cpunodebind=N`, median of 20 copies of 256 MiB, GB/s = 1e9 B/s:

| | host memory node 0 | host memory node 1 |
|---|---|---|
| GPU0 (node 0, x8) H2D / D2H | 6.15 / 6.73 | 6.16 / 6.74 |
| GPU1 (node 1, x16) H2D / D2H | 10.29 / 10.89 | **11.70 / 12.12** |

Reading: the x8 link bounds GPU0 whatever node the memory is on; for GPU1 a host buffer on the far socket costs 12%.

### KEPT: IOMMU passthrough for the GPUs (`identity` groups now, `iommu=pt` from the next boot)
Same tool, same order, after switching both GPU IOMMU groups to `identity` (RUNBOOK 2b):

| | DMA-FQ, node 0 / node 1 host memory | identity, node 0 / node 1 | change |
|---|---|---|---|
| GPU0 (x8) H2D | 6.15 / 6.16 | 6.17 / 6.17 | 0% (link-bound) |
| GPU0 (x8) D2H | 6.73 / 6.74 | 6.59 / 6.59 | -2% |
| GPU1 (x16) H2D | 10.29 / 11.70 | **12.40 / 12.42** | +20% / +6% |
| GPU1 (x16) D2H | 10.89 / 12.12 | **13.09 / 13.15** | +20% / +8% |

At an expert blob's size (3 MiB x 400 copies, identity): GPU1 12.16 / 12.18 H2D, GPU0 6.03 / 6.11. **The 12%
cross-socket cost was IOMMU translation, not QPI bandwidth**: with passthrough the node a pinned host buffer sits on
does not change copy bandwidth to either GPU. NUMA placement still matters for the CPU pool's own reads (Strata's
NUMA split handles that). Decision: kept. End-to-end effect is measured inside the baselines (all of them run in
passthrough).

### KEPT: kdump reservation released (MemTotal 121.5 -> 125.7 GiB, nodes 62.80 / 62.93 GiB)
kdump was configured off (`USE_KDUMP=0`); the reservation was dead weight on node 1. Released live and dropped from
the kernel command line for later boots.

### sm_60 `STRATA_DP4A` aliasing: no kernel affected on this toolkit; stock code kept, every build gated
- #4274's probe reproduced here exactly: 62,997 / 62,238 / 61,325 / 61,324 mismatches of 65,536 with `"+r"`, 0 with
  `"+&r"`, both GPUs (`p100-results/parity/`).
- `p100-results/tools/vmad_alias_scan.py` flags a VMAD that reads bytes 1-3 of its own accumulator register (the
  aliasing's exact SASS sign). Control: 12 hits in the `"+r"` probe, 0 in the `"+&r"` probe.
- The stock engine (`build-p100/strata`, CUDA 12.8.93, sm_60): **0 of 1,011,704 VMAD** aliased. With the fix: 0 too.
- The fix still changes register allocation: 61 of 87 CUDA objects have different SASS
  (`p100-results/parity/sass-diff-dp4a-fix.txt`). Integer results are the same either way.
- Decision: keep upstream's `dp4a.hpp` (the baseline stays upstream code); run `vmad_alias_scan.py` on every engine I
  build, and refuse a build with any hit. Reconsider #4274 if a new kernel trips it.

### Environment decisions
- Driver 580.178.04 (Ubuntu server branch, prebuilt modules). CUDA 12.8.1 (12.9 avoided per Strata's glibc note;
  12.8 needed a header patch on glibc 2.43 too - RUNBOOK section 3). Host compiler g++-14.

### Upstream state checked (no duplicate work)
See `p100-results/upstream-related-threads.md`. Static reading of `src/kernels/cuda/iq_kernels.cu`
(`STRATA_GU_FMTS` has 12 Q4_K and 13 Q5_K, `STRATA_D_FMTS` has 20 IQ4_NL; gate/up and down kernels are dispatched
separately), `src/kernels/cpu/native_expert.cpp` (`native_fmt` is generic over ggml-cpu traits) and `tools/iq_pack.py`
(no expert-type allowlist) says Strata should already run the bitlamas Q4_K/Q5_K + IQ4_NL expert pairs.

**Confirmed on synthetic weights** (`native_expert_parity --synthetic`, both P100s, commit `0b21037d` adds them to
ctest; `p100-results/parity/`):

| pair | CPU rel. L1 vs float | GPU rel. L1 vs float | CPU vs GPU | verdict |
|---|---|---|---|---|
| q4_K/iq4_nl (layers 0-1, 3-47) | 1.32e-02 | 1.24e-02 | 1.53e-02 | ok (limit 3e-02) |
| q5_K/iq4_nl (layer 2) | 1.36e-02 | 1.17e-02 | 1.42e-02 | ok |
| q4_K/q5_1 (Unsloth control) | 1.29e-02 | 1.21e-02 | 1.41e-02 | ok |

GPU prompt-path dequantizers: 0 of 4,915,200 values differ from ggml `to_float`. No new kernel written (the spec's
"do not duplicate upstream"). Upstream subset `gguf_split_test expert_layout_test native_grouped_parity
iq_avx2_parity native_dense_parity numa_pool_test`: 6/6 passed. Still to do: the same tool on the real shards.

### Opt-in sm_60 kernels: parity on these GP100s before any end-to-end use (`p100-results/parity/`)

| switch | check | result |
|---|---|---|
| `STRATA_Q8_SM60=1` | `tools/q8_sm60_check.cu`, both GPUs | 57 shape x column cases PASS (rel < 1e-5); 450-570 GB/s on the big Q8_0 matrices at T=1, head 568 GB/s |
| `STRATA_PATTN_SM60=1` | `qsa_pattn_parity` | 468 cases, 0 differ |
| `STRATA_SELECT_LONG=1` | `qsa_select_long_parity` (with `STRATA_SCORES_DX=1 STRATA_TOPK_MULTI=1`) | 172 cases, 0 differ |
| `STRATA_MMQ_KQ=1` | `prefill_mmq_kquant_test` | Q4_K gate/up rel L2 0.0017, Q5_K 0.0022, screen passed (without the switch the test says "not covered": the library is opt-in) |
| `STRATA_DOWN_WS=1` | no dedicated test; `native_expert_parity` prints the same numbers on/off, which does not prove the path ran | pending: end-to-end LOGPOS rows must be byte-identical (it claims mode 0's bits) |

### CPU pool ceiling: `kq_cpu_bench` (the engine's own row entry points, random experts far beyond L3)

| placement | q4_K/iq4_nl, 1 token per expert | 3 tokens | q4_K/q5_1 (Unsloth), 1 token |
|---|---|---|---|
| node 0 only, 1 thread | 5.5 GB/s | 2.9 | 7.0 |
| node 0 only, 8 threads | 36.3 | 18.6 | 36.9 |
| node 0 only, 17 threads | **45.9** | 37.3 | 45.9 |
| both nodes, memory interleaved, 17 threads | 62.9 | 37.5 | - |
| both nodes, memory interleaved, 34 threads | **77.9** | 66.7 | - |

One socket reaches ~90% of DDR3-1600 x4 (51.2 GB/s) at one token per expert. IQ4_NL downs cost more ALU per byte than
Q5_1 on one thread but not once the socket is memory-bound. Both sockets give 1.7x one socket even with interleaved
(half remote) memory; Strata's NUMA split, which reads only local memory, should do better. Implication for the
"keep a GPU's experts on its own socket" idea: that placement would cap each layer's CPU experts at one socket's
~46 GB/s. To be measured end to end, not assumed.

### SHELVED (low ceiling): huge pages under the NUMA split
With the NUMA split, `src/core/expert_source.cpp` marks the complement arena `MADV_NOHUGEPAGE` so each expert's halves
can sit on different nodes at 4 KiB granularity. A huge-page arena would need a different layout (all node-0 halves
in one region, node-1 halves in another). Ceiling check first: one socket already streams experts at 45.9 GB/s (~90%
of 51.2 GB/s) with 4 KiB pages; `kq_cpu_bench` 17 threads node 0 with THP `always` 46.2 / 46.4 GB/s vs 45.9
(`madvise`). The bench allocates per expert, so only ~0.35 GiB of its 4 GiB got huge pages: inconclusive, but the
upside is bounded at ~10% of the CPU side. Not pursued unless the CPU pool turns out to be the critical path.
Also noted: `kernel.numa_balancing = 1` (Ubuntu default); an A/B with it off is on the Phase 4 list.

### GPU expert decode, one layer's top-10 at one token (`native_expert_parity --synthetic --bench`, GPU1)

| pair | default | `STRATA_DOWN_WS=1` |
|---|---|---|
| q4_K/iq4_nl (bitlamas) | 152.0 us, 182 GB/s of weights | **131.2 us, 211 GB/s (-13.7%)** |
| q4_K/q5_1 (Unsloth) | 185.4 us, 166 GB/s | 183.5 us (not applicable) |

Microbenchmark only; end-to-end A/B and bit identity still needed. All of these run at ~25-29% of HBM2's 732 GB/s:
GP100 emulates dp4a with four VMADs, so the grouped expert kernels are ALU-bound. Q4_K gate/up is two thirds of the
bytes: the main kernel-level target for Phase 4 if the window profile shows GPU experts on the critical path.

### Static reading for Phase 2 and 3 (unconfirmed until measured)
- Pascal decode attention (`attn_chunk_kernel_pre75`, `src/kernels/cuda/qsa_decode_attn.cu`) already reads INT8 K and
  rotated Q4_0 V straight from the pools into registers; no FP16 copy of the cache. QSA attends only the indexer's
  top-k cells. So the q4p core idea is already in Strata; what is left is conversion micro-work (scalar int8/nibble
  to float per element). Worth it only if the GPU stage profile shows attention as a real share of a window.
- `STRATA_SPEC_COST=1` (opt-in) is an online, cost-aware version of the learned stop rule: acceptance by draft
  probability bucket (decayed), measured round cost per width and per context bucket, the width with the best
  expected tokens per ms, and delayed observations for gated sampled proposals so a window is never steered by its own
  draws. Phase 3 starts with an A/B of it against `--spec-min-p 0.5`, and a `--spec-min-p` sweep.

### sm_60 `STRATA_DP4A` accumulator aliasing (open upstream PR #4274)
`include/strata/kernels/dp4a.hpp` declares the VMAD accumulator `"+r"`; #4274 changes it to `"+&r"` (early clobber)
after a probe that FORCES the aliasing showed ~94% wrong dot products. The PR author did not show a real kernel
compiling to the aliased pattern. Check here: build every CUDA object with and without the fix and compare SASS.

- Strata already splits the CPU expert pool per NUMA node (0.2.2, `STRATA_NUMA_POOL`, rows of every CPU-side expert
  halved across nodes). Not re-implemented. A layer-affine placement (each GPU's layers' experts on its own node) is a
  Phase 4 experiment, to be measured against it.
- Strata's MTP gate already keeps sampling exact (`include/strata/core/spec_prob.hpp`, `gated_proposal.hpp`). A learned
  stop rule would replace `--spec-min-p`, not fix a bias. Phase 3.

## Phase 2: the model runs on both P100s (2026-11-06)

- Download: all four shards match the repo's `SHA256SUMS` (`p100-results/model/sha256-check.txt`). Pack:
  `tools/iq_pack.py --compat-bf16` as-is (46 s; 195 small projections rounded to BF16, experts and PLE unchanged;
  layer 11 split across shards 2/3 -> `native_experts.txt` v4).
- `native_expert_parity` on real layers 0, 2 (Q5_K gate/up), 11 (split) and 47, both GPUs: 0 failures; GPU dequant
  bit-exact against ggml's `to_float` (`p100-results/parity/native_expert_parity_real_layers.txt`).
- Stock 2-GPU smoke (`p100-results/runs/smoke-2gpu`): layer split auto K=24 (CUDA0 layers 0-23, 3484 slots 9.03 GiB;
  CUDA1 24-47 + head + MTP, 2908 slots 7.49 GiB). Coherent answers (the sheep puzzle answered 18), tool call
  returned, `/metrics` request 0 = the newest request (prompt counts match). Decode 25-38 tok/s on short cases.
- Engine choices seen in the log: `--ple-io ram` is chosen by the residency policy (121 GiB available); the PLE lock
  fails (RLIMIT_MEMLOCK 0); the arena gets THP when free 2 MiB blocks are plentiful (right after a previous engine
  exited) and 4 KiB pages otherwise (page cache full); **"NUMA split on, but this model's experts come from the
  arena path: memory is not placed per node (known limitation)"**.

### Window profile, stock (`STRATA_DECODE_TIMING=1`, code prompt, 512 tokens, `p100-results/runs/probe-timing-stock`)
88.7 ms/window, 3.39 tokens/window (38.2 tok/s): verify 82.8 of it. CPU pool 28.6 ms/window for ~351 expert-blob
reads of 2.76 MB = ~34 GB/s, below one socket's 45.9 GB/s ceiling: the arena had been faulted in by
`cudaHostRegister` from one thread, so it sat mostly on one node and the other node's pool slice read remotely.

### KEPT (opt-in `STRATA_ARENA_NUMA=1`): NUMA placement of the expert arena
New code (`src/core/expert_source.cpp` ArenaNumaPlace, a `pretouch` hook in `PinnedArena`): before the registration,
each expert's row halves (the pool's slices, `numa_slice_ranges`) are first-touched by 8 threads per node, bound with
`set_mempolicy(MPOL_BIND)` so a node's own file cache is reclaimed instead of spilling; the arena is set
`MADV_NOHUGEPAGE` first, because a 2 MiB THP page holds ~4 halves of both slices (with THP the halves landed ~50/50,
measured). `move_pages` sampling after the registration and the load: 4091/4096 slice-0 pages on node 0, 4075/4096
slice-1 pages on node 1. Start cost 6.3 s. Placement only; no computation changes (same accepted drafts and hits).

| arena (2 GPUs, KV int8, 32K) | CPU pool ms/window code / prose | decode tok/s code / prose |
|---|---|---|
| stock (THP, one-thread fault-in) | 28.6 / 17.3 | 38.2 / 29.5 |
| THP + touched from both nodes (~50/50 per slice) | 19.0-20.2 / 12.0-12.7 | 40.8-40.9 / 30.7-31.1 |
| `STRATA_ARENA_NUMA=1` (4 KiB, halves on their nodes) | **14.0 / 8.8** | 41.1 / 31.0 |

Single-run screens (same prompts, greedy); ABBA confirmation in Phase 4. The CPU pool time halved, but decode moved
little: the host then waited on the GPUs ("GPU-reach wait" rose), so the CPU had left the critical path.

### Window profile with the arena placed (`STRATA_VERIFY_PROFILE=1`, `p100-results/runs/probe-gpuprof-numa`)
Stage 0 35.9 + stage 1 45.6 ms/window (serial). Largest parts: **waitB 21.3 ms** (the GPU waits for flag B, the
PCIe share's DMA: CUDA1's share was 0.34, sized for a CPU assumed to stream 65 GB/s), expert kernels 15.8, Q8_0
input projections 9.5, hyper-connection norm/down/up ~9.7, head 3.9 (CUDA1), k/v+norm-rope 2.7, out-proj 5.1.

### KEPT (config): `--pcie-frac 0` with the placed arena
A 2.76 MB blob takes ~223 us over x16 (445 us over GPU0's x8) while the placed pool computes one in ~40 us of wall
time. `--pcie-frac 0`: 82.6 -> 65.5 ms/window, **code 41.1 -> 51.4 tok/s, prose 31.0 -> 37.5** (single-run screen;
GPU and CPU round experts differently, so the drafts accepted differ by 1-11 and quality is re-checked in Phase 3).

### Screen round 1: opt-in switches on the placed arena + `--pcie-frac 0` (single runs, `p100-results/runs/screen-*`)
Same two greedy prompts (LRU-cache code, 400-word prose), 512 tokens, 2 GPUs, KV int8, 32K. Base = `STRATA_ARENA_NUMA=1
--pcie-frac 0`: code 51.4, prose 37.5 tok/s.

| switch | code | prose | drafts accepted (code / prose) | note |
|---|---|---|---|---|
| base | 51.4 | 37.5 | 363/429, 260/458 | |
| `STRATA_Q8_SM60=1` | **59.6 (+16%)** | **41.8 (+11%)** | 360/424, 243/473 | not bitwise; packed copies cost 6.6 GiB of slots over both cards (6392 -> 3830) |
| `STRATA_DOWN_WS=1` | 52.6 (+2%) | 39.2 (+5%) | 363/429, 260/458 (same) | bit for bit, as documented |
| `--pipeline-windows 2` | 52.5 | 36.8 | 358/432, 251/484 | **did not run**: the server's instruct default `presence_penalty: 1.5` makes every request serial |
| `--trim-stage-weights` | 53.9 (+5%) | 38.1 (+2%) | 365/417, 260/467 | +762 slots, all on CUDA1 (auto split loads CUDA0 untrimmed) |

Finding: `serve/server.py` MODEL_CARD_SAMPLING gives instruct requests (reasoning off) `presence_penalty 1.5` unless the
request or config says otherwise, also under temperature 0. A penalty forces the serial decode loop
(`pl_serial`, generate.cpp), so `--pipeline-windows` only helps thinking requests or clients that set their own sampling.

### Screen round 2: the four kept switches together (`p100-results/runs/screen-exp-combo*`)
combo = `STRATA_ARENA_NUMA=1 STRATA_Q8_SM60=1 STRATA_DOWN_WS=1 --pcie-frac 0 --trim-stage-weights`.

| split | slots CUDA0 + CUDA1 | ms/window code | code | prose |
|---|---|---|---|---|
| stock (auto 24, nothing of the above) | 3484 + 2908 | 88.7 | 38.2 | 29.5 |
| combo, `layer_split auto` (picked 19) | 2357 + 2590 | 54.7 | 61.5 | 44.4 |
| combo, `layer_split "24"` | **3733 + 2915** | 52.6 | **64.0** | 44.4 |
| combo, `layer_split "22"` | 3862 + 2783 | 53.0 | 63.5 | 45.4 |

An explicit split lets CUDA0 load trimmed too (+1.7k slots over auto). 22 vs 24 is within single-run noise; 24 kept
as the working base, to be settled by ABBA.

### Screen round 3 on the combo, split 24 (`p100-results/runs/screen-exp-c24-*`, single runs)
Base (instruct default `presence_penalty 1.5`): code 64.0, prose 44.4.

| change | code | prose | verdict |
|---|---|---|---|
| PLE table locked (`prlimit --memlock=unlimited`, "PLE table locked in RAM") | 63.9 | 44.4 | neutral: the table's pages were resident anyway; not kept for speed |
| `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` | 64.0 | 44.4 | neutral: 2x the swaps, CPU experts 8.5 -> 6.5 per layer-window, decode unchanged: the CPU is off the critical path |
| `--kv k8v4` | 64.0 | 44.4 | neutral at 32K (+34 slots); its use is long context, tested there |
| `kernel.numa_balancing=0` | 64.2 | 44.5 | neutral, not kept |
| requests with `presence_penalty: 0`, base | 65.3 | 49.9 | (other text: more drafts accepted on prose) |
| same requests, `--pipeline-windows 2` | **72.8** | 46.4 | code +11.5%, prose -7%: needs the ABBA over more cases |
| same, `--pipeline-windows 2` + `STRATA_MTP_STAGE0=1` | 71.2 | 42.7 | worse than the drafter on the last card |

Decode times are now near-constant run to run (8.00-8.01 s for 512 tokens): the window is GPU-bound.

## Phase 3: quality (`p100-results/quality/`, `tools/quant_quality.py`, 5 x 1000 tokens of data/quant_quality/v1)
Yardstick runs use CPU experts (one VRAM slot), FP16 K/V unless named, GPU1 alone, no adaptation, no PCIe share.

- `STRATA_ARENA_NUMA=1` vs stock: the LOGPOS files are **byte-identical** (md5 989d6f4f...; top-1 100.00%, KL 0,
  PPL 4.112 both). Placement only, as designed. Side result: with every expert on the CPU (this yardstick's setup)
  a 1000-token segment took 39 s stock and 26 s placed: the CPU expert path is 1.5x faster when it is the bottleneck.

### The yardstick's noise floor on this MoE, and the KV formats / Q8_SM60 against it
All against `S-stock-fp16` (CPU experts, FP16 K/V) unless said; 4995 positions.

| run | what differs | top-1 | KL | KL p99 | PPL |
|---|---|---|---|---|---|
| S-stock-fp16 | reference | - | - | - | 4.112 |
| S-numa-fp16 | `STRATA_ARENA_NUMA=1` | 100.00% | 0 (byte-identical) | 0 | 4.112 |
| **G-stock-fp16** | **the experts on the GPU (same weights; it rounds differently): the floor** | **94.93%** | **0.0278** | 0.397 | 4.023 |
| S-q8sm60-fp16 | `STRATA_Q8_SM60=1` | 94.79% | 0.0275 | 0.375 | 4.067 |
| S-stock-int8 | `--kv int8` | 95.12% | 0.0293 | 0.458 | 4.012 |
| S-stock-k8v4 | `--kv k8v4` | 93.05% | 0.0479 | 0.684 | 4.020 |
| S-stock-q4_0 | `--kv q4_0` | 92.13% | 0.0606 | 0.957 | 4.018 |

Paired PPL with the same K/V reference (int8): k8v4 +0.2%, q4_0 +0.15% (both inside the +1% gate).

Reading: with 10 of 512 experts routed per layer, a rounding-level change flips near-tied routes and the logits move
by KL ~0.03 on this text. The CPU-vs-GPU difference of the *same* expert weights is 0.028, so the spec's 0.002 KL
target is below the measurement floor of any non-bitwise change here. Q8_SM60 and int8 K/V are at the floor; k8v4 and
q4_0 are above it (KL rises with K/V precision loss) while PPL does not move. The PPL of FP16 K/V with CPU experts
(4.112, agent segment 4.96 against 4.50-4.73 elsewhere) is the outlier of the set. llama.cpp's CPU run (REF) decides
which of these is nearest the model.

### Against llama.cpp (the implementation check): `p100-results/quality/REF-llama-cpu.tsv`
llama.cpp at Strata's pin (3cf03257), CPU, the same 5 x 1000 token ids teacher-forced in one batch per segment
(`p100-results/tools/llama_logpos.cpp`): PPL **4.017**.

| Strata run | top-1 vs llama.cpp | KL vs llama.cpp | PPL (llama.cpp 4.017) |
|---|---|---|---|
| S-stock-fp16 (CPU experts, FP16 K/V) | 92.77% | 0.0469 | 4.112 (+2.4%) |
| S-stock-int8 | 92.77% | 0.0481 | 4.012 (-0.1%) |
| S-stock-k8v4 | 91.69% | 0.0609 | 4.020 (+0.1%) |
| S-stock-q4_0 | 91.71% | 0.0677 | 4.018 (+0.0%) |
| S-q8sm60-fp16 | **93.09%** | **0.0405** | 4.067 |
| G-stock-fp16 (GPU experts) | 93.17% | 0.0445 | 4.023 (+0.15%) |
| G-q8sm60-fp16 | **93.31%** | **0.0406** | 4.029 (+0.3%) |
| G-downws-fp16 | = G-stock (byte-identical LOGPOS) | | 4.023 |

- The model runs correctly on Strata: PPL within 0.15% of llama.cpp (GPU experts), KL ~0.04 = two engines' rounding
  through a top-10-of-512 router (Strata's own CPU-vs-GPU floor is 0.028).
- **KEPT `STRATA_Q8_SM60=1`**: nearer llama.cpp than the stock kernels on both expert paths (KL 0.0405 vs 0.0469 CPU,
  0.0406 vs 0.0445 GPU), paired PPL vs stock GPU +0.15%.
- **KEPT `STRATA_DOWN_WS=1`**: byte-identical to stock end to end (GPU experts), as its docs say.
- K/V: k8v4 +0.1% PPL vs llama.cpp, q4_0 +0.0%, int8 -0.1%; KL ordering int8 < k8v4 < q4_0. All pass the +1% gate.

### Screen round 4 (single runs on the combo split 24; `p100-results/runs/screen-exp-c24-*`), and why it drifted
Base this morning 64.0 / 44.4 (code / prose); a base re-run at the end of round 4 gave 61.0 / 43.1, with the CPU pool at
18.2 instead of 15.2 ms per window. Not the GPUs (36-40 C, 1328 MHz, no throttle), not swap (34 pages), not NUMA
balancing (an `mbind` of the arena VMA did not change it). The arena placement had decayed after the registration:
`move_pages` sampling showed 863-1313 of 4096 node-1 sample pages on node 0 after `cudaHostRegister`, 100% local just
before it. Reproduced outside the engine (`scratchpad/pin_test.cu`): at the full 63.5 GiB, `cudaHostRegister` alone
migrated 1.14 M pages and left 14.5% of the node-1 half on node 0. Cause: node 1 has 7329 CMA-type pageblocks (14.3 GiB;
`/proc/pagetypeinfo`; CmaTotal 0, so not a reserved CMA area), anonymous pages placed there cannot stay under a
long-term pin, and the pin's migration targets the page's node only while it has free memory; after the 31.6 GiB half
it has none, so they go to node 0. How many depends on how full the node is: the "drift".

**Fix (code, under `STRATA_ARENA_NUMA=1`)**: the toucher threads take io_uring fixed-buffer pins (FOLL_LONGTERM)
chunk by chunk right after touching and release them at once, so the CMA pages migrate while their node still has
room; `mlock2(MLOCK_ONFAULT)` keeps reclaim off the fresh pages, one `mbind` keeps NUMA balancing off the VMA. Result:
95.1% of the node-1 half local (200 of 4096 samples off), CPU pool 14.97 ms/window, **code 63.9, prose 44.4 tok/s**
(the morning's numbers back). Needs RLIMIT_MEMLOCK (the service sets LimitMEMLOCK=infinity); without it the touch still
runs and says the pin is unavailable. +2.7 s start.

Round 4 against the decayed base (61.0 / 43.1), so +-3% is noise here: corpus-trained profile 61.4 / 43.7 (more VRAM
hits = more work on the GPUs, the bottleneck; rejected); `STRATA_GU_SWIGLU` 62.1 / 44.8; `STRATA_QSA_ROWS` 62.0 / 43.3;
`STRATA_ATTN_DAILY` 62.1 / 43.6 (bitwise, same drafts); `STRATA_TSUM + GR_Q8_PFS` 60.2 / 44.5 (kernel bench: HC read
-10% at 4 tokens, not visible end to end); `STRATA_GR_V3` 63.0 / 44.8 (rounding level; not taken without a cleaner
A/B); `STRATA_SPEC_COST` 60.5 / **47.6** (shorter windows: code -1%, prose +10%); `--spec 3` 61.7 / 46.5; `--spec 5`
61.5 / 42.3. `tools/calibrate.py` on the combo: 32 pool workers best (54.4 / 50.0 vs 21: 53.0 / 49.7, 16: 51.9, 8:
43.2); adapt-tier candidates within noise (its PCIe/draft-floor sweep output was lost to a `tail`; not re-run).
Kept for the final ABBA: the combo; `STRATA_SPEC_COST=1` as a third arm.

### Coding suite: HumanEval, 164 problems (`p100-results/runs/humaneval-{stock,best}`, `p100-results/tools/humaneval_serve.py`)
openai/human-eval `HumanEval.jsonl.gz` (sha256 b796127e...), each problem one greedy chat request with the server's
defaults (reasoning off; the model card's instruct sampling with temperature 0), the reply's code + the problem's own
tests in a subprocess (python3 -I, fresh temp dir, 15 s). The harness passes all 164 canonical solutions.

| config | pass@1 | median decode | completion tokens |
|---|---|---|---|
| stock (`strata-bitlamas-stock-2gpu-32k-int8.json`) | **159/164 (96.95%)** | 45.5 tok/s | 37593 |
| best (`strata-bitlamas-p100-best.json`) | **158/164 (96.34%)** | **72.5 tok/s (+59%)** | 37836 |

133 of 164 answers are byte-identical between the two. One problem differs: HumanEval/32 (`find_zero`, numeric
root finding, its `< 1e-4` check) passes on stock and fails on best; five fail on both (93, 127, 129, 132, 145). A
one-problem difference is what a rounding-level change does to greedy decoding (the same split happens between
Strata's CPU and GPU expert paths); not a measurable coding regression.

## Phase 4: end-to-end validation (interleaved, fresh server per block)

### Long context: 128K context, ABBA (`p100-results/runs/final-long`, `p100-plans/final-long.json`)
A = stock setup config at 131072 tokens (`--kv int8 --kv-resident 32768`); B = the tuned config at 131072
(`--kv k8v4 --kv-resident 32768`). Fresh nonce-prefixed prompts (0 reused tokens), greedy, 256 tokens out.

| case | prompt tokens | decode A (n=2) | decode B (n=2) | prompt A | prompt B |
|---|---|---|---|---|---|
| L64k | 64896 | 32.73 | **51.88 (+59%)** | 536.4 tok/s | 532.1 |
| L120k | 121863 | 36.31 | **49.80 (+37%)** | 514.1 | 503.9 |
| chat | 57 | 24.26 | **42.63 (+76%)** | | |

KV streaming (B): 94.8-96.1% of the long prompts' K/V block reads hit VRAM; the whole K/V of 128K is 0.60 GiB of
pinned RAM (12 QSA layers have K/V; the 36 GDN layers carry a fixed state). Prompt reading is unchanged (-1 to -2%):
nothing kept changes the prompt path. Power: `power.limit` = `enforced.power.limit` = 150.00 W in all 4166 samples;
busy averages 133-136 W per card, the firmware's running-average cap shows as 1-s samples above 150 W 9.1% of the time
and the SW power-cap reason active in 23% of samples (long prompt reads, SM clocks averaging 1066-1126 MHz there):
the cap binds the prompt path at long context, and stays (owner's requirement).

### The decisive 32K run: stock vs tuned vs tuned + `STRATA_SPEC_COST=1` (`p100-results/runs/final-32k-abba`)
Order ABCCBAABCCBA (4 fresh servers per arm), 8 cases, KV int8, 32K; PLE locked in every arm (memlock unlimited, as the
service runs). Decode tok/s, p50 [min-max of 4]; prompt speed of the long cases unchanged (L8k 313, L32k 493 tok/s).

| case | A stock | B tuned | C tuned + SPEC_COST | C > B |
|---|---|---|---|---|
| code | 37.2 [36.9-37.4] | 61.6 [59.3-62.1] | **62.3** [60.6-62.8] | 2/4 |
| edit (4.5K-token file) | 33.6 [33.1-34.6] | 60.5 [56.3-63.0] | **61.8** [57.8-62.9] | 2/4 |
| prose | 28.9 [28.4-29.7] | 43.4 [42.5-45.1] | **46.8** [45.6-47.1] | 4/4 |
| chat | 27.4 [26.5-29.8] | 37.0 [35.6-44.4] | **51.0** [44.9-52.1] | 4/4 |
| tool call | 23.5 [23.2-24.0] | 44.3 [40.6-48.3] | **46.5** [44.6-47.2] | 3/4 |
| think (sampled, thinking on) | 31.8 [31.3-33.3] | 57.0 [54.1-58.7] | 56.6 [54.8-58.7] | 2/4 |
| L8k (15.4K tokens) | 33.3 [32.9-33.5] | 54.5 [52.7-57.4] | **56.4** [54.2-58.2] | 2/4 |
| L32k (32.3K tokens) | 39.0 [35.6-39.7] | 55.6 [55.5-56.4] | 55.5 [53.7-57.6] | 2/4 |
| **geomean vs stock** | 1.00 | **x1.62** | **x1.73** | |

**KEPT `STRATA_SPEC_COST=1`** (the calibrated, cost-aware MTP stop rule upstream ships opt-in; exact verification, so
the sampled distribution is unchanged): never below B beyond noise, +8% prose, +38% chat, +5% tool. It is now in
`p100-configs/strata-bitlamas-p100-best.json`.

### 256K context (`p100-results/runs/long-256k`, one run per arm, the tuned config at 262144 tokens, k8v4)
| arm | prompt | prompt read | decode (256 tokens) |
|---|---|---|---|
| tuned | 254143 tokens | 361.8 tok/s (11.7 min) | 48.07 tok/s |
| tuned + `STRATA_SELECT_LONG=1` | 254143 | 317.7 tok/s (13.3 min) | 50.78 tok/s |

The model's trained context works on this box: the whole K/V of 256K is 1.20 GiB of pinned RAM (32768 cells per QSA
layer in VRAM, 94% of the long prompt's K/V block reads hit VRAM), VRAM as at 32K (15.6 / 15.8 GiB). SELECT_LONG: +5.6%
decode but -12% prompt in its one run, a net loss for a fresh prompt; not kept without an interleaved A/B. The time to
first token is the limit at these lengths (the prompt path, power-capped at 150 W, ~360 tok/s at 254K).

## Summary (2026-11-06)

**Fastest quality-safe configuration**: `p100-configs/strata-bitlamas-p100-best.json` (RUNBOOK.md section 7): the stock
setup config + `STRATA_ARENA_NUMA=1` (new code) + `--pcie-frac 0` + `STRATA_Q8_SM60=1` + `STRATA_DOWN_WS=1` +
`--trim-stage-weights` with `"layer_split": "24"` + `STRATA_SPEC_COST=1`. 128K / 256K variants with k8v4.

| | stock | tuned | |
|---|---|---|---|
| decode, 8-case geomean (32K, int8) | 1.00 | **x1.73** | final ABBA, 4 servers per arm |
| code / prose / chat decode | 37.2 / 28.9 / 27.4 | **62.3 / 46.8 / 51.0 tok/s** | |
| 64K / 120K-token prompt, then decode (128K ctx) | 32.7 / 36.3 | **51.9 / 49.8 tok/s** | k8v4 vs stock int8 |
| 254K-token prompt, then decode | - | 48.1 tok/s | 262144 ctx, k8v4 |
| prompt read 15K / 32K / 120K | 313 / 493 / 514 | 314 / 494 / 504 tok/s | unchanged |
| PPL vs llama.cpp CPU (5 x 1000 tokens) | +0.15% (GPU experts) | +0.3% (GPU experts, Q8_SM60) | KL vs llama.cpp 0.044 -> 0.041 |
| HumanEval pass@1 (164) | 159 | 158 | 133 answers byte-identical |

Quality delta: paired PPL vs stock +0.15% (same K/V), k8v4 +0.1% PPL vs llama.cpp; every KL against stock is at the
engine's own CPU-vs-GPU floor (0.028); the spec's 0.002 KL target is below that floor for any non-bitwise change on
this MoE (Phase 3). NUMA placement and DOWN_WS are byte-identical.

Memory: RAM - expert arena 63.5 GiB pinned (31.9 GiB node 0 / 31.6 GiB node 1, each node holding the row halves its
own CPU slice reads), PLE table 26.8 GiB locked, conversation cache up to 8 GiB, K/V beyond the VRAM window 0.6 GiB at
128K / 1.2 GiB at 256K, the rest file cache (~24 GiB available while serving). VRAM - CUDA0 (layers 0-23): 3713 expert
slots (9.63 GiB) + trimmed dense ~2.8 GiB + Q8_SM60 packed copies ~1.45 GiB; CUDA1 (layers 24-47): 2895 slots (7.45 GiB)
+ dense, packed copies, head (0.64 + 0.6 GiB packed), MTP draft layer 0.84 GiB + draft head 0.28 GiB; 0.6-0.8 GiB free
per card. Power: enforced 150 W on both cards throughout; busy averages 133-136 W.

**Next bottleneck**: the GPUs. Per window (~53 ms at T 3.8): grouped expert kernels ~28% (Q4_K gate/up ALU-bound on
GP100's emulated dp4a: ~260 GB/s of 732), the hyper-connection reads ~18% (Q8 down/up at 80-90 GB/s; TSUM+PFS -10% in
the kernel bench), the two stages run serially (each card idle about half the window; `--pipeline-windows 2` helps
code +12% but only on requests without penalties, and the server's instruct default adds presence_penalty 1.5), MTP
draft ~10%. At long context the prompt path is power-capped (SW power cap active in 23% of samples). Candidates, in
order: a weight-staged Q4_K gate/up kernel for sm_60 (the DOWN_WS pattern, bitwise), penalties support in the
pipelined decode loop, the HC Q8 kernels' occupancy on 56 SMs, `STRATA_PATTN_SM60` for long prompts with int8 K/V.

Not verified: the 150 W cap's persistence across a reboot (the service is enabled and tested live; a reboot needs the
owner), and the GRUB `iommu=pt` / no-crashkernel line (applied live, takes effect at the next boot).

### The owner's NUMA request: "keep RAM->GPU expert transfers inside one NUMA node" - measured, and what was built instead
- The cost being avoided, measured: with the IOMMU in DMA-FQ mode a cross-socket pinned copy to GPU1 ran at 10.29 GB/s
  against 11.70 local (-12%); with the GPU groups in `identity` (iommu=pt) it is 12.40 vs 12.42 GB/s, also with 3 MiB
  blobs (`p100-results/pcie-bw-iommu-*.txt`). The penalty was IOMMU translation, not the socket link, and it is gone.
- Where RAM->GPU expert traffic still happens in the tuned config: the prompt path (it streams the experts the GPUs do
  not hold) and the adaptive tier's ~24 swaps per decode window (`--pcie-frac 0`: no expert is read over PCIe for
  compute). The prompt path with the arena on one node (stock: faulted in by one thread) and with every expert split
  across the two nodes (tuned) reads at the same speed: 313 vs 314 tok/s at 15K, 493 vs 494 at 32K, 514 vs 504 at 120K
  (final ABBAs). So the transfers are socket-neutral on this box after passthrough.
- What NUMA locality did buy, measured: the CPU expert pool, which reads every CPU-side expert from RAM, is 2x faster
  with each row half on its reader's node (28.6 -> 15 ms per window; 1.5x on the all-CPU yardstick). A layer-affine
  layout (each GPU's layers on its own node) would make the transfers local but cap each layer's CPU work at one
  socket's 46 GB/s instead of both sockets' local ~92 GB/s, for a transfer gain measured at 0.2%. Not built.

### Strict KL 0 option and the boot check (added after the summary)
- `p100-configs/strata-bitlamas-p100-exact.json` = the tuned config without `STRATA_Q8_SM60`. Its switches on the
  yardstick (GPU experts, `STRATA_ARENA_NUMA DOWN_WS SPEC_COST`): LOGPOS byte-identical to G-stock (KL 0). Speed
  (`p100-results/runs/exact-vs-best`, BDDB, n=2): code 52.7 vs 63.8, edit 55.2 vs 64.0, prose 40.2 vs 47.1, chat 41.1
  vs 43.8, tool 46.3 vs 49.6, L8k 46.9 vs 54.0 tok/s (-6 to -17%), about x1.5 over stock.
- `p100-boot-check.service` (enabled) records after every boot whether the 150 W cap, `iommu=pt` and the dropped
  crashkernel came back (`p100-results/boot-checks/`). Reboot not yet done.
- Standalone summary: `p100-results/FINAL_SUMMARY.md`.

### Reboot check: PASSED (2026-11-06 18:55 UTC)
Owner-approved reboot. `p100-boot-check.service` 20 s after boot: 150 W power.limit and enforced.power.limit on both
P100s, `iommu=pt` and no crashkernel on the kernel line, both GPU groups `identity`, scratch mounted, sshd active
(`p100-results/boot-checks/boot-20261106T185510Z.txt`). The 150 W cap now holds across reboots without any manual step.

### REJECTED (measured): NUMA-local RAM->GPU transfers, `STRATA_ARENA_NUMA_LAYERS=24` (`p100-results/runs/numa-layers-vs-halves`)
The owner's original request, built as an opt-in mode: whole experts of layers 0-23 on node 0 (GPU0's socket) and of
24-47 on node 1 (GPU1's), so every expert a GPU copies in comes from its own socket. A/B against the row halves (each
expert split by the CPU slice that reads it), BLLB, both arms with the lock and the pre-pin (an earlier attempt ran
without RLIMIT_MEMLOCK in both arms and was discarded: `numa-layers-vs-halves-aborted-memlock`).

| case | row halves (B) | whole layers per GPU socket (L) | prompt B / L |
|---|---|---|---|
| code | 63.4 | 50.7 (-20%) | |
| prose | 46.9 | 40.6 (-13%) | |
| chat | 49.9 | 35.7 (-28%) | |
| L8k | 54.5 | 41.6 (-24%) | 310.3 / 308.8 tok/s |
| L32k | 56.8 | 50.7 (-11%) | 482.7 / 481.0 tok/s |

Socket-local transfers gain nothing even on the prompt path (the one that streams experts to the GPUs), as the copy
bandwidth measurements predicted under `iommu=pt`; the CPU pool then reads half its rows remotely and decode loses
11-28%. The row-half placement stays. The layers mode stays in this branch for the record and is not in the PR.

### `STRATA_ARENA_NUMA=1` alone on the stock config (PR evidence, `p100-results/runs/pr-numa-stock`, ANNAANNA)
| case | stock | stock + `STRATA_ARENA_NUMA=1` | median gain | wins |
|---|---|---|---|---|
| code | 38.42 [36.66-38.66] | 40.25 [40.09-40.86] | +4.8% | 4/4 |
| prose | 30.18 [29.65-30.62] | 31.60 [31.18-31.92] | +4.7% | 4/4 |
| chat | 27.62 [23.48-28.29] | 29.93 [29.62-32.16] | +8.4% | 4/4 |
| L8k | 31.48 [31.30-31.54] | 34.35 [33.58-34.93] | +9.1% | 4/4 |

Alone the placement is worth +5-9% end to end (the stock config's PCIe share keeps the GPUs waiting); with
`--pcie-frac 0` the CPU's halved time is what lets the GPUs run without the PCIe share (OPTLOG Phase 2).

### Owner decisions (2026-11-06 evening)
- Start on boot: **yes** - `strata-bitlamas.service` enabled and started (serves the best config on 127.0.0.1:8080).
  Benchmarks need it stopped first: `sudo systemctl stop strata-bitlamas.service` (two engines do not fit the RAM).
- Upstream PR: **yes** - branch `engine-arena-numa-placement` (worktree `~/projects/strata-pr-numa`, from upstream
  main 54828c7e): only `STRATA_ARENA_NUMA` (row halves + pre-pin), docs bullet, as a draft (HIP / SYCL not built here).

### KEPT `STRATA_Q8_SM60_EXACT=1`: the Pascal Q8_0 GEMV made bitwise (new code, `src/kernels/cuda/q8_sm60.cuh`)
`STRATA_Q8_SM60`'s only difference from the exact kernels is the order of its float additions. The twin keeps its
packed planes, shared-memory activations and one warp per row, and reproduces `native_small_mmvq_kernel`'s arithmetic:
each lane carries the four threads of the exact kernel's row (blocks tid/4 + 32k, ints 2(tid%4)), the same
`d0 * d1 * float(sumi)` per block, warp 0 + 1 + 2 + 3 in order and the same xor butterfly (the exact multi-column
kernels are bitwise the single-column one, so one twin covers 1..8 columns).
- `p100-results/tools/q8_sm60_exact_check.cu` (the engine's own `native_mmvq` on a packed and a GGUF-layout copy of the
  same random matrix): 184 shapes (n_in 512..10240 incl. the small-K path, n_out 512..12288, 1..8 columns), **0 with a
  differing bit**; control: the plain Q8_SM60 kernel differs in 173 of the 184.
- Yardstick with CPU experts (`S-q8exact-fp16`: `STRATA_Q8_SM60=1 STRATA_Q8_SM60_EXACT=1 STRATA_ARENA_NUMA=1`):
  LOGPOS **byte-identical** to S-stock - every dense Q8_0 GEMV and the head bitwise the stock kernels.
- With GPU experts the same switches differ from G-stock at the floor (KL 0.029, PPL 3.986 vs 4.023) for one reason: the
  packed copies take VRAM, the card holds 1937 instead of 3341 expert slots, and the experts that move to the CPU round
  as the CPU does. Any change of VRAM use (the trim, the context length) does the same to the stock engine.
- Speed vs plain Q8_SM60 (`p100-results/runs/q8exact-vs-q8`, BXXB): code 59.8 vs 60.9, prose 45.4 vs 45.5, chat 46.4
  vs 46.1, tool 48.4 vs 47.7, L8k 52.5 vs 54.9 (about -1%).
Now in `strata-bitlamas-p100-best.json` (and the 128K / 256K variants): every kernel of the tuned configuration is
bitwise the stock engine's; the spec's KL target holds for every kernel change, the floor-level KL that remains is the
CPU-vs-GPU placement of experts that any VRAM budget change produces.

### KEPT (2026-11-07): the PLE table half per node, and the arena pinned before the touch (`p100-results/runs/ple-numa-cold`)
After a cold boot the service's start placed only 88% of slice 0 on node 0 (98-100% on warm restarts). `numa_maps`
showed why node 0 was short: all 26.8 GiB of the PLE table (7,016,591 pages) on node 0 (its fault-in starts with one
`MADV_WILLNEED` from the main thread, so the file cache is allocated on that thread's node), node 0 1.9 GiB free while
node 1 held 29 GiB of plain file cache. The mechanism under it: an arena page in a CMA-type pageblock migrates when the
pre-pin takes it, and the migration takes a free page of the other node before it reclaims the file cache of its own
(`/proc/zoneinfo` "cma": node 0 2.5 GiB, node 1 6.2 GiB). Whichever node is fuller loses part of its slice.

Three changes, all inside `STRATA_ARENA_NUMA=1` (the stock path is untouched):
- `src/kernels/ngram.cpp`: the PLE table's first half is faulted in by 8 threads on node 0, the second by 8 on node 1
  (`MPOL_PREFERRED`, `MADV_WILLNEED` per 64 MiB piece from those threads), and pages an earlier reader left in the file
  cache on the wrong node are moved (`move_pages`). `STRATA_PLE_NUMA=0` keeps the old fault-in.
- `src/core/expert_source.cpp`: each chunk is pinned (io_uring fixed buffers) **before** it is touched. The pin's own
  faults allocate outside CMA-type memory under the thread's `MPOL_BIND`, so nothing migrates.
  `STRATA_ARENA_NUMA_PREPIN=after` is the old order.
- The BIND/PREFERRED room estimate leaves out the node's CMA memory and locked pages (a long-term pin can use neither);
  the start line prints it (`room 44.9 / 42.7 GiB`).

The halves alone were not enough: a warm-cache smoke run with the table halved and the old touch-then-pin
(`p100-results/runs/ple-numa-smoke`) put slice 1 at 3559/4096 (node 1 now the fuller one).

ABBA, the tuned 32K config, `sync; echo 3 > /proc/sys/vm/drop_caches` before every start (a cold file cache, as after a
boot; `p100_bench.py` plan key `pre_start`). A = the engine of commit 9eda09a9 (before this change), B = this change.

| block | slice 0 / slice 1 on their nodes (of 4096 sampled pages, after registration and load) | arena placement | PLE lock | load |
|---|---|---|---|---|
| A1 | 4096 / **3346** (81.7%) | 8.9 s | 86.6 s | 252 s |
| B1 | 4096 / 4096 | 1.7 s | 85.8 s | 244 s |
| B2 | 4096 / 4096 | 1.8 s | 86.0 s | 246 s |
| A2 | **4075** / 4096 | 8.6 s | 87.4 s | 254 s |

The service's own warm restart afterwards (the file cache left by A2, the PLE table on one node): 3,498,432 PLE pages
moved in 1.7 s, slices 4096/4096 and 4096/4096.

Decode, medians of 2 (tok/s, A / B): code 55.53 / 59.23, prose 44.98 / 45.32, chat 45.16 / 45.19, tool 44.68 / 47.69,
L8k 53.42 / 50.59. These move with the draft acceptance of each run (e.g. A2 code accepted 326/370, A1 350/404), so the
comparison is the time per decode step (decode_ms / (generated - accepted)), B/A: code 1.015, prose 0.998, chat 0.983,
tool 0.968, L8k 0.938, geomean **0.980** - within the runs' spread (L8k B 44.3 and 50.6 ms/step). **No speedup is
claimed**: in the tuned config the GPUs are the bottleneck; the old engine's 81.7% start (A1) and its 99.5% start (A2)
also decoded within ~2% per step of each other. What the change buys is a deterministic placement (100% / 100% in every
start measured, cold and warm, against 81.7-100% and the boot's 88%) and a 5x shorter placement (1.7 vs 8.7 s).
Output: only where pages are changes (no kernel, no order of arithmetic).

### Upstream PR #5436, revision 2 (2026-11-07): the same change on the PR branch (`p100-results/runs/pr-numa-rev2`)
Commit 0f800ea9 on `engine-arena-numa-placement`: the pin before the touch, the CMA/locked-aware room check and the PLE
table half per node (as on this branch). Validated on the PR's own build, the stock setup config + `STRATA_ARENA_NUMA=1`,
cold file cache before every start, RNNR (R = revision 1, commit 2850a55f): slice pages local after the registration and
the load R1 3963 / 4096, N1 4096 / 4096, N2 4096 / 4096, R2 4096 / 4094 (of 4096); placement 10.8 / 1.7 / 1.7 / 7.6 s;
decode per step N/R geomean 0.998. Yardstick `S-prnuma-rev2-fp16` (PR build, `STRATA_ARENA_NUMA=1`): LOGPOS md5
989d6f4f..., byte-identical to S-stock.
The pin before the touch without the PLE split (`p100-results/runs/pr-numa-pinfirst-nosplit-stopped`, stopped after one
block): the room check found 31.4 GiB on node 0 (the whole PLE table locked there) for a 31.9 GiB slice, chose
`MPOL_PREFERRED`, and 3240/4096 of slice 0 stayed local - why the PR carries the PLE split too.

### Pictures on the CPU (2026-11-07, owner request: "reduce everything above 1440p to 1440p", Q8 projector, 16 threads)
In the spirit of the owner's 27B setup (image cap = 2560x1440's pixel area so small text stays legible; a Q8_0 projector
judged against BF16), with Strata's own CPU picture reader and no engine change: `"vision"` section in
`strata-bitlamas-p100-best-256k-k8v4.json` (`max_tokens` 3600, `threads` 16, ggml-org's Q8_0 mmproj).
- Projector, Q8_0 vs the BF16 reference (`p100-results/tools/vision_compare.py`, `p100-results/vision/compare`): 4K
  test picture 3600 tokens (80x45 grid = 2560x1440), cosine mean 0.9977 (min row 0.620), rel-L2 0.073; 1024x768 chart
  768 tokens, cosine 0.9994, rel-L2 0.040. Encode time is the same (80.0 vs 79.3 s; 7.6 vs 7.7 s: attention-bound);
  peak RSS 1.6 vs 1.3 GiB.
- End to end through the LAN API (`p100-results/vision/e2e-q8.json`): chart 5/5 values; the 4K report 12/12 item lines
  exact (codes and quantities down to the 20 px rows, ~13 px after the downscale) and 15/15 circles by color; 4K
  picture 97.9 s to the answer (encode ~80 s, 3638-token prompt 14.7 s, decode 60.7 tok/s).
- Text: `--vision` adds a 3 MiB image-position table per GPU (6505 instead of 6507 expert slots; free VRAM 789 / 597 MiB as
  before). Yardstick with `--vision` forced (`tools/quant_quality.py` drops it from args; a wrapper exe appends it):
  LOGPOS md5 989d6f4f, byte-identical to S-stock. Text decode spot checks after the restart: 61.6 / 66.8 tok/s (not an A/B).

### A Stop while a picture is read no longer holds every request (2026-11-07, owner report; `p100-results/vision/CANCEL.md`)
Before: a cancelled picture's encode held the engine's single turn to the end. The reader cannot stop in the middle of
a picture, and the server checked for a cancel only between pictures, so every request waited. Text after a cancelled
1440p picture took 73.2 s (s1) and 69.3 s (s3); after a medium picture 13.4-13.7 s (streamed, non-streamed and
`/v1/messages` alike).

Fix (`serve/server.py`, `serve/test_vision_cancel.py`): `Service.fifo` is a `Turn` that counts its waiters. A waiting
request registers its pictures. The encode of a cancelled picture is ended (reader killed, started again in 0.9-1.2 s)
as soon as another request waits and none of the waiting ones sends that picture. Otherwise it is read to the end and
kept for the follow-up.

After: text 1.9 s (s1), 1.6 s (s3), 1.3-1.8 s (s7-s9). A follow-up sent during the cancelled encode uses it: 83.0 /
83.8 s against 83.4 s before, so the picture is not read twice. The cost: a follow-up about a picture whose encode was
ended for another request reads it again (94.1 s in s1, against 15.0 s cached).

Not changed (engine): a cancel during a prompt read takes effect at the end of the current chunk. A 49,889-token prompt
stopped at 8,192 tokens, 24.2 s in, so the next request waited 18 s. Answers at temperature 0 drift from run to run
with or without a cancel (`cancel-residue-check.json`).

### `--prompt-cache-spread` for coding harnesses (2026-11-13; `p100-results/prompt-cache-spread/SPREAD.md`)
Why: the served log showed a harness re-sending a 24,915-token prompt after its session reached 35,542 tokens. The
default cache (the system prompt plus the last few turns) resumed it at 3,416 and read 21,499 tokens again (51.5 s).

A/B on fresh engines with the served config (base, spread, base2), the same 36-turn scripted agent session
(`tools/agent_cache_ab.py`: 8 tools, repo source as tool results, 9.8K to 99.4K tokens). Four jumps back:
- **Turn 16's prompt again, 16K back:** spread read 2,476 tokens (14.1 s) against 10,202 (36.1 / 41.7 s).
- **An old tool result cleared, 20K back:** spread read 28,003 (108.2 s) against 32,983 (123.6 / 131.9 s).
- **10K and 25K back:** the same both ways.
- **The 36 normal turns:** the same reuse in all three runs.
- **The whole session:** 140,651 tokens read against 153,354 / 153,358 (-8.3%). Time 688 s against 704 / 742 s, so
  the time difference is inside the run-to-run spread.

No request read more tokens with spread. Answers right in all three (two planted codes). First-token logprobs differ
base vs base2 as much as base vs spread (3 of 40 first tokens in each pair). Not yet in the served config.

### The 150 W cap was lost after an unattended upgrade (2026-11-13; RUNBOOK section 2)
At 06:19 a libc upgrade restarted `nvidia-persistenced` and `strata-bitlamas` together, the driver reset both cards to
250 W, and the boot-only cap did not run again. Found at 15:33 at the start of the A/B and re-capped at 15:34.

Two drop-ins now apply the cap on every start of either service:
- `strata-bitlamas`: `ExecStartPre=+`, fails closed.
- `nvidia-persistenced`: `ExecStartPost=-+`.

The service restart at 16:18 went through the new `ExecStartPre` and came up at 150 W.

### Sub-agents on the layer split keep the main agent's parked conversation (2026-11-14; `p100-results/split-borrow/BORROW.md`)
Upstream #5612: on a layer split a sub-agent that shares only the system prompt took the main agent's whole parked
conversation and dropped its later checkpoints. Opt-in `STRATA_SPLIT_BORROW=1` (9ca96758 here, b232e857 on upstream
main in `~/projects/strata-split-borrow`) copies each stage's K/V prefix and the checkpoints instead.

Main agent of 12 turns, an unrelated chat, two sub-agents, fresh engines, both cards at 150 W in every sample:
- Main agent's return after sub-agent 1: 18,684 tokens read again (50-54 s) -> 2,755 (13.8 s).
- After sub-agent 2: 33,317 (105-110 s) -> 2,672 (13.3 s). All 22 requests: 100,555 read, 385-399 s -> 53,981, 252 s.
- With `--adapt-every 0`: the sub-agents' requests identical with and without; the two returns differ by at most
  0.02 logprob (a re-read against the state as first read); two runs without the switch differ by up to 0.59.

Served since 22:01 UTC together with `--prompt-cache-spread`; live check: 2 borrows, 2,755 / 2,672 tokens read.
The first patch failed on its first borrow (a guard in the prefix restore) and stopped that test engine; fixed.
Service down 18:15-18:53, 18:56-19:06 and 21:31-22:04 UTC (the last includes the deploy restart).

### Cache limits for several agents: 16 parked conversations, 12 checkpoints, 16 GiB (2026-11-15; `p100-results/cache-limits/LIMITS.md`)
Owner: the box will serve several agents in turn (a chat bot, coding harnesses). `--prompt-cache 12
--conversation-cache-slots 16 --conversation-cache-mib 16384` against the defaults (6, 4, 8192 MiB), fresh engines,
both cards at 150 W in every sample:
- Eight agents with different system prompts in turn: 196,234 tokens read, 792 s -> 105,080, 454 s. A returning agent
  read 12-13 K tokens (50-52 s) before and its new turn (2.3-2.9 K, 13-15 s) after; evictions 19 -> 0.
- One coding agent, 36 turns: 140,599 read, 645 s -> 130,549, 648 s (rewrites far back resume nearer the edit; an
  ordinary turn parks 0.2-0.8 s longer).
- Cache at most 14.5 GiB, lowest `MemAvailable` 11.2 GiB, no parking skipped. Not an exactness check (adaptation on).
Served since 07:08 UTC. Service down 00:14-01:02 UTC for the A/B.

### The experimental speed projection, loaded and off by default (2026-11-15; `p100-results/projection/PROJECTION.md`)
Owner request. The vector in `data/experimental-speed-projection/` (a refusal-direction projection by its package's
description) is loaded with the four engine arguments; `"sampling": {"experimental_speed_projection": false}` keeps it
off unless a request sends `"experimental_speed_projection": true`.
- Loaded and off against an engine without it, `--adapt-every 0`: logprob difference 0.000000 on five requests.
- Agents with it on and off in turn each reuse their own parked conversation (2.4-2.8 K tokens read per return).
- 400-token reply: 42.6 / 43.0 tok/s off, 43.2 / 42.4 on. Refusal rate and quality with it on: not measured.
Served since 07:18 UTC. Service down 06:53-07:11 (tests) and 07:15-07:18 UTC (deploy).

### Upstream pull requests (2026-11-15, owner request)
- #6009 `[Docs]` the projection's switch and parked conversations (branch `docs-projection-parked`).
- #6010 `[Docs]` more clients than cache slots, the eight-agent measurement (branch `docs-cache-slots-agents`).
- #6011 `[Engine]` draft: the Stop during a prompt read (395fcb23 on main 13eedf55, worktree
  `~/projects/strata-stop-pr`; CUDA build passed, measurements are the 2026-11-07 ones, HIP / SYCL not built).
