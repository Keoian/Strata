# Qwen3.8-Flash-Next on a dual-Xeon, dual-P100 box with Strata

What was done to run a ~95 GB mixture-of-experts model fast on 2016-era server hardware, what each step bought, and
what went upstream. Every number below was measured on this box; the raw logs are in `p100-results/`, the step-by-step
record in `OPTLOG.md`, and the operating guide in `RUNBOOK.md`.

## TL;DR

- **1.73x faster decode than stock Strata** (8-case geometric mean, interleaved A/B with fresh servers): code
  37.2 -> 62.3 tok/s, chat 27.4 -> 51.0, tool calls 23.5 -> 46.5, a 15K-token prompt 33.3 -> 56.4.
- **Same quality:** HumanEval 158 vs 159 of 164, perplexity within 0.3% of llama.cpp's CPU reference, and every GPU
  kernel in the served config produces bit-for-bit the stock engine's results. The one deliberate approximation is
  the 256K config's 8-bit-key / 4-bit-value cache (+0.1-0.2% perplexity).
- **256K context** (the model's full trained length) with a 254K-token prompt decoding at 48 tok/s, **pictures**
  (CPU encoder, 1440p cap), served on the LAN with an API key, started at boot.
- **NUMA was the big lesson:** the expert memory was on the wrong socket, and a kernel memory quirk (CMA-type
  pageblocks) silently moved up to a third of it back to the wrong socket during GPU pinning. Fixed in engine code;
  the CPU's expert work per decode step went from 28.6 to 14 ms.
- **Two upstream pull requests** (NUMA placement, a bit-exact fast Pascal kernel) and one diagnostic report.
- Both GPUs held at a **150 W** cap the whole time.

## The box

| Part | Details |
|---|---|
| CPUs | 2x Xeon E5-2696 v3 (Haswell, 18 cores / 36 threads each, AVX2), 2 NUMA nodes |
| RAM | 8x 16 GB DDR3-1600 RDIMM, one per channel, 125.7 GiB usable |
| GPUs | 2x Tesla P100 PCIe 16 GB (GP100, sm_60), one per socket; GPU0 runs at PCIe 3.0 x8 (slot wiring), GPU1 at x16; both capped at 150 W |
| Storage | 500 GB SATA SSD (OS, model, results); 1 TB HDD for downloads and cold storage only |
| Software | Ubuntu 26.04, kernel 7.0, NVIDIA driver 580.178.04, CUDA 12.8.1, gcc-14; Strata engine 0.1.57 |

## The model, and why it is hard for this box

Qwen3.8-Flash-Next, bitlamas Q4_K_XL-IQ4NL quant (Q4_K / Q5_K gate-up, IQ4_NL down projections). It is a mixture of
experts: 48 layers, 512 experts per layer, 10 used per token.

| Part | Size | Used per token |
|---|---|---|
| Routed experts (24,576 of them) | 63.5 GiB | 480 of them, about 2% |
| PLE table (per-layer token embeddings) | 26.8 GiB | a few small rows |
| Dense weights (attention / GDN, shared expert, router, norms) | ~4 GiB | all of them |
| Output head and token embedding | 1.3 GiB | all / one row |

About 95 GB never fits in 32 GB of VRAM. Unlike llama.cpp with a dense model (where every weight is read every token,
so everything must sit on the GPU), Strata puts what every token needs on the GPUs (dense weights split by layer: 0-23 on
GPU0, 24-47 on GPU1), keeps the experts in pinned RAM, uses the leftover VRAM as a cache of the most-used experts, and
computes the misses on the CPUs straight from RAM. That makes three things matter at once: GPU kernel speed, CPU memory
bandwidth (NUMA), and how the VRAM is spent.

## How it was measured

- **Interleaved A/B with a fresh server per block** (ABBA, ABCCBA...), medians over 2-4 servers per arm, never a single
  run. Single-run "screens" were used only to choose what to A/B.
- The engine's own timers (decode tok/s = generated tokens / decode time), 8 cases: code, edit (a 4.5K-token file),
  prose, chat, tool call, thinking (sampled), 15K-token and 32K-token prompts; plus 64K / 120K / 254K-token prompts.
- Power, clocks, throttle reasons and temperatures logged every second for every block.
- Quality: Strata's `quant_quality.py` yardstick (5 x 1000 tokens, teacher-forced log-probabilities), a llama.cpp CPU
  run of the same tokens as the reference, and HumanEval (164 problems) through the server.
- Bit-identity checks wherever a change claimed not to change results (md5 of the log-probability files).

## Step 1: system setup

| Change | Why | Benefit (measured) |
|---|---|---|
| Driver 580 + CUDA 12.8.1 (+ a one-line header patch for glibc 2.43), gcc-14 | CUDA 12.9 conflicts with this glibc per Strata's notes | sm_60 engine builds cleanly |
| **150 W power cap on both cards at every boot** (`p100-power-cap.service`) | the owner's requirement | enforced 150 W in every sample of every run; busy averages 133-136 W |
| **IOMMU passthrough** (`iommu=pt`) | with the IOMMU translating, pinned copies across sockets lost 12% | GPU1 host-to-device 10.29 -> 12.40 GB/s (+20%) from the far socket, 11.70 -> 12.42 (+6%) from its own; copies became socket-neutral |
| Released the kdump memory reservation (kdump was off anyway) | 4.25 GiB sat unused on node 1 | usable RAM 121.5 -> 125.7 GiB |
| Boot check (`p100-boot-check.service`) | prove the cap and kernel settings survive a reboot | verified after two reboots: 150 W, `iommu=pt`, no crashkernel, SSH up |
| Scan of every built engine for a known sm_60 register-aliasing bug (upstream #4274) | a fault there would silently corrupt dot products | 0 of 1,011,704 affected instructions; every build is gated on the scan |

## Step 2: bring-up and baseline

- Downloaded the four model shards and checked them against the repo's SHA-256 list; built Strata's pack (46 s).
- Parity tests on both GPUs for this quant's expert formats (Q4_K/IQ4_NL, Q5_K/IQ4_NL) on synthetic and real layers:
  all passed; GPU dequantization bit-exact with ggml's. Strata ran this quant without new kernels.
- Stock smoke run: coherent answers, tool calls, 25-38 tok/s.
- **Baseline profile:** 88.7 ms per decode window (3.39 tokens per window, 38.2 tok/s on code). The CPU expert pool
  took 28.6 ms per window at ~34 GB/s, below one socket's 45.9 GB/s, because the expert memory sat mostly on one socket.
  The engine said so itself: "memory is not placed per node (known limitation)".

## Step 3: tuning, one step at a time

Speeds in this section are single-run screens on code / prose prompts unless marked; the final interleaved A/B is in
Step 4.

| # | Change | What it does | Benefit |
|---|---|---|---|
| 1 | **`STRATA_ARENA_NUMA=1`** (new engine code) | places each expert's two row halves on the socket of the CPU threads that read them, before the GPU registration (4 KiB pages, node-bound threads) | CPU expert time per window **28.6 -> 14.0 ms**; on the stock config alone **+4.7 to +9.1% decode** (ANNAANNA, 4/4 wins in every case); all-CPU yardstick 39 -> 26 s per 1000 tokens |
| 2 | **`--pcie-frac 0`** | stop copying missed experts over PCIe; the (now fast) CPU computes them instead | 82.6 -> 65.5 ms per window; **code 41.1 -> 51.4, prose 31.0 -> 37.5 tok/s** |
| 3 | **`STRATA_Q8_SM60=1`** (upstream, opt-in) | a Pascal-specific Q8_0 matrix-vector kernel for the dense projections and the head | **+16% code, +11% prose**; costs VRAM for packed copies (~1.45 GiB per card) |
| 4 | **`STRATA_DOWN_WS=1`** (upstream, opt-in) | weight-staged expert down projection | -13.7% time per layer's experts (microbenchmark), +2% / +5% end to end; bit-for-bit identical |
| 5 | **`--trim-stage-weights` + an explicit 24/24 layer split** | each card loads only its own layers' dense weights | +1,700 expert slots over the automatic split; **code 61.5 -> 64.0** |
| 6 | **Long-term pre-pin** (new code, part of 1) | stops the GPU registration from migrating placed pages to the wrong socket (see "The NUMA story") | node 1's half back from 68-79% to 95%+ local; CPU time 18.2 -> 15.0 ms per window; restored 63.9 / 44.4 tok/s |
| 7 | **`STRATA_SPEC_COST=1`** (upstream, opt-in) | cost-aware MTP stop rule: each step picks how many draft tokens to verify from measured acceptance and measured per-width cost | 8-case geomean **x1.62 -> x1.73** over stock; chat +38%, prose +8%, tool +5% (interleaved, 4 servers per arm); sampling stays exact |
| 8 | **`STRATA_Q8_SM60_EXACT=1`** (new engine code) | the fast Pascal Q8_0 kernel rewritten to add its numbers in exactly the stock kernel's order | **bit-for-bit identical** to the stock kernels (195 shapes, 0 differing bits; the plain kernel differs in 180); about 1% slower than the plain fast kernel |
| 9 | **PLE table half per socket + pin before first touch** (new code, part of 1) | read the 26.8 GiB PLE table half on each socket; pin each expert chunk before writing it, so pages are allocated on the right socket and never migrate | placement **100% / 100% on every start** (cold boot and warm restart) vs 81.7-99.5% before; placement time 8.7 -> 1.7 s; decode unchanged (the GPUs are the bottleneck) |
| 10 | **256K context, 8-bit keys / 4-bit values, first 32K positions in VRAM** | the model's full context with the same VRAM footprint as 32K; the rest of the cache in 1.2 GiB of pinned RAM | 254K-token prompt then 48.1 tok/s decode; perplexity cost +0.1-0.2% |

The served configuration (`p100-configs/strata-bitlamas-p100-best-256k-k8v4.json`) carries all of the above.

## Step 4: results

### Decode, 32K context, interleaved (4 fresh servers per arm)

| Case | Stock | Tuned | Speedup |
|---|---|---|---|
| code | 37.2 | 62.3 | x1.67 |
| edit (4.5K-token file) | 33.6 | 61.8 | x1.84 |
| prose | 28.9 | 46.8 | x1.62 |
| chat | 27.4 | 51.0 | x1.86 |
| tool call | 23.5 | 46.5 | x1.98 |
| thinking (sampled) | 31.8 | 56.6 | x1.78 |
| 15K-token prompt | 33.3 | 56.4 | x1.70 |
| 32K-token prompt | 39.0 | 55.5 | x1.42 |
| **geometric mean** | | | **x1.73** |

Prompt reading did not change (stock 313 / 493 / 514, tuned 314 / 494 / 504 tok/s at 15K / 32K / 120K): nothing kept
touches the prompt path, which at long context is limited by the 150 W cap. This run used the plain fast Q8 kernel; the bit-exact version that is served
now measured about 1% slower.

### Coding (HumanEval, 164 problems, greedy, through the server)

| | Stock | Tuned |
|---|---|---|
| pass@1 | 159 | 158 |
| decode p05 / p50 / p95 | 40.0 / 45.5 / 48.1 tok/s | **65.9 / 72.5 / 77.0 tok/s (+59% median)** |

Run on the 32K tuned config with the plain fast Q8 kernel. 133 answers byte-identical; one numeric root-finding problem flipped, the size of difference a rounding-level change
makes to greedy decoding (Strata's own CPU and GPU expert paths differ the same way).

### Long context

| Prompt | Stock (int8 KV) | Tuned (8/4-bit KV) |
|---|---|---|
| 64K tokens, then decode | 32.7 tok/s | **51.9 tok/s (+59%)** |
| 120K tokens, then decode | 36.3 tok/s | **49.8 tok/s (+37%)** |
| 254K tokens (256K context) | - | read at 362 tok/s (11.7 min), decode 48.1 tok/s |

94-96% of the long prompts' cache reads hit VRAM even though only 32K positions per layer live there.

## Step 5: quality validation

| Check | Result |
|---|---|
| Perplexity vs llama.cpp CPU on the same 5,000 tokens (llama.cpp 4.017) | stock 4.023 (+0.15%), tuned 4.029 (+0.3%) |
| KL divergence vs llama.cpp | stock 0.0445, tuned 0.0406 (closer) |
| Engine's own noise floor | the same expert weights on CPU vs GPU differ by KL 0.028 on this MoE (10-of-512 routing flips near-ties), so any non-bit-exact change lands at or above that |
| Bit-identity of the new code | NUMA placement, PLE placement, pin-first, DOWN_WS and the Q8 exact kernel: log-probability files byte-identical to stock (md5 `989d6f4f...`) |
| KV formats (perplexity vs llama.cpp) | int8 -0.1%, 8/4-bit +0.1%, q4_0 +0.0%; all inside the +1% gate |
| HumanEval | 158 vs 159 (above) |
| Non-finite logits | none in any run |

A strict option (`strata-bitlamas-p100-exact.json`, without the fast Q8 kernel's VRAM layout) is byte-identical to stock
even with GPU experts, at about x1.5 instead of x1.73.

## The NUMA story

1. **Problem:** Strata already splits the CPU expert work per socket, but with this model the experts come from a pinned
   RAM "arena" that one thread faulted in, so the memory sat on one socket and the other socket's threads read every
   row across the QPI link.
2. **First fix:** place each row half on its reader's socket before the GPU registration: CPU time per window halved
   (28.6 -> 14 ms).
3. **The drift:** over a day, the CPU time crept back to 18 ms. Sampling showed 21-32% of socket 1's half had moved to
   socket 0 *after* the GPU registration. Cause, reproduced outside the engine: socket 1 has ~14 GiB of CMA-type
   pageblocks (although the system reserves no CMA area). A page there cannot stay under a long-term GPU pin, so the
   kernel migrates it, and when the page's own socket is full the migration takes the other socket's memory instead
   of freeing file cache. A 30-line test at 63.5 GiB migrated 1.14 million pages.
4. **Second fix:** pin each chunk from its socket-bound thread right after touching it, while the socket still has room:
   95-100% local again.
5. **Third fix (2026-11-07):** after a cold boot, the 26.8 GiB PLE table had been read onto one socket, leaving it too
   little room, and the placement dropped to 88%. Now the table is read half per socket, and each expert chunk is
   pinned *before* it is written, so the pin itself allocates the page on the right socket outside CMA memory and
   nothing ever migrates: 100% / 100% on every start, and placement takes 1.7 s instead of 8.7 s.
6. **What did not help:** keeping each GPU's experts on its own socket (so every RAM-to-GPU copy is local). With
   `iommu=pt` copies are equally fast from either socket, and that layout made the CPU read half its rows remotely:
   decode -11 to -28%. Measured, rejected.

## Tried and not kept (all measured)

| Change | Result | Why not kept |
|---|---|---|
| Whole layers per GPU socket (`STRATA_ARENA_NUMA_LAYERS`) | decode -11 to -28%, prompt unchanged | CPU reads go remote; copies were already socket-neutral |
| Expert profile trained on our own prompts | 61.4 / 43.7 vs 61.0 / 43.1 | more VRAM hits put more work on the GPUs, which are the bottleneck |
| `--pipeline-windows 2` | code +11.5%, prose -7% | only runs on requests without presence penalty; the server's instruct default sets 1.5 |
| `STRATA_MTP_STAGE0=1` | worse than the drafter on the last card | |
| `STRATA_SELECT_LONG=1` at 256K | decode +5.6%, prompt -12% | a net loss for a fresh long prompt |
| `--spec 3` / `--spec 5` | 61.7 / 46.5 and 61.5 / 42.3 | the default 4 with the stop rule is better |
| `STRATA_GU_SWIGLU`, `STRATA_QSA_ROWS`, `STRATA_ATTN_DAILY`, `STRATA_TSUM` + `GR_Q8_PFS`, `STRATA_GR_V3` | within noise | no measurable gain |
| More aggressive expert adaptation, `kernel.numa_balancing=0`, PLE lock for speed, 8/4-bit KV at 32K | neutral | the CPU is off the critical path |
| Automatic layer split | picked 19, fewer slots | the explicit 24/24 split holds 1.7K more experts |
| Huge pages for the expert arena | shelved | the ceiling was ~10% of the CPU side, which is no longer critical |

## Step 6: serving

- `strata-bitlamas.service`: starts at boot after the power cap, with the 256K config, `LimitMEMLOCK=infinity` (needed
  for the pins and the PLE lock), restart on failure.
- On the LAN at `http://<box>:8080/` (web chat page) and `/v1` (OpenAI- and Anthropic-compatible API), with an API key
  kept in a mode-600 file.
- VRAM use per card: dense weights ~2 GiB, Pascal kernel copies 1.44 GiB, hyper-connection copies 0.63 GiB, cache and
  buffers ~1.5 GiB; GPU1 also the head (1.26 GiB) and the MTP draft model (~1.1 GiB); the rest is the expert cache:
  **6,505 experts (16.8 GB)**. In the 32K config (6,648 experts) the cache served 62-78% of the expert work on code,
  prose, chat and long prompts.
- RAM: experts 63.5 GiB pinned (31.9 / 31.6 GiB per socket), PLE table 26.8 GiB locked (half per socket), 8 GiB
  conversation cache, 1.2 GiB of long-context cache, ~23 GiB left.

## Step 7: pictures

- Strata's CPU picture reader with the Q8_0 encoder (ggml-org) and a cap of 3,600 image tokens (2560x1440's pixel area:
  larger pictures are scaled down, aspect kept), 16 threads. The idea came from the owner's 27B llama.cpp setup and was
  carried over with Strata's own settings.
- No VRAM used: 2 expert slots lost to a 3 MiB image-position table per GPU; text output byte-identical with it on.
- Q8_0 vs the BF16 reference encoder: cosine 0.998 / 0.999 on a 4K and a chart test picture.
- End to end: a 4K test sheet read exactly (12/12 lines, codes down to ~13 px after the downscale, 15/15 shapes counted),
  a chart read exactly. Cost: a 1440p picture takes ~80 s to encode on these CPUs (98 s to the answer); a 1024x768
  picture ~8 s (17 s to the answer). The encoder holds 1.3 GB of RAM and no CPU while idle.

## Upstream contributions

| Contribution | What it is | Status | What everyone gets if accepted |
|---|---|---|---|
| [PR #5436](https://github.com/Niko1221/Strata/pull/5436) | opt-in `STRATA_ARENA_NUMA`: per-socket placement of the expert arena on two-socket Linux, pin-before-touch (no CMA migration), the PLE table half per socket, a CMA-aware room check, placement reported in the log | draft (HIP / SYCL builds could not be run here) | two-socket machines on the arena path stop reading half their experts across sockets: CPU expert time halved here, **+5-9% decode on the stock config**, deterministic placement in ~2 s, output byte-identical |
| [PR #5438](https://github.com/Niko1221/Strata/pull/5438) | opt-in `STRATA_Q8_SM60_EXACT`: the Pascal Q8_0 kernel in the stock kernels' exact addition order | ready for review | P100 / GP100 owners can use the fast Q8 kernel (+16% code here) **without changing the model's output**, so it can be judged with a simple "same answers" check |
| [Issue #3708 comment](https://github.com/Niko1221/Strata/issues/3708#issuecomment-6697893811) | a fourth two-socket report: the CMA pin-migration mechanism, the numbers, the measured case against socket-local GPU copies | posted | explains why the same placement measures differently from box to box, and how to check it |

Possible next contributions, not started: proposing the MTP stop rule (`STRATA_SPEC_COST`) as a P100 default (needs a
stock-config A/B first), and combining it with an offline-fit acceptance table (the approach of the owner's 27B
llama.cpp branch), since everyone running this model uses the same MTP head.

## Where the time goes now, and what is next

The GPUs are the bottleneck. Per ~53 ms decode window: grouped expert kernels ~28% (Q4_K gate/up is ALU-bound on
GP100's emulated dp4a, ~260 of 732 GB/s), hyper-connection reads ~18%, the MTP draft ~10%, and the two cards' stages run
one after the other. Long prompts are limited by the 150 W cap. Candidates: a bit-exact weight-staged Q4_K gate/up
kernel for sm_60, penalty support in the pipelined decode loop, the hyper-connection kernels' occupancy, and moving
GPU0 to an x16 slot once its new cooler is printed.
