# bitlamas Qwen3.8-Flash-Next Q4_K_XL-IQ4NL on 2x Tesla P100 with Strata: final summary (2026-11-06)

Hardware: 2x Tesla P100-PCIE-16GB (GPU0 PCIe 3.0 x8, GPU1 x16), both capped at 150 W; 2x Xeon E5-2696 v3, 125.7 GiB
DDR3 (one P100 per socket). Strata engine 0.1.57, branch `p100-bitlamas-iq4nl`. Details of every attempt: `OPTLOG.md`;
how to run, the services and rollback: `RUNBOOK.md`.

## Fastest quality-safe configuration

`p100-configs/strata-bitlamas-p100-best.json`: the stock setup config plus `STRATA_ARENA_NUMA=1` (new engine code on
this branch), `--pcie-frac 0`, `STRATA_Q8_SM60=1` with `STRATA_Q8_SM60_EXACT=1` (new code: bitwise), `STRATA_DOWN_WS=1`, `--trim-stage-weights` with an explicit
24/24 layer split, `STRATA_SPEC_COST=1`. Long context: the `-128k-k8v4` / `-256k-k8v4` variants.

Decode, 32K context, KV int8. Interleaved ABCCBAABCCBA, a fresh server for every block, 4 blocks per arm; p50 / p95
over the 4 (with 4 samples p95 is near the maximum):

| case | stock p50 / p95 | tuned p50 / p95 | speedup (p50) |
|---|---|---|---|
| code | 37.2 / 37.4 | 62.3 / 62.7 | x1.67 |
| edit (4.5K-token file) | 33.6 / 34.5 | 61.8 / 62.8 | x1.84 |
| prose | 28.9 / 29.6 | 46.8 / 47.1 | x1.62 |
| chat | 27.4 / 29.6 | 51.0 / 51.9 | x1.86 |
| tool call | 23.5 / 23.9 | 46.5 / 47.2 | x1.98 |
| thinking (sampled) | 31.8 / 33.1 | 56.6 / 58.6 | x1.78 |
| 15K-token prompt | 33.3 / 33.5 | 56.4 / 57.9 | x1.70 |
| 32K-token prompt | 39.0 / 39.6 | 55.5 / 57.5 | x1.42 |
| **geomean** | | | **x1.73** |

HumanEval (164 requests per config, decode per request): stock p05 / p50 / p95 40.0 / 45.5 / 48.1 tok/s, tuned
65.9 / 72.5 / 77.0 tok/s.

Long context (ABBA, 128K context; stock int8 vs tuned k8v4; two runs per arm, both shown: 64K 32.89, 32.57 -> 52.21,
51.55; 120K 36.56, 36.05 -> 49.29, 50.31 tok/s): 64K-token prompt then decode 32.7 -> 51.9 tok/s, 120K
prompt 36.3 -> 49.8. At 262144 context the tuned config read a 254K-token prompt at 362 tok/s and decoded at 48.1.
Prompt reading is unchanged everywhere (313 / 494 / 504 tok/s at 15K / 32K / 120K).

## Quality delta

| check | stock | tuned |
|---|---|---|
| PPL vs llama.cpp CPU on the same 5 x 1000 tokens (llama.cpp 4.017) | 4.023 (+0.15%) | 4.029 (+0.3%) |
| KL vs llama.cpp | 0.0445 | 0.0406 (nearer) |
| HumanEval pass@1, 164 problems | 159 | 158 (133 answers byte-identical; one problem split) |
| non-finite logits | none (the yardstick stops on one; every run finished) | none |
| GPU kernels for every quant type here (Q4_K, Q5_K, IQ4_NL) | parity test passed on both cards | same; per-window counts show the GPUs computing their cached experts, 0 over PCIe |

KL against stock: **every kernel of the tuned configuration is now bitwise the stock engine's.** Its one former
exception, `STRATA_Q8_SM60` (float addition order only), runs as a bitwise twin with `STRATA_Q8_SM60_EXACT=1`
(new code; 184 shapes with no differing bit, about -1% speed against the plain kernel): on the CPU-experts yardstick
the tuned switches give log-probabilities byte-identical to stock (KL 0). What remains with GPU experts is the
engine's own floor (KL ~0.028): a different VRAM budget changes which experts the GPU computes, and the CPU rounds
differently - the same happens to the stock engine between two context lengths. `strata-bitlamas-p100-exact.json`
(no Q8_SM60) keeps the stock VRAM layout and is byte-identical with GPU experts too, at about x1.5 over stock. K/V formats
(PPL vs llama.cpp): int8 -0.1%, k8v4 +0.1%, q4_0 +0.0%.

## Memory

| | node 0 / CUDA0 (layers 0-23) | node 1 / CUDA1 (layers 24-47) |
|---|---|---|
| RAM: expert arena (pinned) | 31.9 GiB (the row halves node 0's CPUs read) | 31.6 GiB |
| RAM: PLE table 26.8 GiB (locked), conversation cache <= 8 GiB, K/V beyond the VRAM window 0.6 GiB at 128K / 1.2 GiB at 256K, file cache for the rest (~24 GiB available while serving) | | |
| VRAM: expert cache | 3713 slots, 9.63 GiB | 2895 slots, 7.45 GiB |
| VRAM: dense weights (own layers), Q8_SM60 packed copies | ~2.8 + ~1.45 GiB | ~2.8 + ~1.45 GiB, head 0.64 + 0.6 GiB packed |
| VRAM: MTP draft layer + its head | - | 0.84 + 0.28 GiB |
| VRAM free after load | 0.79 GiB | 0.60 GiB |

Power: power.limit = enforced.power.limit = 150 W on both cards in every sample; busy averages 133-136 W.

## Options and open items

- KL: done - `STRATA_Q8_SM60_EXACT=1` makes the Pascal Q8_0 GEMV bitwise (see Quality delta).
- NUMA-local RAM->GPU transfers (whole layers on each GPU's socket, `STRATA_ARENA_NUMA_LAYERS=24`): built and measured,
  11-28% slower decode, prompt reading unchanged (OPTLOG); not used.
- Upstream (from the owner's account, commits under the GitHub no-reply address):
  - https://github.com/Niko1221/Strata/pull/5436 (draft): `STRATA_ARENA_NUMA` arena placement with the long-term pre-pin;
    draft because HIP / SYCL could not be built here. Linked in issue #3708 (the arena-path limitation), where
    https://github.com/Niko1221/Strata/issues/3708#issuecomment-6697893811 adds this box's evidence (the CMA pin migration).
  - https://github.com/Niko1221/Strata/pull/5438: `STRATA_Q8_SM60_EXACT` (CUDA-only file; 195 shapes bitwise, the yardstick
    byte-identical).
- Start on boot: `strata-bitlamas.service` enabled and running (owner's choice).

## What changed in Strata (code)

`STRATA_ARENA_NUMA=1` (`src/core/expert_source.cpp`, `src/core/pinned.cu`; opt-in, default path unchanged): the expert
arena's row halves are placed on the NUMA node of the CPU pool slice that reads them, before the GPU registration, and
pinned long-term from the touching threads so that `cudaHostRegister` does not migrate them (it moved 15-27% of node 1's
half to node 0: node 1 has 14 GiB of CMA-type pageblocks). CPU pool time per decode window 28.6 -> 15 ms, output
byte-identical. Also `tools/quant_quality.py run --kv` (opt-in) and the parity pair list in `CMakeLists.txt`.

## Next bottleneck

The GPUs: per ~53 ms decode window the grouped expert kernels take ~28% (Q4_K gate/up ALU-bound on GP100's emulated
dp4a), the hyper-connection reads ~18%, the MTP draft ~10%, and the two stages run one after the other. At long context
the prompt path is power-capped (the SW power cap was active in 23% of samples). Next: a bitwise weight-staged Q4_K
gate/up kernel for sm_60, penalty support in `--pipeline-windows 2` (+12% on code where it can run), HC kernel occupancy.

## Verified after the reboot (2026-11-06 18:55 UTC)

`p100-results/boot-checks/boot-20261106T185510Z.txt`, written by `p100-boot-check.service` 20 s after boot:
power.limit = enforced.power.limit = 150 W on both P100s (PASS), `iommu=pt` on the kernel line (PASS), no
crashkernel (PASS), both GPU IOMMU groups `identity`, `/mnt/scratch` mounted, sshd active. MemTotal 125.7 GiB.
GPU0 stays at PCIe x8 for this effort (the owner's planned slot swap is in RUNBOOK.md section 1).
