**Problem and result.** `STRATA_Q8_SM60=1` (#1424) runs the Q8_0 dense projections and the head on a GP100 kernel that
reads them much faster than the exact kernels, but it adds the per-block products in another order (`q8_sm60.cuh`: "not
bitwise the exact kernels"), so turning it on changes the model's output. On a Mixture-of-Experts model that is more than a
rounding detail in practice: a changed bit can flip a near-tied route of the router, so the switch cannot be judged by a
"same answers" check and costs a quality gate where one is needed. This PR adds `STRATA_Q8_SM60_EXACT=1` (opt-in, used with
`STRATA_Q8_SM60=1`): the same kernel structure in the exact kernels' addition order. Every output is then bitwise the exact
kernels'; the speed is about 1% below the plain kernel. Nothing changes unless the switch is set.

**Related discussion.** #1424 (the kernel), #3751 (the MTP drafter on it), #3867 (the bitwise `STRATA_DOWN_WS`, the same
"bitwise, opt-in" pattern for GP100).

**Approach.** The exact Q8_0 path on sm_60 is `native_small_mmvq_kernel<Q80Block, 8>` (its multi-column kernels are bitwise
the single-column one, per the comment above `MAX_NCOLS`). A row there is 4 warps x 32 threads; thread `tid` takes blocks
`tid / 4 + 32 k` at ints `2 (tid % 4)` and `2 (tid % 4) + 1`, accumulates `d0 * d1 * float(sumi)` over k, then warp 0 adds
warps 1, 2 and 3 in that order and runs `warp_sum` (xor 16 .. 1). The new `q8sm60::kernel_exact` keeps the packed planes,
the activations staged once per workgroup in shared memory, and one warp per row with no barrier per row; each lane carries
the four threads `L, L + 32, L + 64, L + 96` of the exact kernel's row as four accumulators per column, adds them in the
same order and runs the same butterfly. The small-K rule (below 1024 inputs) changes only how many rows a workgroup takes,
not a row's arithmetic, so the same twin covers it. `launch_one` picks it when `STRATA_Q8_SM60_EXACT=1` and logs it once.
The header is compiled only by CUDA builds (`native_mmvq.cu` includes it under `!__HIPCC__`).

**Validation** (revision: this branch on main 54828c7e; Ubuntu 26.04, Linux 7.0.0-43, CUDA 12.8.93, gcc-14,
`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60`; Tesla P100 PCIe 16 GB, driver 580.178.04)

Passed:
- CUDA build of `strata` and of the new `tools/q8_sm60_exact_check.cu` against this branch's `libstrata_kernels.a`.
- `STRATA_Q8_SM60=1 STRATA_Q8_SM60_EXACT=1 ./q8_sm60_exact_check`: 195 cases (n_in 512, 640, 1024, 2048, 2560, 4096, 6144,
  10240 x n_out 512, 2560, 12288 x 1..8 columns, and the head 2560 -> 248320 at 1, 4, 8 columns), **0 with a differing
  bit**. Control, the plain kernel (`STRATA_Q8_SM60=1` only): 180 of the 195 differ.
- End to end, this branch's `strata`: `tools/quant_quality.py run` (5 x 1000 tokens of `data/quant_quality/v1`, CPU experts,
  FP16 K/V; Qwen3.8-Flash-Next Unsloth-layout Q4_K_XL-IQ4NL with Q8_0 dense projections and head) with
  `STRATA_Q8_SM60=1 STRATA_Q8_SM60_EXACT=1`: the LOGPOS file is byte-identical to the run without either switch. The plain
  `STRATA_Q8_SM60=1` on the same yardstick: top-1 94.79%, KL 0.0275 against it.
- Speed, 2x P100 layer split, KV int8, the same config with the plain kernel (B) and with the twin (X), BXXB, decode tok/s
  medians of 2: code 60.86 / 59.77, prose 45.50 / 45.44, chat 46.10 / 46.37, tool 47.68 / 48.39, 15K-token prompt 54.90 /
  52.45 (about -1% overall). For reference the plain kernel was +16% decode (code) over the exact kernels on this box (one screening run each).

Failed: none observed.

Not run:
- Another GP100 (PH402) or a P100 with other drivers; compute capability 6.1 (the kernel is not used there).
- The MTP drafter's `STRATA_MTP_Q8_SM60=1` projections with the twin (they go through the same launch and should follow
  it; not checked bit for bit).
- HIP / SYCL: not compiled there (see Approach).

**Limits.** Bitwise to the exact kernels of this revision: a change of `native_small_mmvq_kernel`'s mapping or reduction
needs the twin changed with it (the check above catches the difference). The packed copies still cost their VRAM.

Measured and prepared with Claude Code on this box; the numbers are from its logs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01N976PznRue6hXdFtmrbJNj
