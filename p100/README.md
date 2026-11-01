# Qwen3.8-Flash-Next on two Tesla P100s

The setup and settings that serve Qwen3.8-Flash-Next with Strata on two Tesla P100-PCIE-16GB cards, and the tools used to
tune them. The production configs run stock Strata (v0.2.8, engine 0.1.54, commit 870aedd) and are configuration
only. This branch also adds two opt-in engine kernels for the P100 ([below](#opt-in-engine-kernels-2026-10-31)); with
their switches off the engine computes exactly what stock does.

| Machine | |
| --- | --- |
| GPUs | 2x Tesla P100-PCIE-16GB (sm_60, no DP4A), 150 W limit, no NVLink; the second card sits on a PCH x4 slot (~3 GB/s) |
| CPU, RAM | i7-7700 (4 cores / 8 threads, AVX2), 47 GiB |
| Engine | Strata's CUDA 12 engine for sm_60 (`STRATA_EXPERIMENTAL_SM60`), compiled with CUDA 12.4 |

## Results

Decode speed: six fixed prompts x 256 tokens, greedy, thinking off, two rounds (`bench/dbench.py`). Prompt speed: one fresh
~4.5K-token prompt. Context 262,144 tokens with int8 KV in every row.

| Pack | Default | Tuned | Prompt, 4.5K tokens | RAM available | Top-1 vs Q8_0 ¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0 | 43.1 tok/s | **52.9 tok/s** (+23%) | 274 tok/s | ~7 GB | 80.6% |
| IQ3_XXS (`--resident-experts`) | 34.3 tok/s | **42.7 tok/s** (+25%) | 224 tok/s | ~5-7 GB | 84.5% |
| IQ3_S (`--resident-experts`) | 29.7 tok/s | not tuned | 192 tok/s | **0.7 GB** | 87.7% |

¹ From Strata's docs/QUANTS.md. IQ3_S needs more than 47 GiB of RAM to be usable here.

`tools/strata_bench.py --long`, Q2_0 default vs tuned:

| | Default | Tuned |
| --- | ---: | ---: |
| Short chat decode | 41.2 tok/s | 50.7 tok/s |
| Agent session follow-ups, decode | 43.8 tok/s | 56.4 tok/s |
| 4K / 16K / 32K prompt | 273 / 345 / 388 tok/s | 272 / 341 / 358 tok/s |

The 32K prompt median is 8% lower when tuned, though the two runs' ranges overlap (default 353-389, tuned 343-367).

Long context (Q2_0 tuned, `tools/needle_bench.py`): a code word in the middle of a 249,498-token prompt was found. The
prompt was read in 771 s (324 tok/s) and decode at that depth ran at ~44 tok/s. At the lowest point, RAM available was
6.7 GB. Both GPUs stayed at 68 C or below. IQ3_XXS at that depth was not measured.

## What the tuning is

Each step was measured on its own (Q2_0 numbers):

1. **Q8_0 dense weights with the P100 kernel.** `tools/q8_dense_gguf.py` writes a copy of shard 1 whose dense projections
   and head are Q8_0 (+3.8 GB; every routed expert keeps its byte offset, so the existing pack still works).
   `STRATA_Q8_SM60=1` runs them on a kernel for sm_60. On its own: 43.1 -> 43.2. Each card then held every layer's dense
   weights, which took VRAM from the expert cache.
2. **Each card keeps only its own layers' dense weights** (`STRATA_STAGE_TRIM=1`, which needs an explicit
   `--layer-split`). Together with step 1, split 33: 50.2.
3. **`--vram-reserve-later-mib 1000`:** the second card keeps 1 GB free. No speed cost, and it clears Strata's
   "card nearly full" warning.
4. **`--spec 3 --spec-min-p 0.7`** instead of 4 / 0.5: fewer wasted draft guesses, 52.0.
5. **`--layer-split 30`:** the second card (on the x4 link) takes more layers, 52.9. For IQ3_XXS: split 36 38.75,
   33 41.4, 30 42.7, 27 41.55.

Tried and dropped: `--pipeline-windows 2`, alone and with `STRATA_MTP_STAGE0=1` (no gain). `STRATA_DOWN_WS=1` (+1%, noise).
Draft depth 2 or 5, and confidence bars 0.3, 0.5 and 0.85 (all slower). Reserving VRAM for layer-major prefill
(`--vram-reserve-mib 1800`): long prompts were no faster (351 tok/s at 36K, against 358 at 32K without it).

## Opt-in engine kernels (2026-10-31)

Two kernels for the P100, each behind its own environment switch, plus Strata's own `--aux-cpus auto`. IQ3_XXS,
production settings otherwise:

| | Stock engine | With the three |
| --- | ---: | ---: |
| Decode, 6 prompts x 256 tokens (two ABAB pairs, 3 rounds each) | 42.1 tok/s | **43.95 tok/s (+4.4%)** |
| Time per verify window (`STRATA_DECODE_TIMING=1`) | 46.7 ms | 45.0 ms (-3.6%) |
| 4.5K-token prompt | 222-225 tok/s | **261 tok/s (+16%)** |
| 16K / 32K prompt (`tools/strata_bench.py --long`) | 232 / 239 tok/s | **366 / 418 tok/s (+57% / +75%)** |
| Expert cache, card 1 / card 2 | 5,228 / 4,127 slots | 6,437 / 5,041 slots |
| Agent session decode / first token of a turn | 46.8 tok/s / 13.5 s | 48.0 tok/s / 11.9 s |

1. **`STRATA_Q8_SM60_GGUF=1`** (with `STRATA_Q8_SM60=1`): the Pascal Q8_0 kernel reads the dense weights straight from
   their GGUF blocks (`q8sm60g` in `src/kernels/cuda/q8_sm60.cuh`), staged through shared memory with aligned 16-byte
   loads, instead of from a packed second copy. The copy was 2.9 GiB over the two cards; the freed VRAM holds 23% more
   experts, which cuts the experts each 8K prompt chunk has to stream (the second card over its x4 link). Every output
   is bitwise the packed kernel's. The output head stays on its packed copy (`STRATA_Q8_SM60_GGUF_HEAD=1` moves it too;
   its 248,320 rows are where the GGUF read is furthest behind).
2. **`STRATA_GR_DOWN_RPF=1`**: the hyper-connection down kernel prefetches its next activation tile into registers
   (`gr_down_rpf_kernel` in `src/kernels/cuda/fused_gr.cu`). Below sm_80 Strata's async copy is a plain load and store,
   so each of the 8 tiles waited for its data. The read at 2 tokens: 65.4 -> 58.3 us. Byte-identical end to end.
3. **`--aux-cpus auto`** (Strata): helper threads on the spare cores.

To run it: `start-flashnext.sh iq3_xxs-opt` (`configs/strata-prod-iq3_xxs-opt.json`: the IQ3_XXS config plus the three,
with the engine built from this branch and copied to its own folder so a later build cannot change it):

```
cmake -S . -B build-opt -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_EXPERIMENTAL_SM60=ON \
    -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
ninja -C build-opt strata
mkdir -p /work/flashnext/engine-opt && cp build-opt/strata build-opt/libstrata_kq.so /work/flashnext/engine-opt/
```

The build needs about 3 GB of free RAM for its largest file (`native_mmvq.cu`), more than this machine has while
Flash-Next is serving: build with the server stopped (6 minutes at -j4 with a warm ccache).

Quality, Strata's yardstick (`tools/quant_quality.py` with `--resident-experts` kept, 5 x 999 positions): the stock
engine is deterministic run to run. With the switches off, or with `STRATA_GR_DOWN_RPF=1` alone, results are
byte-identical to stock. `STRATA_Q8_SM60_GGUF=1` changes where experts run, not how: more VRAM puts more experts on the
GPU, and Strata's GPU and CPU expert paths round differently. Stock against stock with a different cache size differs by
the same amount (top-1 95.6%, mean KL 0.019); perplexity on the yardstick falls as more experts run on the GPU (stock
3,000 slots 4.152, stock auto 4.139, the three 4.132). Prompt mode (20 prompts of 256-1,000 tokens): top-1 20/20, mean
KL 0.0002. Needles at 32K, 64K and 123K tokens were found (read at 407, 439, 484 tok/s).

Tried and not used (measured on IQ3_XXS; run-to-run noise about +-1-2%, bar 2%):

| Idea | Decode | Prompt | |
| --- | ---: | ---: | --- |
| `STRATA_Q8_SM60_GGUF=2` (`q8sm60g2`: aligned loads, no staging; rounding level) | -3% | +9% | slower kernel |
| `STRATA_HC_KSPLIT=2/4/8` (down projection split along K) | worse | - | fails the engine's bitwise self-test, which falls back to a slower read |
| `STRATA_PCIE_FRAC_STAGES=0.6,0.27` / `0.75,0.2` (per-card PCIe share of missed experts) | +0.2% / -2% | - | noise / worse |
| The 27B fork's PRMT+XMAD dot-product emulation | +0.1% | +2% | not bit-identical inside Strata (KL 2e-4); not committed |
| `--adapt-async 1` | +0.5-1% | -8% | |
| `STRATA_PCIE_AUTO_SPLIT=1`, `STRATA_SPLIT_OVERLAP=1`, `STRATA_GR_V3`, `GR_UP_WIDE`, `GR_NORM_FAST` | 0 to +0.6% | 0 | |
| `STRATA_HC_Q8=1` (8-bit hyper-connection weights; lossy) | -2.1% | -9% | |
| CPU workers on the hyperthreads / 4 workers | -2% / 0 | - | |
| Draft depth 4 at 0.7 | +0.4% | - | depth 3 stays |

Why decode moves so little: of a ~45 ms verify window, the expert kernels take ~9 ms, the hyper-connection reads ~7 ms,
waiting for experts that are not in VRAM ~7.5 ms (each miss is a ~188 us CPU round trip on this i7), the dense 8-bit
matrices ~6 ms (already at 450-530 of ~550 GB/s) and the draft 3 ms. The two cards run one after the other.

## Accuracy

- Strata's default `STRATA_ROUTE_TAIL_SKIP=7` skips an expert that missed the GPU cache when every token in the window
  ranks it 7 or lower (Strata's ranks; among the last of its 10). That is not bit-exact. `STRATA_ROUTE_TAIL_SKIP=0` computes every routed expert, at 52.9 -> 50.0
  tok/s.
- The Q8_0 copy rounds the dense weights again. The worst element error is 0.4% of a matrix's largest weight.
- Draft checking is exact: each position's token is drawn from the model's own distribution, and a draft is kept only
  when it matches that draw (src/core/verify.cpp).

## Files

| Path | What |
| --- | --- |
| `configs/strata-prod-iq3_xxs-opt.json` | the IQ3_XXS config with the opt-in kernels and `--aux-cpus auto` (engine from this branch) |
| `configs/strata-prod-q2_0.json`, `configs/strata-prod-iq3_xxs.json` | the tuned server configs (port 8090, all interfaces, key from `STRATA_API_KEY`, alias `qwen3.8-27b`, thinking off by default, pictures on the CPU reader) |
| `start-flashnext.sh [iq3_xxs\|q2_0\|iq3_xxs-opt]` | starts the server in its own process group, reading the key from a file outside the repo; the default is iq3_xxs |
| `stop-flashnext.sh` | stops that process group and waits for the GPUs to empty |
| `bench/dbench.py URL LABEL [ROUNDS] [KEY]` | the decode benchmark above; appends a row to a results file |
| `bench/sbench.py URL LABEL [ROUNDS] [KEY]` | the same with the server's default sampling instead of greedy |
| `bench/pbench.py URL N_WORDS` | one fresh prompt of N random words, prints prompt tok/s |
| `bench/variant.py BASE.json NAME [--args ...] [--env K=V ...] [--set k=json ...]` | writes `strata-NAME.json` next to BASE, with extra engine args, env or keys |
| `bench/restart.sh CONFIG.json [PORT]` | restarts a test server; waits for empty GPUs and cards under 60 C |
| `bench/mon.sh OUT` | logs RAM available and each GPU's memory, temperature and power every 20 s |
| `bench/chat1.py`, `bench/vision1.py` | one chat request; one picture request |

The scripts use this machine's paths (`/work/flashnext`, with the Strata data in `/work/flashnext/Strata-data`).

## Setting it up again

```
PATH=/work/flashnext/tools-venv/bin:$PATH ./setup.sh --family qwen --model Q2_0 --gpus 0,1 --cuda 12 --context 262144 \
    --vision cpu --data-dir /work/flashnext/Strata-data --port 8091 --no-start --yes --no-browser
.venv/bin/python tools/q8_dense_gguf.py <models>/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf <models>/Q2_0-q8dense
```

`setup.sh` needs cmake 3.24 or newer; it compiles the CUDA 12 engine for sm_60. Building the IQ3 packs:

```
.venv/bin/python tools/iq_pack.py --gguf <models>/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out <data>/packs/iq3_xxs
```

`--resident-experts` maps the GGUF in place, so no `experts.bin` is needed. Shard 2 (the 28.8 GB n-gram table) is the
same file for every pack and can be hard-linked. Once a `*-q8dense` copy exists, the original shard 1 is redundant: the
copy holds every byte of it.

Things about this machine that matter:

- The Unraid pool refuses new files at about 36-40 GB free, even though `df` still shows space. Keep at least 50 GB free.
- The container's PID 1 is `sleep infinity`, so finished processes stay zombies. The wait loops here skip processes in
  state Z.
