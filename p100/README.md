# Qwen3.8-Flash-Next on two Tesla P100s

The setup and settings that serve Qwen3.8-Flash-Next with Strata on two Tesla P100-PCIE-16GB cards, and the tools used to
tune them. Strata itself is unchanged (v0.2.8, engine 0.1.54, commit 870aedd); everything here is configuration.

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
| `configs/strata-prod-q2_0.json`, `configs/strata-prod-iq3_xxs.json` | the tuned server configs (port 8090, all interfaces, key from `STRATA_API_KEY`, alias `qwen3.8-27b`, thinking off by default, pictures on the CPU reader) |
| `start-flashnext.sh [iq3_xxs\|q2_0]` | starts the server in its own process group, reading the key from a file outside the repo; the default is iq3_xxs |
| `stop-flashnext.sh` | stops that process group and waits for the GPUs to empty |
| `bench/dbench.py URL LABEL [ROUNDS] [KEY]` | the decode benchmark above; appends a row to a results file |
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
