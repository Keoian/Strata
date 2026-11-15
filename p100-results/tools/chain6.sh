#!/bin/bash
S=/home/user/projects/strata
cd $S
echo "### yardstick: exact switches on GPU experts $(date +%T)"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/quant_quality.py run --config p100-configs/strata-bitlamas-stock-1gpu-32k-int8.json \
  --out p100-results/quality/G-exactset-fp16.tsv --gpu-experts auto --env STRATA_ARENA_NUMA=1 --env STRATA_DOWN_WS=1 \
  --env STRATA_SPEC_COST=1 > p100-results/quality/G-exactset-fp16.log 2>&1; tail -6 p100-results/quality/G-exactset-fp16.log
cmp p100-results/quality/G-stock-fp16.tsv p100-results/quality/G-exactset-fp16.tsv && echo "BYTE-IDENTICAL to G-stock" || echo "DIFFERS from G-stock"
echo "### ABBA exact vs best $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/exact-vs-best.json --out p100-results/runs/exact-vs-best > p100-results/runs/exact-vs-best.log 2>&1; tail -8 p100-results/runs/exact-vs-best.log
echo "### chain6 done $(date +%T)"
