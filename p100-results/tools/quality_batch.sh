#!/bin/bash
# quality_batch.sh: the Phase 3 yardstick runs, one engine at a time (each needs most of the RAM).
#   REF  llama.cpp CPU (llama_logpos)  - the implementation check
#   S-*  Strata quant_quality run (CPU experts unless -gpu), FP16 K/V unless the name says otherwise
set -u
S=/home/user/projects/strata
Q=$S/p100-results/quality
M=/home/user/models/bitlamas-Qwen3.8-Flash-Next-GGUF/Q4_K_XL-IQ4NL/Qwen3.8-Flash-Next-Q4_K_XL-IQ4NL-00001-of-00004.gguf
IDS=$S/data/quant_quality/v1/set.ids.json
CFG=$S/p100-configs/strata-bitlamas-stock-1gpu-32k-int8.json   # GPU1 (x16), no layer split
cd $S
run() {   # name, extra quant_quality args...
  local n=$1; shift
  [ -s "$Q/$n.tsv" ] && { echo "skip $n (done)"; return; }
  echo "=== $n $(date +%T)"
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/quant_quality.py run --config $CFG --out $Q/$n.tsv "$@" 2>&1 | tail -7
}
for job in "$@"; do
  case $job in
    REF) [ -s $Q/REF-llama-cpu.tsv ] || { echo "=== REF $(date +%T)"; /home/user/projects/ref-tools/llama_logpos --model $M --ids $IDS --out $Q/REF-llama-cpu.tsv --threads 36 2>&1 | grep -v "^llama_\|^load\|^print_info\|^\.\+$" | tail -8; } ;;
    S-stock) run S-stock-fp16 ;;
    S-numa) run S-numa-fp16 --env STRATA_ARENA_NUMA=1 ;;
    S-int8) run S-stock-int8 --kv int8 ;;
    S-k8v4) run S-stock-k8v4 --kv k8v4 ;;
    S-q4) run S-stock-q4_0 --kv q4_0 ;;
    S-q8sm60) run S-q8sm60-fp16 --env STRATA_Q8_SM60=1 ;;
    G-stock) run G-stock-fp16 --gpu-experts auto ;;
    G-downws) run G-downws-fp16 --gpu-experts auto --env STRATA_DOWN_WS=1 ;;
    G-q8sm60) run G-q8sm60-fp16 --gpu-experts auto --env STRATA_Q8_SM60=1 ;;
    *) echo "unknown job $job" ;;
  esac
done
