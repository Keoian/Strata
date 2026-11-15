#!/bin/bash
S=/home/user/projects/strata
cd $S
echo "memlock: $(ulimit -l)"; [ "$(ulimit -l)" = unlimited ] || exit 1
sudo systemctl stop strata-bitlamas.service; for i in $(seq 1 60); do pgrep -f -- "--serve --pack" >/dev/null || break; sleep 3; done
echo "### yardstick: full tuned set with the Q8 twin, GPU experts $(date +%T)"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/quant_quality.py run --config p100-configs/strata-bitlamas-stock-1gpu-32k-int8.json \
  --out p100-results/quality/G-tuned-q8exact-fp16.tsv --gpu-experts auto --env STRATA_ARENA_NUMA=1 --env STRATA_DOWN_WS=1 \
  --env STRATA_SPEC_COST=1 --env STRATA_Q8_SM60=1 --env STRATA_Q8_SM60_EXACT=1 > p100-results/quality/G-tuned-q8exact-fp16.log 2>&1; tail -3 p100-results/quality/G-tuned-q8exact-fp16.log
cmp p100-results/quality/G-stock-fp16.tsv p100-results/quality/G-tuned-q8exact-fp16.tsv && echo "BYTE-IDENTICAL to G-stock" || echo "DIFFERS from G-stock"
echo "### speed: Q8 twin vs Q8_SM60 $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/q8exact-vs-q8.json --out p100-results/runs/q8exact-vs-q8 > p100-results/runs/q8exact-vs-q8.log 2>&1; tail -6 p100-results/runs/q8exact-vs-q8.log
sleep 10; sudo systemctl start strata-bitlamas.service; echo "service: $(systemctl is-active strata-bitlamas.service)"
echo "### chain10 done $(date +%T)"
