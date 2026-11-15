#!/bin/bash
S=/home/user/projects/strata
while kill -0 15379 2>/dev/null; do sleep 10; done
cd $S
sudo systemctl stop strata-bitlamas.service; for i in $(seq 1 60); do pgrep -f -- "--serve --pack" >/dev/null || break; sleep 3; done
echo "### yardstick: Q8 twin, CPU experts (placement fixed) $(date +%T)"
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tools/quant_quality.py run --config p100-configs/strata-bitlamas-stock-1gpu-32k-int8.json   --out p100-results/quality/S-q8exact-fp16.tsv --env STRATA_Q8_SM60=1 --env STRATA_Q8_SM60_EXACT=1 --env STRATA_ARENA_NUMA=1 > p100-results/quality/S-q8exact-fp16.log 2>&1; tail -2 p100-results/quality/S-q8exact-fp16.log
cmp p100-results/quality/S-stock-fp16.tsv p100-results/quality/S-q8exact-fp16.tsv && echo "BYTE-IDENTICAL to S-stock" || echo "DIFFERS from S-stock"
sleep 5; sudo systemctl start strata-bitlamas.service; echo "service: $(systemctl is-active strata-bitlamas.service)"
echo "### chain11 done $(date +%T)"
