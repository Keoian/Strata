#!/bin/bash
S=/home/user/projects/strata
while kill -0 82814 2>/dev/null; do sleep 10; done
cd $S
echo "### 256K $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/long-256k.json --out p100-results/runs/long-256k > p100-results/runs/long-256k.log 2>&1; tail -6 p100-results/runs/long-256k.log
echo "### chain5 done $(date +%T)"
