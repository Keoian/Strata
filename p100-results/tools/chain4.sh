#!/bin/bash
S=/home/user/projects/strata
while kill -0 74375 2>/dev/null; do sleep 10; done
cd $S
echo "### ABBA 32K (rerun) $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/final-32k-abba.json --out p100-results/runs/final-32k-abba > p100-results/runs/final-32k-abba.log 2>&1; tail -12 p100-results/runs/final-32k-abba.log
echo "### chain4 done $(date +%T)"
