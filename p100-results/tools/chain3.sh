#!/bin/bash
# final validation: waits for chain2 and for the best config, then the 32K ABBA, HumanEval (stock, best), long context
S=/home/user/projects/strata
while kill -0 65235 2>/dev/null; do sleep 10; done
until [ -s $S/p100-configs/strata-bitlamas-p100-best.json ] && [ -s $S/p100-plans/final-long.json ]; do sleep 10; done
cd $S
echo "### ABBA 32K $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/final-32k-abba.json --out p100-results/runs/final-32k-abba > p100-results/runs/final-32k-abba.log 2>&1; tail -12 p100-results/runs/final-32k-abba.log
echo "### HumanEval best $(date +%T)"
.venv/bin/python p100-results/tools/humaneval_serve.py p100-configs/strata-bitlamas-p100-best.json p100-results/runs/humaneval-best > p100-results/runs/humaneval-best.log 2>&1; tail -1 p100-results/runs/humaneval-best.log
echo "### HumanEval stock $(date +%T)"
.venv/bin/python p100-results/tools/humaneval_serve.py p100-configs/strata-bitlamas-stock-2gpu-32k-int8.json p100-results/runs/humaneval-stock > p100-results/runs/humaneval-stock.log 2>&1; tail -1 p100-results/runs/humaneval-stock.log
echo "### long context $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/final-long.json --out p100-results/runs/final-long > p100-results/runs/final-long.log 2>&1; tail -12 p100-results/runs/final-long.log
echo "### chain3 done $(date +%T)"
