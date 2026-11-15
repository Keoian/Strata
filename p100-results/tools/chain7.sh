#!/bin/bash
S=/home/user/projects/strata
cd $S
echo "### NUMA layers vs halves $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/numa-layers-vs-halves.json --out p100-results/runs/numa-layers-vs-halves > p100-results/runs/numa-layers-vs-halves.log 2>&1; tail -6 p100-results/runs/numa-layers-vs-halves.log
echo "### chain7 done $(date +%T)"
