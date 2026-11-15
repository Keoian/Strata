#!/bin/bash
S=/home/user/projects/strata
while kill -0 4195 2>/dev/null; do sleep 10; done
cd $S
echo "### PR evidence: stock vs stock+STRATA_ARENA_NUMA $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/pr-numa-stock.json --out p100-results/runs/pr-numa-stock > p100-results/runs/pr-numa-stock.log 2>&1; tail -5 p100-results/runs/pr-numa-stock.log
echo "### chain8 done $(date +%T)"
