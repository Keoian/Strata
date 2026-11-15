#!/bin/bash
# runs with RLIMIT_MEMLOCK unlimited inherited from its launcher (checked below), then starts the boot service
S=/home/user/projects/strata
cd $S
echo "memlock: $(ulimit -l)"
[ "$(ulimit -l)" = unlimited ] || { echo "memlock not unlimited: stop"; exit 1; }
echo "### NUMA layers vs halves $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/numa-layers-vs-halves.json --out p100-results/runs/numa-layers-vs-halves > p100-results/runs/numa-layers-vs-halves.log 2>&1; tail -6 p100-results/runs/numa-layers-vs-halves.log
echo "### PR evidence: stock vs stock+STRATA_ARENA_NUMA $(date +%T)"
.venv/bin/python p100-results/tools/p100_bench.py --plan p100-plans/pr-numa-stock.json --out p100-results/runs/pr-numa-stock > p100-results/runs/pr-numa-stock.log 2>&1; tail -5 p100-results/runs/pr-numa-stock.log
echo "### start service $(date +%T)"
sleep 10; sudo systemctl start strata-bitlamas.service; echo "service: $(systemctl is-active strata-bitlamas.service)"
echo "### chain9 done $(date +%T)"
