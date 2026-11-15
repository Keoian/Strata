#!/bin/bash
S=/home/user/projects/strata
while kill -0 58755 2>/dev/null; do sleep 10; done
echo "### gpu profile c24 $(date +%T)"
$S/p100-results/tools/probe_server.sh $S/p100-configs/exp-c24.json $S/p100-results/runs/probe-gpuprof-c24 STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1 2>&1 | grep -E "decode timing|generated in|died"
grep -E "GPU stages" $S/p100-results/runs/probe-gpuprof-c24/engine.log | head -2
echo "### routing trace $(date +%T)"
$S/.venv/bin/python $S/p100-results/tools/collect_routing_serve.py $S/p100-configs/exp-c24.json $S/p100-results/runs/routing-corpus 2>&1 | tail -4
echo "### quality $(date +%T)"
$S/p100-results/tools/quality_batch.sh S-stock S-numa S-q8sm60 S-int8 S-k8v4 S-q4 G-stock G-downws G-q8sm60 REF 2>&1
echo "### chain1 done $(date +%T)"
