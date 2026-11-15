#!/bin/bash
# screen.sh NAME... : one probe per p100-configs/exp-numa-pcie0-NAME.json (NAME "base" = exp-numa-pcie0.json)
S=/home/user/projects/strata
for n in "$@"; do
  cfg=$S/p100-configs/exp-numa-pcie0-$n.json; [ "$n" = base ] && cfg=$S/p100-configs/exp-numa-pcie0.json
  echo "=== $n"
  $S/p100-results/tools/probe_server.sh $cfg $S/p100-results/runs/screen-$n STRATA_DECODE_TIMING=1 2>&1 | grep -E "decode timing|generated in|server died"
done
