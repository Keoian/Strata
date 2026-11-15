#!/bin/bash
# screen2.sh NAME... : one probe per p100-configs/NAME.json; PROBE_EXTRA / extra env via NAME:ENV=V,ENV=V
S=/home/user/projects/strata
for spec in "$@"; do
  n=${spec%%:*}; envs=""; [ "$spec" != "$n" ] && envs=$(echo "${spec#*:}" | tr ',' ' ')
  tag=$n${envs:+-$(echo $envs | tr ' =' '-_')}
  echo "=== $tag"
  $S/p100-results/tools/probe_server.sh $S/p100-configs/$n.json $S/p100-results/runs/screen-$tag STRATA_DECODE_TIMING=1 $envs 2>&1 | grep -E "decode timing|generated in|server died"
  grep -E "expert cache [0-9]+ slots|runs layers [0-9]+-" $S/p100-results/runs/screen-$tag/engine.log | head -3
done
