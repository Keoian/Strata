#!/bin/bash
S=/home/user/projects/strata
while kill -0 65235 2>/dev/null; do sleep 10; done
echo "### base re-run $(date +%T)"
$S/p100-results/tools/screen2.sh exp-c24 2>&1
mv $S/p100-results/runs/screen-exp-c24 $S/p100-results/runs/screen-exp-c24-r4base
echo "### chain2b done $(date +%T)"
