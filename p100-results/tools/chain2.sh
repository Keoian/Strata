#!/bin/bash
S=/home/user/projects/strata
while kill -0 61280 2>/dev/null; do sleep 10; done
echo "### calibrate $(date +%T)"
cd $S && .venv/bin/python tools/calibrate.py p100-configs/exp-c24.json 2>&1 | tail -40
echo "### round 4 $(date +%T)"
$S/p100-results/tools/screen2.sh exp-c24-prof exp-c24-guswiglu exp-c24-qsarows exp-c24-attndaily exp-c24-tsumpfs exp-c24-grv3 exp-c24-speccost exp-c24-spec3 exp-c24-spec5
echo "### chain2 done $(date +%T)"
