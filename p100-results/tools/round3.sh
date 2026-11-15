#!/bin/bash
S=/home/user/projects/strata; T=$S/p100-results/tools
echo "### memlock"; PROBE_MEMLOCK=1 $T/screen2.sh exp-c24 2>&1 | sed 's/^=== exp-c24$/=== exp-c24 (memlock)/'
mv $S/p100-results/runs/screen-exp-c24 $S/p100-results/runs/screen-exp-c24-memlock
$T/screen2.sh exp-c24-adapt160 exp-c24-k8v4
echo "### numa_balancing 0"; sudo sysctl -q kernel.numa_balancing=0; $T/screen2.sh exp-c24; sudo sysctl -q kernel.numa_balancing=1
mv $S/p100-results/runs/screen-exp-c24 $S/p100-results/runs/screen-exp-c24-nonumabal
echo "### presence_penalty 0 requests"; export PROBE_EXTRA='\"presence_penalty\":0,'
$T/screen2.sh exp-c24 exp-c24-pipe2 exp-c24-pipe2:STRATA_MTP_STAGE0=1
mv $S/p100-results/runs/screen-exp-c24 $S/p100-results/runs/screen-exp-c24-pp0; mv $S/p100-results/runs/screen-exp-c24-pipe2 $S/p100-results/runs/screen-exp-c24-pipe2-pp0
echo "### done"; cat /proc/sys/kernel/numa_balancing
