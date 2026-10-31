#!/bin/bash
# stop-flashnext.sh: stop the Strata production server (its process group) and wait for the GPUs to empty.
PGF=/work/flashnext/prod.pgid; [ -f $PGF ] || { echo "not running (no $PGF)"; exit 0; }
PG=$(cat $PGF); kill -TERM -- -$PG 2>/dev/null
for i in $(seq 60); do ps -eo pgid=,stat= | awk -v g=$PG '$1==g && $2 !~ /Z/' | grep -q . || break; sleep 1; done
kill -KILL -- -$PG 2>/dev/null; rm -f $PGF
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')" -lt 100 ]; do sleep 2; done
echo "Flash-Next stopped"
