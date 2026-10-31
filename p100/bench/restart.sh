#!/bin/bash
# restart.sh CONFIG.json [PORT]: stop the running Strata test server (its process group, by the pid file), wait for the
# GPUs to empty, start serve/server.py with CONFIG, wait until /health says loaded. Never touches llama-server.
set -u
CFG=$1; PORT=${2:-8091}; S=/work/flashnext/strata; PIDF=/work/flashnext/bench/server.pgid
if [ -f $PIDF ]; then PG=$(cat $PIDF); kill -TERM -- -$PG 2>/dev/null; alive() { ps -eo pgid=,stat= | awk -v g=$PG '$1==g && $2 !~ /Z/' | grep -q .; }; for i in $(seq 60); do alive || break; sleep 1; done; kill -KILL -- -$PG 2>/dev/null; rm -f $PIDF; fi
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')" -lt 100 ]; do sleep 2; done
# cool down before a measured start (cards under 60 C)
until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -lt 60 ]; do sleep 5; done
cd $S
setsid "$S/.venv/bin/python" "$S/serve/server.py" --engine strata --config "$CFG" --port $PORT ${EXTRA_SERVER_ARGS:-} > /work/flashnext/serve-$(basename $CFG .json).log 2>&1 < /dev/null &
echo $! > $PIDF
for i in $(seq 240); do curl -sf -m 3 localhost:$PORT/health 2>/dev/null | grep -q '"loaded": true' && break; ps -o stat= -p $! | grep -qv Z || { echo "server exited"; tail -20 /work/flashnext/serve-$(basename $CFG .json).log; exit 1; }; sleep 5; done
curl -s -m 3 localhost:$PORT/health; echo
