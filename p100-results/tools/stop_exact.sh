#!/bin/bash
# The exactness check after a Stop mid prompt read: the prototype binary with the old Stop (off) and with both fixes
# (both), each on a fresh transient unit on 127.0.0.1:8080, the same requests (stop_exact.py), then the comparison.
# Stops the service first and starts it again at the end.  S (the folder with stop-<v>.json) must be set by the caller.
set -u
: "${S:?set S to the folder with the stop-<variant>.json configs}"
R=/home/user/projects/strata
SEED=${SEED:-8107}
VARIANTS=${VARIANTS:-"off both"}
KEY=$(grep -oP '^STRATA_API_KEY=\K.*' ~/.config/strata/bitlamas.env)
gpus() { nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | sed "s/^/  [gpu $1] /"; }
loaded() { curl -s -m 3 http://127.0.0.1:8080/v1/status -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("loaded") else 1)' 2>/dev/null; }
wait_loaded() { for i in $(seq 1 120); do loaded && return 0; sleep 5; done; return 1; }

nvidia-smi --query-gpu=timestamp,index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -l 5 > $S/exact-gpus.log &
SMI=$!
echo "== $(date +%T) stopping the service"; gpus before
sudo systemctl stop strata-bitlamas
for v in $VARIANTS; do
  u=strata-exacttest-$v
  : > $S/stop-$v.log
  echo "== $(date +%T) $v: starting"
  sudo systemd-run --quiet --unit=$u --uid=user --gid=user -p WorkingDirectory=$R \
    -p EnvironmentFile=/home/user/.config/strata/bitlamas.env -p LimitMEMLOCK=infinity -p TimeoutStopSec=180 \
    $R/.venv/bin/python serve/server.py --engine strata --config $S/stop-$v.json --host 127.0.0.1 --port 8080
  t0=$(date +%s)
  if ! wait_loaded; then echo "!! $v did not load"; journalctl -u $u -o cat --no-pager | tail -20; sudo systemctl stop $u; continue; fi
  echo "   loaded in $(( $(date +%s) - t0 )) s"; gpus $v
  (cd $R && .venv/bin/python p100-results/tools/stop_exact.py --out $S/exact-$v.json --seed $SEED --unit $u --log $S/stop-$v.log) 2>&1 | sed 's/^/   /'
  cp $S/stop-$v.log $S/exact-$v-engine.log
  gpus $v
  sudo systemctl stop $u
  sleep 3
done
echo "== $(date +%T) starting the service again"
sudo systemctl start strata-bitlamas
t0=$(date +%s); wait_loaded && echo "   service loaded in $(( $(date +%s) - t0 )) s" || echo "!! the service did not load"
gpus after
kill $SMI
(cd $R && for p in ${COMPARE:-"off both"}; do a=${p%%:*}; b=${p##*:}; echo "== $a vs $b"; .venv/bin/python p100-results/tools/stop_exact.py --compare $S/exact-$a.json $S/exact-$b.json; done)
