#!/bin/bash
# A/B of the engine's STOP during a prompt read: the prototype binary with both fixes off, fix 1 only, fixes 1+2.
# Stops the service, runs each variant as a transient unit on 127.0.0.1:8080, starts the service again at the end.
set -u
S=${S:?set S to a folder that holds stop-off.json, stop-stages.json and stop-both.json}
R=/home/user/projects/strata
SEED=${SEED:-8107}
VARIANTS=${VARIANTS:-"off stages both"}
KEY=$(grep -oP '^STRATA_API_KEY=\K.*' ~/.config/strata/bitlamas.env)
gpus() { nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | sed "s/^/  [gpu $1] /"; }
loaded() { curl -s -m 3 http://127.0.0.1:8080/v1/status -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("loaded") else 1)' 2>/dev/null; }
wait_loaded() { for i in $(seq 1 120); do loaded && return 0; sleep 5; done; return 1; }

echo "== $(date +%T) stopping the service"; gpus before
sudo systemctl stop strata-bitlamas
for v in $VARIANTS; do
  u=strata-stoptest-$v
  : > $S/stop-$v.log
  echo "== $(date +%T) $v: starting"
  sudo systemd-run --quiet --unit=$u --uid=user --gid=user -p WorkingDirectory=$R \
    -p EnvironmentFile=/home/user/.config/strata/bitlamas.env -p LimitMEMLOCK=infinity -p TimeoutStopSec=180 \
    $R/.venv/bin/python serve/server.py --engine strata --config $S/stop-$v.json --host 127.0.0.1 --port 8080
  t0=$(date +%s)
  if ! wait_loaded; then echo "!! $v did not load"; journalctl -u $u -o cat --no-pager | tail -20; sudo systemctl stop $u; continue; fi
  echo "   loaded in $(( $(date +%s) - t0 )) s"; gpus $v
  (cd $R && .venv/bin/python p100-results/tools/cancel_repro.py --out $S/stop-$v-result.json --cases s10,s12,s13 --seed $SEED \
     --unit $u --log $S/stop-$v.log) 2>&1 | sed 's/^/   /'
  gpus $v
  sudo systemctl stop $u
  sleep 3
done
echo "== $(date +%T) starting the service again"
sudo systemctl start strata-bitlamas
t0=$(date +%s); wait_loaded && echo "   service loaded in $(( $(date +%s) - t0 )) s" || echo "!! the service did not load"
gpus after
