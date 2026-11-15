#!/bin/bash
# Layer-split borrow A/B (upstream #5612) on the served config, with the engine and server of another checkout
# (W, upstream main + the patch): the same main-agent / sub-agent session (split_borrow_ab.py) on fresh engines,
# without the switch (stock), with STRATA_SPLIT_BORROW=1 (borrow), and without it again (stock2, the run-to-run floor).
# Each runs as a transient unit on 127.0.0.1:8080.  Stops the service first and starts it again at the end, also
# when this script fails or is stopped.  Refuses to run unless both cards are at the 150 W limit.
#   EXTRA_ARGS adds engine arguments to every variant ("--adapt-every 0" for the exactness runs).
#   W=~/projects/strata-split-borrow S=p100-results/split-borrow bash p100-results/tools/split_borrow_ab.sh
set -u
R=/home/user/projects/strata
W=$(realpath "${W:-/home/user/projects/strata-split-borrow}")
S=$(realpath -m "${S:-$R/p100-results/split-borrow}")
SEED=${SEED:-8107}
VARIANTS=${VARIANTS:-"stock borrow stock2"}
SERVED=$R/p100-configs/strata-bitlamas-p100-best-256k-k8v4.json
KEY=$(grep -oP '^STRATA_API_KEY=\K.*' ~/.config/strata/bitlamas.env)
mkdir -p $S
gpus() { nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | sed "s/^/  [gpu $1] /"; }
capped() { [ "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | sort -u | tr '\n' ' ')" = "150.00 " ]; }
loaded() { curl -s -m 3 http://127.0.0.1:8080/v1/status -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("loaded") else 1)' 2>/dev/null; }
wait_loaded() { for i in $(seq 1 120); do loaded && return 0; sleep 5; done; return 1; }
config() {   # $1 variant: the served config with W's engine and its own log; borrow sets the switch
  python3 -I - "$SERVED" "$S/agent-$1.json" "$S/engine-$1.log" "$1" "$W/build-p100/strata" "${EXTRA_ARGS:-}" <<'PY'
import json, sys
src, dst, log, v, exe, extra = sys.argv[1:]
c = json.load(open(src))
c["log"] = log
c["exe"] = exe
c["args"] = c["args"] + extra.split()      # EXTRA_ARGS="--adapt-every 0": the exactness runs
if v.startswith("borrow"):
    c.setdefault("env", {})["STRATA_SPLIT_BORROW"] = "1"
json.dump(c, open(dst, "w"), indent=1)
PY
}
SMI=
finish() {
  for v in $VARIANTS; do sudo systemctl stop strata-borrowtest-$v 2>/dev/null; done
  if ! systemctl is-active -q strata-bitlamas; then
    echo "== $(date +%T) starting the service again"
    sudo systemctl start strata-bitlamas
    t0=$(date +%s); wait_loaded && echo "   service loaded in $(( $(date +%s) - t0 )) s" || echo "!! the service did not load"
  fi
  gpus after
  [ -n "$SMI" ] && kill $SMI 2>/dev/null
}
capped || { echo "!! a card is not at the 150 W limit: not starting"; gpus now; exit 2; }
trap finish EXIT
trap 'exit 130' INT TERM

nvidia-smi --query-gpu=timestamp,index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -l 5 > $S/gpus.log &
SMI=$!
echo "== $(date +%T) stopping the service"; gpus before
sudo systemctl stop strata-bitlamas
for v in $VARIANTS; do
  u=strata-borrowtest-$v
  config $v
  : > $S/engine-$v.log
  capped || { echo "!! a card left the 150 W limit: stopping"; gpus $v; break; }
  echo "== $(date +%T) $v: starting"
  sudo systemd-run --quiet --unit=$u --uid=user --gid=user -p WorkingDirectory=$W \
    -p EnvironmentFile=/home/user/.config/strata/bitlamas.env -p LimitMEMLOCK=infinity -p TimeoutStopSec=180 \
    $R/.venv/bin/python serve/server.py --engine strata --config $S/agent-$v.json --host 127.0.0.1 --port 8080
  t0=$(date +%s)
  if ! wait_loaded; then echo "!! $v did not load"; journalctl -u $u -o cat --no-pager | tail -20; tail -20 $S/engine-$v.log; sudo systemctl stop $u; continue; fi
  echo "   loaded in $(( $(date +%s) - t0 )) s"; gpus $v
  (cd $R && .venv/bin/python p100-results/tools/split_borrow_ab.py --out $S/$v.json --seed $SEED) 2>&1 | sed 's/^/   /'
  gpus $v
  grep -E "conversation cache: (borrowed|restored|cannot|discard|skip)" $S/engine-$v.log | sed -E 's/^/   engine: /' | cut -c1-200
  sudo systemctl stop $u
  sleep 3
done
finish
trap - EXIT
logs=; runs=; for v in $VARIANTS; do [ -f $S/$v.json ] && { runs="$runs $S/$v.json"; logs="$logs $S/engine-$v.log"; }; done
(cd $R && .venv/bin/python p100-results/tools/agent_cache_ab.py --table $runs --engine-logs $logs) | tee $S/table.md
