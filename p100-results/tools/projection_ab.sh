#!/bin/bash
# The experimental speed projection (control vector) on the served config, loaded but off by default: fresh engines,
#   stock     the served config, --adapt-every 0                      -> the exactness prompts
#   esp       + the vector, default off, --adapt-every 0              -> the exactness prompts (default, then ON)
#   esplive   + the vector, default off, as it would be served        -> agents with it on and off in turn, speed
# Each runs as a transient unit on 127.0.0.1:8080.  Stops the service first and starts it again at the end, also when
# this script fails or is stopped.  Refuses to run unless both cards are at the 150 W limit.
#   S=p100-results/projection bash p100-results/tools/projection_ab.sh
set -u
R=/home/user/projects/strata
S=$(realpath -m "${S:-$R/p100-results/projection}")
SEED=${SEED:-8107}
VARIANTS=${VARIANTS:-"stock esp esplive"}
VEC=$R/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf
SERVED=$R/p100-configs/strata-bitlamas-p100-best-256k-k8v4.json
KEY=$(grep -oP '^STRATA_API_KEY=\K.*' ~/.config/strata/bitlamas.env)
mkdir -p $S
gpus() { nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | sed "s/^/  [gpu $1] /"; }
capped() { [ "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | sort -u | tr '\n' ' ')" = "150.00 " ]; }
loaded() { curl -s -m 3 http://127.0.0.1:8080/v1/status -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("loaded") else 1)' 2>/dev/null; }
wait_loaded() { for i in $(seq 1 120); do loaded && return 0; sleep 5; done; return 1; }
mem() { echo "   mem $1: available $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB, swap used $(free -m | awk '/Swap/{print $3}') MiB"; }
config() {
  python3 -I - "$SERVED" "$S/agent-$1.json" "$S/engine-$1.log" "$1" "$VEC" <<'PY'
import json, sys
src, dst, log, v, vec = sys.argv[1:]
c = json.load(open(src))
c["log"] = log
if v.startswith("esp"):
    c["args"] += ["--control-vector-scaled", vec + ":1.0", "--control-vector-layer-range", "4", "44",
                  "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    c.setdefault("sampling", {})["experimental_speed_projection"] = False
if v != "esplive":
    c["args"] += ["--adapt-every", "0"]
json.dump(c, open(dst, "w"), indent=1)
PY
}
SMI=; MEM=
finish() {
  for v in $VARIANTS; do sudo systemctl stop strata-esptest-$v 2>/dev/null; done
  if ! systemctl is-active -q strata-bitlamas; then
    echo "== $(date +%T) starting the service again"
    sudo systemctl start strata-bitlamas
    t0=$(date +%s); wait_loaded && echo "   service loaded in $(( $(date +%s) - t0 )) s" || echo "!! the service did not load"
  fi
  gpus after
  [ -n "$SMI" ] && kill $SMI 2>/dev/null
  [ -n "$MEM" ] && kill $MEM 2>/dev/null
}
capped || { echo "!! a card is not at the 150 W limit: not starting"; gpus now; exit 2; }
trap finish EXIT
trap 'exit 130' INT TERM

nvidia-smi --query-gpu=timestamp,index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -l 5 > $S/gpus.log &
SMI=$!
( while sleep 5; do echo "$(date +%T) $(awk '/MemAvailable|SwapFree/{printf "%s %d MiB  ", $1, $2/1024}' /proc/meminfo)"; done ) > $S/mem.log &
MEM=$!
echo "== $(date +%T) stopping the service"; gpus before
sudo systemctl stop strata-bitlamas
for v in $VARIANTS; do
  u=strata-esptest-$v
  config $v
  : > $S/engine-$v.log
  capped || { echo "!! a card left the 150 W limit: stopping"; gpus $v; break; }
  echo "== $(date +%T) $v: starting"
  sudo systemd-run --quiet --unit=$u --uid=user --gid=user -p WorkingDirectory=$R \
    -p EnvironmentFile=/home/user/.config/strata/bitlamas.env -p LimitMEMLOCK=infinity -p TimeoutStopSec=180 \
    $R/.venv/bin/python serve/server.py --engine strata --config $S/agent-$v.json --host 127.0.0.1 --port 8080
  t0=$(date +%s)
  if ! wait_loaded; then echo "!! $v did not load"; journalctl -u $u -o cat --no-pager | tail -20; tail -20 $S/engine-$v.log; sudo systemctl stop $u; continue; fi
  echo "   loaded in $(( $(date +%s) - t0 )) s"; gpus $v; mem loaded
  grep -m3 -i -E "control vector|conversation cache sized" $S/engine-$v.log | sed "s/^/   engine: /" | cut -c1-220
  grep -E "VRAM free with everything" $S/engine-$v.log | sed "s/^/   engine: /"
  run() { (cd $R && .venv/bin/python p100-results/tools/projection_ab.py "$@" --seed $SEED) 2>&1 | sed "s/^/   /"; }
  case $v in
    stock)   run --mode exact --out $S/stock.json ;;
    esp)     run --mode exact --out $S/off.json; run --mode exact --on --out $S/on.json ;;
    esplive) run --mode live --out $S/live.json ;;
  esac
  mem "after the session"; gpus $v
  grep -E "conversation cache: (parked|restored|borrowed|dropped|cannot|skip)" $S/engine-$v.log | sed -E "s/^/   engine: /" | cut -c1-200
  sudo systemctl stop $u
  sleep 3
done
finish
trap - EXIT
(cd $R && .venv/bin/python p100-results/tools/agent_cache_ab.py --table $S/stock.json $S/off.json $S/on.json) | tee $S/table.md
