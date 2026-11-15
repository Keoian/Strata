#!/bin/bash
# Cache limits A/B on the served config: fresh engines, the served config as it is (base) and with BIG_ARGS (big).
# Each variant runs two sessions: the scripted coding-agent session with rewritten history (agent_cache_ab.py, what
# the checkpoint count changes) and eight agents with different system prompts taking turns (cache_limits_ab.py,
# what the parking slots and the budget change).  Each runs as a transient unit on 127.0.0.1:8080.  Stops the
# service first and starts it again at the end, also when this script fails or is stopped.  Refuses to run unless
# both cards are at the 150 W limit.
#   S=p100-results/cache-limits bash p100-results/tools/cache_limits_ab.sh
set -u
R=/home/user/projects/strata
S=$(realpath -m "${S:-$R/p100-results/cache-limits}")
SEED=${SEED:-8107}
VARIANTS=${VARIANTS:-"base big"}
BIG_ARGS=${BIG_ARGS:-"--prompt-cache 12 --conversation-cache-slots 16 --conversation-cache-mib 16384"}
SERVED=$R/p100-configs/strata-bitlamas-p100-best-256k-k8v4.json
KEY=$(grep -oP '^STRATA_API_KEY=\K.*' ~/.config/strata/bitlamas.env)
mkdir -p $S
gpus() { nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | sed "s/^/  [gpu $1] /"; }
capped() { [ "$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | sort -u | tr '\n' ' ')" = "150.00 " ]; }
loaded() { curl -s -m 3 http://127.0.0.1:8080/v1/status -H "Authorization: Bearer $KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("loaded") else 1)' 2>/dev/null; }
wait_loaded() { for i in $(seq 1 120); do loaded && return 0; sleep 5; done; return 1; }
mem() { echo "   mem $1: available $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB, swap used $(free -m | awk '/Swap/{print $3}') MiB"; }
config() {
  python3 -I - "$SERVED" "$S/agent-$1.json" "$S/engine-$1.log" "$1" "$BIG_ARGS" <<'PY'
import json, sys
src, dst, log, v, big = sys.argv[1:]
c = json.load(open(src))
c["log"] = log
if v.startswith("big"):
    c["args"] = c["args"] + big.split()
json.dump(c, open(dst, "w"), indent=1)
PY
}
SMI=; MEM=
finish() {
  for v in $VARIANTS; do sudo systemctl stop strata-cachetest-$v 2>/dev/null; done
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
  u=strata-cachetest-$v
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
  grep -m4 -i -E "conversation cache|checkpoint" $S/engine-$v.log | sed 's/^/   engine: /' | cut -c1-220
  echo "   -- coding-agent session"
  (cd $R && .venv/bin/python p100-results/tools/agent_cache_ab.py --out $S/$v-agent.json --seed $SEED) 2>&1 | sed 's/^/   /'
  mem "after the agent session"
  echo "   -- eight agents in turn"
  (cd $R && .venv/bin/python p100-results/tools/cache_limits_ab.py --out $S/$v-rotation.json --seed $SEED) 2>&1 | sed 's/^/   /'
  mem "after the rotation"; gpus $v
  grep -E "conversation cache: parked" $S/engine-$v.log | tail -1 | sed -E 's/^/   engine: /' | cut -c1-220
  echo "   engine: $(grep -cE 'conversation cache: (cannot|skip|not park)' $S/engine-$v.log) lines about skipped parking"
  sudo systemctl stop $u
  sleep 3
done
finish
trap - EXIT
for s in agent rotation; do
  runs=; for v in $VARIANTS; do [ -f $S/$v-$s.json ] && runs="$runs $S/$v-$s.json"; done
  echo "## $s"; (cd $R && .venv/bin/python p100-results/tools/agent_cache_ab.py --table $runs)   # the server's own counts: one engine log holds both sessions
done | tee $S/table.md
