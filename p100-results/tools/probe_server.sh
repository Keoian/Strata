#!/bin/bash
# probe_server.sh CONFIG OUTDIR [ENV=VAL ...] - one server with extra env, numa_maps of the engine, a few requests, the engine log.
set -u
CFG=$1; OUT=$2; shift 2
S=/home/user/projects/strata
PROBE_EXTRA=${PROBE_EXTRA:-}
mkdir -p "$OUT"
[ "${PROBE_MEMLOCK:-0}" = 1 ] && sudo prlimit --memlock=unlimited:unlimited --pid $$ && echo "memlock: $(ulimit -l)"
python3 - "$CFG" "$OUT/config.json" "$OUT/engine.log" "$@" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["log"] = sys.argv[3]; c["port"] = 8091
for kv in sys.argv[4:]:
    k, _, v = kv.partition("="); c.setdefault("env", {})[k] = v
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
cd $S; .venv/bin/python serve/server.py --engine strata --config "$OUT/config.json" --port 8091 --no-api-key > "$OUT/server.log" 2>&1 &
SRV=$!
until curl -s 127.0.0.1:8091/health | grep -q '"loaded": *true'; do sleep 3; kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; done
EPID=$(for p in $(pgrep -f -- "--serve --pack"); do [ "$(readlink /proc/$p/exe)" = "$S/build-p100/strata" ] && echo $p; done | head -1)
echo "engine pid $EPID"
awk '{for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/) {split($i,a,"="); n[a[1]]+=a[2]}} END{for(k in n) printf "%s %.2f GiB\n", k, n[k]*4096/2^30}' /proc/$EPID/numa_maps > "$OUT/numa_maps_total.txt"
awk '{tot=0; for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"="); tot+=a[2]} if (tot*4096 > 2^30) print}' /proc/$EPID/numa_maps > "$OUT/numa_maps_big.txt"
cat "$OUT/numa_maps_total.txt"; cut -c1-300 "$OUT/numa_maps_big.txt"
for q in "Write a Python implementation of an LRU cache with O(1) get and put, type hints, docstrings and a short pytest test suite." "Explain in about 400 words how the printing press changed science in early modern Europe."; do
  curl -s 127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' -d "{${PROBE_EXTRA//\\/}\"model\":\"strata\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":512,\"temperature\":0,\"reasoning_effort\":\"none\"}" > /dev/null
done
sleep 2
grep -E "decode timing|GPU stages|cpu pool|pool phases|generated in|hit rate" "$OUT/engine.log" | tail -12
kill -TERM $SRV; wait $SRV 2>/dev/null; for i in $(seq 1 60); do pgrep -x strata >/dev/null || break; sleep 2; done
