#!/bin/bash
# start-flashnext.sh [q2_0|iq3_xxs]: serve Qwen3.8-Flash-Next (Strata engine, both P100s) as production on :8090.
# OpenAI API at http://<host>:8090/v1, the same key as the 27B (/work/qwen-api-keys), model names qwen3.8-flash-next,
# qwen3.8-27b (alias, so existing clients keep working) or anything else. Thinking is off unless a request asks
# (reasoning_effort low|medium|high, or chat_template_kwargs.enable_thinking=true).
# Default iq3_xxs (production since 10-31 10:57 UTC: better quality); q2_0 = fastest, ~20% quicker, lower quality. The GPUs must be free (stop the 27B first).
set -u
M=${1:-iq3_xxs}; S=/work/flashnext/strata; CFG=$S/p100/configs/strata-prod-$M.json; PGF=/work/flashnext/prod.pgid
[ -f $CFG ] || { echo "no $CFG"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
[ "$used" -lt 100 ] || { echo "GPUs busy ($used MiB used): stop the other server first"; exit 1; }
export STRATA_API_KEY=$(grep -v '^#' /work/qwen-api-keys | head -1)
cd $S
setsid "$S/.venv/bin/python" "$S/serve/server.py" --engine strata --config "$CFG" --port 8090 > /work/flashnext/serve-8090.log 2>&1 < /dev/null &
echo $! > $PGF
for i in $(seq 240); do curl -sf -m 3 localhost:8090/health 2>/dev/null | grep -q '"loaded": true' && break
  ps -o stat= -p $(cat $PGF) | grep -qv Z || { echo "server exited"; tail -20 /work/flashnext/serve-8090.log; exit 1; }; sleep 5; done
curl -s -m 3 localhost:8090/health; echo
