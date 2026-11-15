#!/usr/bin/env python3
"""make_final.py [ENV=V ...] - write the best run config (exp-c24 + the given env) and the long-context configs and plan
(chain3 starts once these exist)."""
import json, pathlib, sys
R = pathlib.Path(__file__).resolve().parents[2]
best = json.loads((R / "p100-configs/exp-c24.json").read_text())
for kv in sys.argv[1:]:
    k, _, v = kv.partition("="); best["env"][k] = v
best["log"] = str(R / "strata-bitlamas.log"); best["port"] = 8080; best["host"] = "127.0.0.1"
best["model_name"] = "qwen3.8-flash-next-bitlamas-q4_k_xl-iq4nl"
(R / "p100-configs/strata-bitlamas-p100-best.json").write_text(json.dumps(best, indent=1))

def ctx(cfg, n, kv):
    c = json.loads(json.dumps(cfg)); a = c["args"]
    a[a.index("--max-context") + 1] = str(n); a[a.index("--kv") + 1] = kv
    if "--kv-resident" not in a: a += ["--kv-resident", "32768"]
    return c
stock = json.loads((R / "p100-configs/strata-bitlamas-stock-2gpu-32k-int8.json").read_text())
(R / "p100-configs/strata-bitlamas-stock-2gpu-128k-int8.json").write_text(json.dumps(ctx(stock, 131072, "int8"), indent=1))
(R / "p100-configs/strata-bitlamas-p100-best-128k-k8v4.json").write_text(json.dumps(ctx(best, 131072, "k8v4"), indent=1))
plan = {"base": "p100-configs/strata-bitlamas-stock-2gpu-128k-int8.json",
        "arms": {"A": {}, "B": {"config_file": "p100-configs/strata-bitlamas-p100-best-128k-k8v4.json"}},
        "order": "ABBA", "cases": ["chat", "L64k", "L120k"], "port": 8090}
(R / "p100-plans/final-long.json").write_text(json.dumps(plan, indent=1))
print("best env:", best["env"])
print("best args:", " ".join(best["args"]))
