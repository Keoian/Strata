#!/usr/bin/env python3
"""collect_routing_serve.py CONFIG OUTDIR - one server with --dump-routing, every bench/profile_corpus.json prompt once
(greedy, reasoning off, 250 tokens), then stop. The trace is the decode windows' routing (the pool's doorbell)."""
import json, os, pathlib, signal, subprocess, sys, time, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[2]
cfg = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
cfg["args"] += ["--dump-routing", str(out / "serve.trace")]; cfg["log"] = str(out / "engine.log"); cfg["port"] = 8092
(out / "config.json").write_text(json.dumps(cfg, indent=1))
srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                        str(out / "config.json"), "--port", "8092", "--no-api-key"], cwd=str(ROOT),
                       stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
try:
    while True:
        time.sleep(3)
        if srv.poll() is not None: raise SystemExit("server died")
        try:
            if json.loads(urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=5).read()).get("loaded"): break
        except Exception: pass
    corpus = json.load(open(ROOT / "bench/profile_corpus.json"))
    for i, c in enumerate(corpus):
        body = {"model": "strata", "messages": [{"role": "user", "content": c["p"]}], "max_tokens": 250,
                "temperature": 0, "reasoning_effort": "none"}
        t = time.time()
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
        print(f"{i:2d} {c['d']:10s} {r['usage']['completion_tokens']:4d} tok {time.time() - t:5.1f} s", flush=True)
finally:
    srv.send_signal(signal.SIGTERM); srv.wait(180)
print("trace bytes", os.path.getsize(out / "serve.trace"))
