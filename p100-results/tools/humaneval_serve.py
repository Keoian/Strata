#!/usr/bin/env python3
"""humaneval_serve.py CONFIG OUTDIR [--limit N] - HumanEval (164 problems, openai/human-eval data) through one Strata
server: greedy chat requests with the server's defaults (reasoning off), the reply's ```python block, then the problem's
own tests in a subprocess (python3 -I, a fresh temporary directory, 15 s timeout). pass@1 and per-request decode speed.

Program = the prompt up to the target function's `def` (its imports and helpers) + the model's code + the tests +
`check(entry_point)`. The model's code may redefine anything in that prefix.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path("/home/user/datasets/humaneval/HumanEval.jsonl.gz")
PORT = 8093


def post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return json.loads(r.read())


def extract(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    return text


def prefix_of(prompt: str, entry: str) -> str:
    m = re.search(rf"^def {re.escape(entry)}\s*\(", prompt, re.M)
    return prompt[:m.start()] if m else ""


def run_tests(program: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="he-") as d:
        p = pathlib.Path(d) / "prog.py"
        p.write_text(program)
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=d, capture_output=True, text=True, timeout=15)
            return r.returncode == 0, (r.stderr or "")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(a.config))
    cfg["log"] = str(out / "engine.log")
    cfg["port"] = PORT
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    probs = [json.loads(l) for l in gzip.open(DATA, "rt")]
    if a.limit:
        probs = probs[:a.limit]
    srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata", "--config",
                            str(out / "config.json"), "--port", str(PORT), "--no-api-key"], cwd=str(ROOT),
                           stdout=open(out / "server.log", "w"), stderr=subprocess.STDOUT)
    rows = []
    try:
        while True:
            time.sleep(3)
            if srv.poll() is not None:
                raise SystemExit("server died")
            try:
                if get("/health").get("loaded"):
                    break
            except Exception:
                pass
        post("/v1/chat/completions", {"model": "strata", "messages": [{"role": "user", "content": "Reply with READY."}],
                                      "max_tokens": 8, "temperature": 0, "reasoning_effort": "none"})
        for i, pb in enumerate(probs):
            msg = ("Complete the following Python function. Reply with the complete function (and any imports it needs) "
                   "in a single ```python code block, and nothing else.\n\n```python\n" + pb["prompt"] + "```")
            t0 = time.time()
            r = post("/v1/chat/completions", {"model": "strata", "messages": [{"role": "user", "content": msg}],
                                              "max_tokens": 768, "temperature": 0, "reasoning_effort": "none"})
            text = r["choices"][0]["message"].get("content") or ""
            m = (get("/metrics").get("requests") or [{}])[0]
            code = extract(text)
            program = prefix_of(pb["prompt"], pb["entry_point"]) + "\n" + code + "\n\n" + pb["test"] + \
                f"\n\ncheck({pb['entry_point']})\n"
            ok, err = run_tests(program)
            dec = (m.get("engine_generated") or 0) / (m["decode_ms"] / 1000) if m.get("decode_ms") else None
            rows.append({"task_id": pb["task_id"], "passed": ok, "err": err, "completion_tokens": r["usage"]["completion_tokens"],
                         "decode_tok_s": dec, "s": time.time() - t0, "text": text})
            print(f"{i:3d} {pb['task_id']:16s} {'PASS' if ok else 'fail'} {r['usage']['completion_tokens']:4d} tok "
                  f"{dec or 0:6.1f} tok/s", flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(180)
        except subprocess.TimeoutExpired:
            srv.kill()
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    passed = sum(r["passed"] for r in rows)
    decs = sorted(r["decode_tok_s"] for r in rows if r["decode_tok_s"])
    summ = {"n": n, "passed": passed, "pass_at_1": passed / n if n else 0,
            "decode_median": decs[len(decs) // 2] if decs else None,
            "completion_tokens": sum(r["completion_tokens"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
