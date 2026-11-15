#!/usr/bin/env python3
"""p100_bench.py - interleaved A/B (ABBA ...) end-to-end runs of Strata servers on this box.

    .venv/bin/python p100-results/tools/p100_bench.py --plan PLAN.json --out DIR

PLAN.json:
    {"base": "strata-bitlamas.json",                  # a run config (serve/server.py --config)
     "arms": {"A": {}, "B": {"env": {"STRATA_X": "1"}, "args_add": ["--flag", "v"], "args_del": ["--other"]}},
     "order": "ABBA",                                  # one fresh server per letter
     "cases": ["code", "edit", "prose", "chat", "tool", "think", "L8k", "L32k"],
     "port": 8090,
     "pre_start": "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"}   # optional: run before every server start

Every arm block: start serve/server.py with the arm's config, wait for /health "loaded", one warm-up request
(excluded), the cases in order, stop the server. The engine's own timers are the numbers (GET /metrics after each
request): prompt tok/s = (prompt_tokens - reused) / prompt_ms, decode tok/s = engine_generated / decode_ms.
nvidia-smi is sampled every second (power, clocks, utilization, temperature, throttle reasons) and the NUMA nodes'
used memory every 5 s, for the whole block. Long cases start with a nonce unique to the request, so nothing of the
filler can come from the conversation cache; `reused` is recorded and checked.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------------ the cases
EDIT_SOURCE = "\n".join(
    [f"def transform_{i:03d}(values: list[int], k: int = {i % 7 + 1}) -> list[int]:\n"
     f"    out = []\n    for v in values:\n        if v % {i % 5 + 2} == 0:\n            out.append(v * k + {i})\n"
     f"        else:\n            out.append(v - k)\n    return out\n" for i in range(60)])

GREEDY = {"temperature": 0, "reasoning_effort": "none"}
SAMPLED_THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 1,
                 "reasoning_effort": "medium", "chat_template_kwargs": {"enable_thinking": True}}


def long_prompt(target_tokens: int, nonce: str) -> str:
    # ~23.1 tokens per filler line with this model's tokenizer (measured: an 8K target of 668 lines read as 15415 prompt
    # tokens; the engine's prompt_tokens is recorded per request)
    lines = max(1, int(target_tokens / 23.1))
    filler = "\n".join(f"record{i:06d}: value={(i * 7919) % 100003} owner=team{i % 37} state={'open' if i % 3 else 'closed'}"
                       for i in range(lines))
    return (f"[session {nonce}]\nThe following is a data file.\n\n{filler}\n\n"
            "Summarize the structure of these records, then explain step by step how you would find every closed record "
            "owned by team12 and compute the sum of their values. Be specific.")


def case_request(name: str, rep: int, arm: str) -> tuple[dict, int]:
    """(request body, max_tokens) for a case. Every request carries a marker unique to (case, rep, arm block)."""
    tag = f"[case {name}-{rep}-{arm}-{time.time_ns()}]\n"
    if name == "code":
        msg, opts, n = tag + ("Write a Python implementation of an LRU cache with O(1) get and put, type hints, docstrings "
                              "and a short pytest test suite."), GREEDY, 512
    elif name == "edit":
        msg, opts, n = tag + ("Here is a module:\n\n```python\n" + EDIT_SOURCE + "\n```\n\nRewrite transform_017 and transform_042 "
                              "to use list comprehensions, keep their behaviour identical, and show only the two new functions."), GREEDY, 512
    elif name == "prose":
        msg, opts, n = tag + "Explain in about 400 words how the printing press changed science in early modern Europe.", GREEDY, 512
    elif name == "chat":
        msg, opts, n = tag + "What are three practical ways to keep a sourdough starter healthy? Answer briefly.", GREEDY, 256
    elif name == "tool":
        body = {"messages": [{"role": "user", "content": tag + "Use get_weather for Zagreb in celsius."}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Synthetic benchmark lookup; no external action is performed.",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                                   "required": ["city", "unit"], "additionalProperties": False}}}],
                "tool_choice": "auto", "max_tokens": 128, **GREEDY}
        return body, 128
    elif name == "think":
        msg, opts, n = tag + ("A farmer has 17 sheep. All but 9 run away, then he buys twice as many as remain and sells a third "
                              "of the flock. How many sheep does he have? Show your reasoning."), SAMPLED_THINK, 1024
    elif name.startswith("L") and name.endswith("k"):
        msg, opts, n = long_prompt(int(name[1:-1]) * 1024 - 300, f"{name}-{rep}-{arm}-{time.time_ns()}"), GREEDY, 256
    else:
        raise SystemExit(f"unknown case {name}")
    return {"messages": [{"role": "user", "content": msg}], "max_tokens": n, **opts}, n


# ------------------------------------------------------------------------------------------------ server control
def get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_loaded(url: str, proc: subprocess.Popen, limit_s: float = 1800) -> float:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} before loading")
        try:
            h = get(url + "/health", 5)
            if h.get("loaded"):
                return time.time() - t0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError("server did not load in time")


def stream(url: str, body: dict, timeout: float = 1800) -> dict:
    body = {**body, "model": "strata", "stream": True, "stream_options": {"include_usage": True}}
    data = json.dumps(body).encode()
    t0 = time.perf_counter()
    first = first_answer = None
    texts, reasoning, finish, usage, tool_calls = [], [], None, {}, 0
    req = urllib.request.Request(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            c = json.loads(p)
            usage = c.get("usage") or usage
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    first = first if first is not None else time.perf_counter() - t0
                if d.get("content"):
                    texts.append(d["content"])
                    first = first if first is not None else time.perf_counter() - t0
                    first_answer = first_answer if first_answer is not None else time.perf_counter() - t0
                if d.get("tool_calls"):
                    tool_calls += 1
                    first = first if first is not None else time.perf_counter() - t0
                    first_answer = first_answer if first_answer is not None else time.perf_counter() - t0
                finish = ch.get("finish_reason") or finish
    return {"request_sha256": hashlib.sha256(data).hexdigest(), "ttft_s": first, "first_answer_s": first_answer,
            "total_s": time.perf_counter() - t0, "finish": finish, "usage": usage, "tool_call_chunks": tool_calls,
            "text": "".join(texts), "reasoning": "".join(reasoning)}


class Sampler:
    """nvidia-smi every second and the NUMA nodes' used memory every 5 s, into the block's folder."""

    def __init__(self, out: pathlib.Path):
        self.out = out
        self.stop = threading.Event()
        q = ("timestamp,index,power.draw,power.limit,enforced.power.limit,utilization.gpu,utilization.memory,clocks.sm,"
             "clocks.mem,temperature.gpu,memory.used,clocks_event_reasons.active,pcie.link.width.current")
        self.smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,nounits", "-lms", "1000"],
                                    stdout=open(out / "gpu-samples.csv", "w"), stderr=subprocess.DEVNULL)
        self.t = threading.Thread(target=self._numa, daemon=True)
        self.t.start()

    def _numa(self):
        with open(self.out / "numa-samples.csv", "w") as f:
            f.write("time,node,mem_used_gib,mem_free_gib,file_pages_gib\n")
            while not self.stop.is_set():
                for n in (0, 1):
                    kv = {}
                    for line in open(f"/sys/devices/system/node/node{n}/meminfo"):
                        p = line.split()
                        kv[p[2].rstrip(":")] = int(p[3])
                    f.write(f"{time.time():.0f},{n},{kv['MemUsed'] / 2**20:.2f},{kv['MemFree'] / 2**20:.2f},"
                            f"{kv.get('FilePages', 0) / 2**20:.2f}\n")
                f.flush()
                self.stop.wait(5)

    def close(self):
        self.stop.set()
        self.smi.send_signal(signal.SIGINT)
        try:
            self.smi.wait(5)
        except subprocess.TimeoutExpired:
            self.smi.kill()


def arm_config(base: dict, arm: dict) -> dict:
    # "config_file": the arm is another run config entirely (then args_add / env / config apply on top of it)
    cfg = copy.deepcopy(json.loads((ROOT / arm["config_file"]).read_text()) if arm.get("config_file") else base)
    args = list(cfg.get("args", []))
    for flag in arm.get("args_del", []):          # a flag and its value (when the next item is not a flag)
        if flag in args:
            i = args.index(flag)
            del args[i:i + (2 if i + 1 < len(args) and not args[i + 1].startswith("--") else 1)]
    add = arm.get("args_add", [])
    i = 0
    while i < len(add):                           # replace a flag's value when it is already there
        flag = add[i]
        has_val = i + 1 < len(add) and not add[i + 1].startswith("--")
        if flag in args and has_val:
            j = args.index(flag)
            args[j + 1] = add[i + 1]
        elif flag not in args:
            args += add[i:i + (2 if has_val else 1)]
        i += 2 if has_val else 1
    cfg["args"] = args
    cfg["env"] = {**cfg.get("env", {}), **arm.get("env", {})}
    for k, v in arm.get("config", {}).items():
        cfg[k] = v
    return cfg


def run_block(letter: str, arm: dict, base: dict, cases: list[str], rep: int, out: pathlib.Path, port: int,
              extra: dict | None = None, pre_start: str | None = None) -> list[dict]:
    bdir = out / f"block{rep:02d}-{letter}"
    bdir.mkdir(parents=True, exist_ok=True)
    cfg = arm_config(base, arm)
    cfg["log"] = str(bdir / "engine.log")
    cfg["port"] = port
    (bdir / "config.json").write_text(json.dumps(cfg, indent=1))
    url = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    if pre_start:                                    # plan "pre_start", e.g. a cold file cache as after a boot
        r = subprocess.run(pre_start, shell=True)
        print(f"[{letter}{rep}] pre_start: exit {r.returncode}", flush=True)
        if r.returncode != 0:
            raise SystemExit(f"pre_start failed ({r.returncode}): {pre_start}")
    srv = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata",
                            "--config", str(bdir / "config.json"), "--port", str(port), "--no-api-key"],
                           cwd=str(ROOT), stdout=open(bdir / "server.log", "w"), stderr=subprocess.STDOUT, env=env)
    sampler = Sampler(bdir)
    rows = []
    try:
        load_s = wait_loaded(url, srv)
        print(f"[{letter}{rep}] loaded in {load_s:.0f} s", flush=True)
        stream(url, {"messages": [{"role": "user", "content": f"[warmup {letter}{rep}] Reply with READY."}],
                     "max_tokens": 16, **GREEDY})
        for name in cases:
            body, cap = case_request(name, rep, letter)
            body = {**body, **(extra or {})}         # plan "request_extra" (e.g. {"presence_penalty": 0})
            t_start = time.time()
            try:
                r = stream(url, body)
            except urllib.error.HTTPError as e:          # e.g. 400: the prompt does not fit; recorded, the block goes on
                print(f"[{letter}{rep}] {name}: HTTP {e.code}: {e.read()[:200]!r}", flush=True)
                rows.append({"arm": letter, "rep": rep, "case": name, "error": e.code, "decode_tok_s": None, "prompt_tok_s": None,
                             "ttft_s": None})
                continue
            m = get(url + "/metrics")
            eng = (m.get("requests") or [{}])[0]
            gen = eng.get("engine_generated") or 0
            row = {"arm": letter, "rep": rep, "case": name, "cap": cap, "t_start": t_start, **r, "engine": eng,
                   "prompt_tok_s": ((eng.get("prompt_tokens", 0) - (eng.get("reused") or 0)) / (eng["prompt_ms"] / 1000))
                   if eng.get("prompt_ms") else None,
                   "decode_tok_s": gen / (eng["decode_ms"] / 1000) if eng.get("decode_ms") else None}
            rows.append(row)
            print(f"[{letter}{rep}] {name:6s} prompt {eng.get('prompt_tokens')} (reused {eng.get('reused')}) "
                  f"pp {row['prompt_tok_s'] or 0:7.1f} tok/s  decode {gen} tok {row['decode_tok_s'] or 0:6.2f} tok/s  "
                  f"accepted {eng.get('drafts_accepted')}/{eng.get('drafts_offered')}  finish {r['finish']}", flush=True)
        (bdir / "metrics-final.json").write_text(json.dumps(get(url + "/metrics"), indent=1))
        (bdir / "status.json").write_text(json.dumps(get(url + "/v1/status"), indent=1))
    finally:
        sampler.close()
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(120)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()
        time.sleep(5)
    (bdir / "rows.json").write_text(json.dumps(rows, indent=1))
    return rows


def summarize(rows: list[dict], arms: list[str]) -> dict:
    out = {}
    for case in sorted({r["case"] for r in rows}):
        out[case] = {}
        for a in arms:
            d = [r["decode_tok_s"] for r in rows if r["case"] == case and r["arm"] == a and r["decode_tok_s"]]
            p = [r["prompt_tok_s"] for r in rows if r["case"] == case and r["arm"] == a and r["prompt_tok_s"]]
            t = [r["ttft_s"] for r in rows if r["case"] == case and r["arm"] == a and r["ttft_s"]]
            out[case][a] = {"n": len(d), "decode_median": statistics.median(d) if d else None,
                            "decode_p95": sorted(d)[min(len(d) - 1, int(round(0.95 * (len(d) - 1))))] if d else None,
                            "decode_p05": sorted(d)[int(round(0.05 * (len(d) - 1)))] if d else None,
                            "decode_all": d, "prompt_median": statistics.median(p) if p else None,
                            "ttft_median": statistics.median(t) if t else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(pathlib.Path(a.plan).read_text())
    base = json.loads((ROOT / plan["base"]).read_text() if not os.path.isabs(plan["base"]) else pathlib.Path(plan["base"]).read_text())
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    rows = []
    seen = {}
    for letter in plan["order"]:
        seen[letter] = seen.get(letter, 0) + 1
        rows += run_block(letter, plan["arms"][letter], base, plan["cases"], seen[letter], out, plan.get("port", 8090),
                          plan.get("request_extra"), plan.get("pre_start"))
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    s = summarize(rows, sorted(plan["arms"]))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    for case, v in s.items():
        print(case, " | ".join(f"{k}: decode {x['decode_median'] or 0:.2f} pp {x['prompt_median'] or 0:.1f} (n={x['n']})"
                               for k, x in v.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
