#!/usr/bin/env python3
"""Exactness after a Stop in the middle of a prompt read (p100-results/engine-stop/STOP.md).

The same requests, in the same order, on a fresh engine: greedy (temperature 0, no presence penalty) with logprobs
(top 5), and two long prompts cancelled while they are read (6 s and 16 s in, as cancel_repro's s10 and s12).  Run it
once on the engine with the old Stop (STRATA_STOP_STAGES=0 STRATA_STOP_LAYERS=0) and once with both fixes; then
`--compare a.json b.json` says whether every request matches token for token and logprob for logprob.

Before the first cancel the two runs decode the same tokens, so the adaptive expert cache is in the same state; a
cancelled prompt read lends and refills expert slots but decodes nothing.  So a difference after the cancels is the
stop's.

    python p100-results/tools/stop_exact.py --out off.json [--seed 8107] [--unit U --log F]
    python p100-results/tools/stop_exact.py --compare off.json both.json
"""
import argparse
import http.client
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cancel_repro as cr  # noqa: E402

GREEDY = {"temperature": 0, "presence_penalty": 0, "reasoning_effort": "none"}


def ask(label, text, max_tokens, logprobs=True):
    """One non-streamed request; its text, its logprobs entries and the server's timings."""
    body = {"model": "strata", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "stream": False, **GREEDY}
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    t0 = time.monotonic()
    conn = http.client.HTTPConnection(cr.HOST, cr.PORT, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Authorization": "Bearer " + cr.KEY, "Content-Type": "application/json"})
    j = json.loads(conn.getresponse().read() or b"{}")
    conn.close()
    took = round(time.monotonic() - t0, 2)
    if "error" in j:
        return {"label": label, "error": j["error"], "took": took}
    c = j["choices"][0]
    lp = c.get("logprobs") or {}
    t = j.get("timings") or {}
    r = {"label": label, "took": took, "text": c["message"].get("content"),
         "reasoning": c["message"].get("reasoning_content"), "finish": c.get("finish_reason"),
         "usage": j.get("usage"), "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
         "logprobs": [[e["token"], e["logprob"], [[a["token"], a["logprob"]] for a in e.get("top_logprobs") or []]]
                      for e in (lp.get("reasoning_content") or []) + (lp.get("content") or [])]}
    print(f"    {label}: took {took} s, cache_n {r['cache_n']}, prompt_n {r['prompt_n']}, "
          f"{len(r['logprobs'])} logprob rows, {(r['text'] or '')[:50]!r}", flush=True)
    return r


def cancelled(label, text, after):
    q = cr.Req(label, [cr.user_turn(text)], cancel_after=after)
    q.start()
    q.join()
    cr.wait_idle()
    print(f"    {label}: cancelled at {q.r.get('cancelled_at')}, {q.r.get('exception', '')[:60]}", flush=True)
    return q.r


def lines(rnd, n, code=None, at=None):
    out = [f"Line {i}: " + " ".join(rnd.choice(cr.WORDS) for _ in range(12)) + "." for i in range(n)]
    if code is not None:
        out[at] = f"Line {at}: the vault code is {code}."
    return "\n".join(out)


def run(seed):
    rnd = random.Random(seed)
    code_c = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    code_q = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    C = lines(rnd, 1000, code_c, rnd.randrange(600, 900)) + "\n\nWhat is the vault code? Reply with the code only."
    Q = lines(rnd, 700, code_q, rnd.randrange(400, 650)) + "\n\nWhat is the vault code? Reply with the code only."
    L1 = lines(rnd, 1700) + "\n\nHow many lines mention quartz? Reply with a number."
    L2 = lines(rnd, 1700) + "\n\nHow many lines mention basalt? Reply with a number."
    T = "Explain in three sentences why the sky is blue."
    G = "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctests."
    out = {"seed": seed, "codes": {"C": code_c, "Q": code_q}, "requests": [], "cancels": []}
    R, X = out["requests"], out["cancels"]
    cr.wait_idle()
    R.append(ask("T1 short text (fresh engine)", T, 64))
    R.append(ask("P1 the ~29K-token code prompt (fresh)", C, 24))
    X.append(cancelled("L1 ~50K-token prompt, cancelled at 6 s", L1, 6))
    R.append(ask("T2 short text again (after the 6 s cancel)", T, 64))
    R.append(ask("P2 the code prompt again", C, 24))
    X.append(cancelled("L2 ~50K-token prompt, cancelled at 16 s", L2, 16))
    R.append(ask("Q a new ~20K-token code prompt (after the 16 s cancel)", Q, 24))
    R.append(ask("T3 short text again", T, 64))
    R.append(ask("G 200 tokens of code, no logprobs", G, 200, logprobs=False))
    return out


def compare(a_path, b_path):
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    same_all = True
    for ra, rb in zip(a["requests"], b["requests"]):
        la, lb = ra.get("logprobs") or [], rb.get("logprobs") or []
        first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
        if first is None and len(la) != len(lb):
            first = min(len(la), len(lb))
        diff = max((abs(x[1] - y[1]) for x, y in zip(la, lb) if x[0] == y[0]), default=0.0)
        same = first is None and ra.get("text") == rb.get("text") and ra.get("reasoning") == rb.get("reasoning")
        same_all &= same
        print(f"{'SAME' if same else 'DIFF'}  {ra['label']}: {len(la)} logprob rows, text {'=' if ra.get('text') == rb.get('text') else '!='}, "
              f"cache_n {ra.get('cache_n')}/{rb.get('cache_n')}" + ("" if first is None else
              f", first differing row {first}, largest logprob difference {diff:.6f}"))
    print("all requests identical" if same_all else "NOT identical")
    return 0 if same_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=8107)
    ap.add_argument("--unit", default="strata-bitlamas")
    ap.add_argument("--log", default="")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        raise SystemExit(compare(*a.compare))
    cr.LOGS.extend(Path(x) for x in a.log.split(",") if x)
    cr.UNIT[0] = a.unit
    epoch = time.time()
    cr.log_mark()
    res = run(a.seed)
    res["journal"] = cr.journal(epoch)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
