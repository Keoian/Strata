#!/usr/bin/env python3
"""The experimental speed projection (the control vector) loaded but off by default: three sessions.

  --mode exact    five greedy 8-token requests with logprobs (three short chats, an agent's first request of about
                  9,800 tokens with tools, its next turn).  --on sends "experimental_speed_projection": true with
                  each; without it nothing is sent, so the config's default decides.  Run on an engine without the
                  vector and on one with it loaded and the default off, both with --adapt-every 0: the replies'
                  logprobs must be the same.
  --mode live     agents X (vector off, the default) and Y (vector on) take turns: does each find its parked
                  conversation again?  Then X's conversation is asked once with the vector on and once more with
                  it off.  Then the same 400-token reply four times, off / on / off / on, for the decode speed.

    python p100-results/tools/projection_ab.py --mode exact --out stock.json
    python p100-results/tools/agent_cache_ab.py --table stock.json off.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_cache_ab as ab  # noqa: E402
import cancel_repro as cr  # noqa: E402
import split_borrow_ab as sb  # noqa: E402

ON = {"experimental_speed_projection": True}
OFF = {"experimental_speed_projection": False}
CHATS = ["Is it a good idea to water tomatoes at noon in summer?",
         "Explain in two sentences why the sky is blue.",
         "Write a Python function that returns the n-th Fibonacci number."]


def agent(rnd, name):
    return [{"role": "system", "content": f"You are agent {name}.\n" + ab.system_prompt()},
            {"role": "user", "content": "Find where Strata decides which conversation checkpoint to drop when the "
                                        "cache is full, and explain the rule."}]


def exact(seed, on):
    rnd, extra, results = random.Random(seed), (ON if on else None), []
    cr.wait_idle()
    for i, q in enumerate(CHATS):
        results.append(ab.ask(f"C{i + 1}: short chat", [{"role": "user", "content": q}], None, 8, extra))
    conv = agent(rnd, "E")
    results.append(ab.ask("G1: agent, first request", conv, ab.tools(), 8, extra))
    sb.turn(rnd, conv, 1, "e")
    results.append(ab.ask("G2: agent, next turn", conv, ab.tools(), 8, extra))
    results.append({"label": "end", "took": 0, "logprobs": []})     # the table leaves the last request out of the comparison
    return {"seed": seed, "codes": [], "requests": results}


def live(seed):
    rnd, tl, results = random.Random(seed), ab.tools(), []
    x, y = agent(rnd, "X"), agent(rnd, "Y")
    cr.wait_idle()

    def go(label, conv, extra, grow, tag, k):
        results.append(ab.ask(label, conv, tl, 4, extra))
        if grow:
            sb.turn(rnd, conv, k, tag)

    go("X1: agent X, vector off (default)", x, None, True, "x", 1)
    go("X2: agent X next turn, off", x, None, True, "x", 2)
    go("Y1: agent Y, vector ON", y, ON, True, "y", 1)
    go("Y2: agent Y next turn, ON", y, ON, True, "y", 2)
    go("X3: agent X returns, off", x, None, True, "x", 3)
    go("Y3: agent Y returns, ON", y, ON, True, "y", 3)
    go("X4: agent X returns, off (explicit false)", x, OFF, False, "x", 4)
    go("X5: agent X's same prompt with the vector ON", x, ON, False, "x", 5)
    go("X6: agent X's same prompt, off again", x, None, False, "x", 6)
    go("Y4: agent Y returns, ON", y, ON, False, "y", 4)
    q = [{"role": "user", "content": "Explain how a heat pump heats a house, in about 300 words."}]
    for i, (name, extra) in enumerate([("off", None), ("ON", ON), ("off", None), ("ON", ON)]):
        results.append(ab.ask(f"D{i + 1}: 400-token reply, vector {name}", q, None, 400, extra))
    results.append({"label": "end", "took": 0, "logprobs": []})
    return {"seed": seed, "codes": [], "requests": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("exact", "live"), required=True)
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=8107)
    a = ap.parse_args()
    out = exact(a.seed, a.on) if a.mode == "exact" else live(a.seed)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in out["requests"][:-1]:
        t = r.get("timings") or {}
        print(f"  {r['label']}: reused {t.get('cache_n')}, read {t.get('prompt_n')} at {t.get('prompt_per_second')} "
              f"tok/s, generated {t.get('predicted_n')} at {t.get('predicted_per_second')} tok/s")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
