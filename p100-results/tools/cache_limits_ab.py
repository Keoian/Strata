#!/usr/bin/env python3
"""Cache limits A/B: several agents with DIFFERENT system prompts take turns on one server.

  A1..A8 round 1   eight agents, each its own first line of the system prompt (so no shared head to borrow), the
                   same 8 tools, a task and one read call with its result: everything is read once
  A1..A8 round 2   each agent's next turn (about 2,400 new tokens): what does it read again?
  N1, N6           agents 1 and 6 are asked for the code planted in their first message

Every request but N is greedy with max_tokens 4 and logprobs, so its time is the prompt read.

    python p100-results/tools/cache_limits_ab.py --out base.json [--seed 8107] [--agents 8]
    python p100-results/tools/agent_cache_ab.py --table base.json big.json --engine-logs a.log b.log
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

TASKS = ["Find where Strata decides which conversation checkpoint to drop when the cache is full.",
         "List the tests that cover the layer split's checkpoints, with their files.",
         "Find every place that prints a line starting with 'strata serve: conversation cache'.",
         "Explain how the server picks the engine's context size from the config.",
         "Find where the engine reads STRATA_ environment switches and list them.",
         "Explain what --kv-resident does and where it is parsed.",
         "Find how the web page shows the tokens per second of a reply.",
         "List the places where a request can be cancelled, with their files."]


def run(seed, agents):
    rnd = random.Random(seed)
    tl = ab.tools()
    results, codes, convs = [], [], []
    for k in range(agents):
        code = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
        codes.append(code)
        convs.append([{"role": "system", "content": f"You are agent {k + 1} of {agents}, lane {code[:2]}{k}.\n"
                                                    + ab.system_prompt()},
                      {"role": "user", "content": f"Ticket {code}. {TASKS[k % len(TASKS)]}"}])
    out = {"seed": seed, "codes": codes, "requests": results}
    cr.wait_idle()
    for rnd_no in (1, 2):
        for k in range(agents):
            results.append(ab.ask(f"A{k + 1}.{rnd_no}: agent {k + 1}, round {rnd_no}", convs[k], tl))
            sb.turn(rnd, convs[k], rnd_no, f"a{k}r")
    for k in sorted({0, min(5, agents - 1)}):
        convs[k].append({"role": "user", "content": "Before you go on: what was the ticket number in my first "
                                                    "message? Reply with the code only."})
        n = ab.ask(f"N{k + 1}: agent {k + 1}'s planted code", convs[k], tl, max_tokens=3000)
        n["codes_found"] = [codes[k] in (n.get("text") or "")]
        n["codes_in_reasoning"] = [codes[k] in ((n.get("text") or "") + " " + (n.get("reasoning") or ""))]
        results.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=8107)
    ap.add_argument("--agents", type=int, default=8)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.seed, a.agents), indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
