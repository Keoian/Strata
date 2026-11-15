#!/usr/bin/env python3
"""Layer-split borrow A/B (upstream #5612): a main agent and sub-agents that share its system prompt and tools.

One server, one request at a time, as agents arrive in turn:

  M1..M12   the main agent: system prompt + 8 tools (the shared head), a task, 12 turns of reasoning + a read call +
            its result (about 2,400 tokens each)
  O1        an unrelated chat (another system prompt): the main conversation is parked
  S1, S2    a sub-agent: the SAME system prompt and tools, its own task and two turns.  Its deepest match in the
            cache is the main conversation's root checkpoint (the end of the system prompt)
  M13       the main agent's next turn: what does it read again?
  O2, T1, T2, M14   the same once more with a second sub-agent
  S3        the first sub-agent's next turn
  N         the main agent is asked for the two codes planted in its first message and its fifth tool result

Every request but N is greedy with max_tokens 4 and logprobs, so its time is the prompt read.

    python p100-results/tools/split_borrow_ab.py --out stock.json [--seed 8107]
    python p100-results/tools/agent_cache_ab.py --table stock.json borrow.json --engine-logs a.log b.log
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_cache_ab as ab  # noqa: E402
import cancel_repro as cr  # noqa: E402

MAIN_TURNS = 12


def turn(rnd, msgs, t, tag, plant=None):
    path, offset, limit, lines = ab.source_slice(rnd)
    if plant:
        k = rnd.randrange(len(lines) // 3, 2 * len(lines) // 3)
        lines[k] = lines[k].split("\t")[0] + f"\t// build tag: {plant}"
    a = ab.assistant(rnd, t, path, offset, limit)
    a["tool_calls"][0]["id"] = f"call_{tag}{t:02d}"
    msgs.append(a)
    msgs.append({"role": "tool", "tool_call_id": f"call_{tag}{t:02d}", "content": "\n".join(lines)})


def run(seed):
    rnd = random.Random(seed)
    code_u = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    code_f = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    tl, head = ab.tools(), {"role": "system", "content": ab.system_prompt()}
    results = []
    out = {"seed": seed, "codes": [code_u, code_f], "requests": results}
    main = [head, {"role": "user", "content": f"Ticket {code_u}. Find where Strata decides which conversation "
                                              "checkpoint to drop when the cache is full, and explain the rule."}]
    other = [{"role": "system", "content": "You are a home assistant. Answer in one short sentence."},
             {"role": "user", "content": "Is it a good idea to water tomatoes at noon in summer?"}]
    subs = [[copy.deepcopy(head), {"role": "user", "content": "Sub-task: list the tests that cover the layer split's "
                                                              "checkpoints, with their files."}],
            [copy.deepcopy(head), {"role": "user", "content": "Sub-task: find every place that prints a line "
                                                              "starting with 'strata serve: conversation cache'."}]]
    cr.wait_idle()
    t = 0

    def main_turn(label_extra=""):
        nonlocal t
        t += 1
        results.append(ab.ask(f"M{t}: main agent{label_extra}", main, tl))
        turn(rnd, main, t, "m", plant=code_f if t == 5 else None)

    for _ in range(MAIN_TURNS):
        main_turn()
    results.append(ab.ask("O1: unrelated chat", other, None))
    for k in (1, 2):
        results.append(ab.ask(f"S{k}: sub-agent 1 (the main agent's system prompt and tools)", subs[0], tl))
        turn(rnd, subs[0], k, "s")
    main_turn(" returns after sub-agent 1")
    other += [{"role": "assistant", "content": "Better in the morning."},
              {"role": "user", "content": "And cucumbers?"}]
    results.append(ab.ask("O2: unrelated chat again", other, None))
    for k in (1, 2):
        results.append(ab.ask(f"T{k}: sub-agent 2", subs[1], tl))
        turn(rnd, subs[1], k, "t")
    main_turn(" returns after sub-agent 2")
    results.append(ab.ask("S3: sub-agent 1 returns", subs[0], tl))
    main.append({"role": "user", "content": "Before you go on: what was the ticket number in my first message, and "
                                            "what build tag did the file in your fifth tool call show? Reply with "
                                            "both codes only."})
    n = ab.ask("N: the main agent's two planted codes", main, tl, max_tokens=3000)
    blob = (n.get("text") or "") + " " + (n.get("reasoning") or "")
    n["codes_found"] = [code_u in (n.get("text") or ""), code_f in (n.get("text") or "")]
    n["codes_in_reasoning"] = [code_u in blob, code_f in blob]
    results.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=8107)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.seed), indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
