#!/usr/bin/env python3
"""--prompt-cache-spread A/B: one scripted coding-agent session on a fresh engine (p100-results/prompt-cache-spread/).

The conversation is sent the way a coding harness sends it: a system prompt with tool definitions, user tasks, and
turns of assistant reasoning + a tool call + the tool's result (slices of this repo's source files, with line numbers
as a read tool returns them).  Every request is greedy with max_tokens 4 and logprobs (top 5), so its time is the
prompt read.  Between turns the harness goes back, as the served engine's log showed (engine start 25: a 24,915-token
prompt sent again after the session had reached 35,542, resumed at the system prompt's 3,416):

  J4   after turn 16: the prompt of turn 12 again (4 turns back), then the session goes on from its tip
  J6   after turn 22: the prompt of turn 16 again
  J10  after turn 30: the prompt of turn 20 again
  E9   turn 34: the tool result of turn 25 is cleared ("[Old tool result content cleared]"), as OpenCode's prune and
       Claude Code's clearing of old tool results do, and it stays cleared
  N    the end: two codes planted in the first user message and in turn 5's tool result, asked for (answer check)

    python p100-results/tools/agent_cache_ab.py --out base.json [--seed 8107]
    python p100-results/tools/agent_cache_ab.py --table base.json spread.json [base2.json] [--engine-logs a.log b.log ..]
"""
import argparse
import copy
import http.client
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cancel_repro as cr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GREEDY = {"temperature": 0, "presence_penalty": 0}
TURNS = 36
TASK_AT = {13: "Now make the prompt cache log say which checkpoint a request resumed from, and why.",
           25: "Last one: check how a cancelled request's checkpoints are kept, and list anything that looks wrong."}
JUMPS = {16: 12, 22: 16, 30: 20}          # after turn k's request: turn v's prompt again
CLEAR_AT, CLEARED = 34, 25                 # turn 34's request has turn 25's tool result cleared
CLEARED_TEXT = "[Old tool result content cleared]"
SOURCES = ["src/program/generate.cpp", "serve/server.py", "src/prefill/prefill.cpp",
           "include/strata/core/expert_source.hpp", "src/core/expert_source.cpp", "include/strata/program/conv_cache.hpp",
           "src/core/conversation_snapshot.cpp", "src/core/session.cpp"]
TOPICS = ["the conversation checkpoints", "the prompt read's chunking", "how STOP reaches the stages",
          "the expert cache's adaptation", "how the server renders tool calls", "the KV window per layer",
          "where the request's timings come from", "the parking of conversations"]


def tools():
    """Eight function tools with descriptions the length a coding harness sends (their usage notes from the docs)."""
    notes = (REPO / "docs/TERMINAL.md").read_text(errors="replace")
    paras = [p.strip() for p in notes.split("\n\n") if len(p.strip()) > 200]
    spec = [("read", "Read a file from the local filesystem. Returns lines with their numbers.",
             {"filePath": "string", "offset": "integer", "limit": "integer"}),
            ("edit", "Replace an exact string in a file. The old string must occur once.",
             {"filePath": "string", "oldString": "string", "newString": "string"}),
            ("write", "Write a file, replacing it if it exists.", {"filePath": "string", "content": "string"}),
            ("bash", "Run a shell command in the repository and return its output.",
             {"command": "string", "timeout": "integer", "description": "string"}),
            ("grep", "Search file contents with a regular expression.", {"pattern": "string", "path": "string"}),
            ("glob", "Find files by name pattern.", {"pattern": "string", "path": "string"}),
            ("list", "List a directory.", {"path": "string"}),
            ("todowrite", "Keep the task list for this session.", {"todos": "string"})]
    out = []
    for i, (name, desc, props) in enumerate(spec):
        usage = "\n\n".join(paras[i * 3:i * 3 + 3])
        out.append({"type": "function", "function": {
            "name": name, "description": desc + "\n\nUsage notes:\n" + usage,
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
                           "required": list(props)[:1]}}})
    return out


def system_prompt():
    return ("You are a coding agent working in the Strata repository on the user's machine. Use the tools to read "
            "and change files; keep answers short.\n\n# AGENTS.md\n\n" + (REPO / "AGENTS.md").read_text() +
            "\n\n# CONTRIBUTING.md\n\n" + (REPO / "CONTRIBUTING.md").read_text())


def source_slice(rnd, chars=7000):
    path = rnd.choice(SOURCES)
    lines = (REPO / path).read_text(errors="replace").splitlines()
    start = rnd.randrange(0, max(1, len(lines) - 400))
    out, n, i = [], 0, start
    while i < len(lines) and n < chars:
        s = f"{i + 1:6d}\t{lines[i]}"
        out.append(s)
        n += len(s) + 1
        i += 1
    return path, start + 1, i - start, out


def assistant(rnd, t, path, offset, limit):
    topic = rnd.choice(TOPICS)
    words = " ".join(rnd.choice(cr.WORDS) for _ in range(rnd.randrange(30, 70)))
    reasoning = (f"Turn {t}. To see {topic} I should read {path} from line {offset}. The last result did not show "
                 f"the whole function, so the next part should. Notes so far: {words}.")
    return {"role": "assistant", "content": "", "reasoning_content": reasoning,
            "tool_calls": [{"id": f"call_{t:02d}", "type": "function", "function": {
                "name": "read", "arguments": json.dumps({"filePath": path, "offset": offset, "limit": limit})}}]}


def ask(label, messages, tools_, max_tokens=4, extra=None):
    body = {"model": "strata", "messages": messages, "max_tokens": max_tokens, "stream": False,
            "logprobs": True, "top_logprobs": 5, **GREEDY}
    if tools_:
        body["tools"] = tools_
    body.update(extra or {})      # per-request server fields ("experimental_speed_projection")
    t0 = time.monotonic()
    conn = http.client.HTTPConnection(cr.HOST, cr.PORT, timeout=1800)
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Authorization": "Bearer " + cr.KEY, "Content-Type": "application/json"})
    j = json.loads(conn.getresponse().read() or b"{}")
    conn.close()
    took = round(time.monotonic() - t0, 2)
    if "error" in j:
        print(f"    {label}: ERROR {j['error']}", flush=True)
        return {"label": label, "error": j["error"], "took": took}
    c = j["choices"][0]
    lp = c.get("logprobs") or {}
    t = j.get("timings") or {}
    r = {"label": label, "took": took, "text": c["message"].get("content"),
         "reasoning": c["message"].get("reasoning_content"), "tool_calls": c["message"].get("tool_calls"),
         "finish": c.get("finish_reason"), "usage": j.get("usage"), "cache_n": t.get("cache_n"),
         "prompt_n": t.get("prompt_n"), "timings": t,
         "logprobs": [[e["token"], e["logprob"], [[a["token"], a["logprob"]] for a in e.get("top_logprobs") or []]]
                      for e in (lp.get("reasoning_content") or []) + (lp.get("content") or [])]}
    print(f"    {label}: took {took} s, cache_n {r['cache_n']}, prompt_n {r['prompt_n']}", flush=True)
    return r


def run(seed):
    rnd = random.Random(seed)
    code_u = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    code_f = "".join(rnd.choice(cr.ALPHABET) for _ in range(6))
    tl = tools()
    msgs = [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": f"Ticket {code_u}. Find where Strata decides which conversation checkpoint "
                                        "to drop when the cache is full, and explain the rule."}]
    snap, results, tool_at = {}, [], {}
    out = {"seed": seed, "codes": [code_u, code_f], "requests": results}
    cr.wait_idle()
    for t in range(1, TURNS + 1):
        if t in TASK_AT:
            msgs.append({"role": "assistant", "content": "Done. " + " ".join(rnd.choice(cr.WORDS) for _ in range(40)),
                         "reasoning_content": "The task is finished; I will summarise what I found."})
            msgs.append({"role": "user", "content": TASK_AT[t]})
        label = f"turn {t}"
        if t == CLEAR_AT:
            msgs[tool_at[CLEARED]]["content"] = CLEARED_TEXT
            label += f" (E9: turn {CLEARED}'s tool result cleared)"
        snap[t] = copy.deepcopy(msgs)
        results.append(ask(label, msgs, tl))
        if t in JUMPS:
            v = JUMPS[t]
            results.append(ask(f"J{t - v}: turn {v}'s prompt again", snap[v], tl))
        path, offset, limit, lines = source_slice(rnd)
        if t == 5:
            k = rnd.randrange(len(lines) // 3, 2 * len(lines) // 3)
            lines[k] = lines[k].split("\t")[0] + f"\t// build tag: {code_f}"
        msgs.append(assistant(rnd, t, path, offset, limit))
        msgs.append({"role": "tool", "tool_call_id": f"call_{t:02d}", "content": "\n".join(lines)})
        tool_at[t] = len(msgs) - 1
    msgs.append({"role": "user", "content": "Before you go on: what was the ticket number in my first message, and "
                                            "what build tag did the file in your fifth tool call show? Reply with "
                                            "both codes only."})
    n = ask("N: the two planted codes", msgs, tl, max_tokens=3000)
    blob = (n.get("text") or "") + " " + (n.get("reasoning") or "")
    n["codes_found"] = [code_u in (n.get("text") or ""), code_f in (n.get("text") or "")]
    n["codes_in_reasoning"] = [code_u in blob, code_f in blob]
    results.append(n)
    return out


PROMPT_LINE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")


def engine_reads(log):
    """The engine log's prompt lines in order: (tokens, reused, read, ms)."""
    return [tuple(map(int, m.groups())) for m in PROMPT_LINE.finditer(Path(log).read_text(errors="replace"))]


def table(paths, logs):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    names = [Path(p).stem for p in paths]
    reads = [engine_reads(lg) for lg in logs] if logs else [None] * len(runs)
    for nm, r, e in zip(names, runs, reads):
        if e is not None and len(e) != len(r["requests"]):
            print(f"!! {nm}: {len(e)} engine prompt lines for {len(r['requests'])} requests")
    hdr = "| request | prompt tokens | " + " | ".join(f"{n}: reused / read, s" for n in names) + " |"
    print(hdr)
    print("|" + "---|" * (2 + len(names)))
    tot = [[0, 0.0] for _ in names]
    for i, q in enumerate(runs[0]["requests"]):
        cells = []
        size = None
        for k, (r, e) in enumerate(zip(runs, reads)):
            x = r["requests"][i]
            if e is not None:
                size, reused, fresh, ms = e[i]
                sec = ms / 1000
            else:
                reused, fresh = x.get("cache_n") or 0, x.get("prompt_n") or 0
                size, sec = reused + fresh, x["took"]
            tot[k][0] += fresh
            tot[k][1] += sec
            cells.append(f"{reused:,} / {fresh:,}, {sec:.1f}")
        print(f"| {q['label']} | {size:,} | " + " | ".join(cells) + " |")
    print("| **all requests** | | " + " | ".join(f"**{t[0]:,} read, {t[1]:.0f} s**" for t in tot) + " |")
    print()
    for nm, r in zip(names, runs):
        n = r["requests"][-1]
        print(f"{nm}: codes {r['codes']} in the answer {n.get('codes_found')}, answer {(n.get('text') or '')[:60]!r}")
    a = runs[0]
    for nm, b in zip(names[1:], runs[1:]):
        worst, first_diff = 0.0, []
        for ra, rb in zip(a["requests"], b["requests"]):
            la, lb = ra.get("logprobs") or [], rb.get("logprobs") or []
            if ra is a["requests"][-1]:
                continue      # the long answer: decode is not repeatable run to run (STOP.md), so prompt reads only
            d = max((abs(x[1] - y[1]) for x, y in zip(la, lb) if x[0] == y[0]), default=0.0)
            if [x[0] for x in la] != [y[0] for y in lb]:
                first_diff.append(ra["label"])
            worst = max(worst, d)
        print(f"{names[0]} vs {nm}: largest logprob difference over the 4-token replies {worst:.6f}; "
              f"replies with other tokens: {first_diff or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=8107)
    ap.add_argument("--table", nargs="+")
    ap.add_argument("--engine-logs", nargs="*", default=[])
    a = ap.parse_args()
    if a.table:
        return table(a.table, a.engine_logs)
    res = run(a.seed)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
