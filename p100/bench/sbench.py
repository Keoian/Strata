"""sbench.py URL LABEL [ROUNDS] [KEY]: dbench.py with the server's default sampling (no temperature sent; thinking off: 0.7 / 0.8 / 20, presence 1.5) - a sampled decode benchmark for a Strata (or llama.cpp) server.
Six fixed prompts (story, code, explanation, math, list, advice), sampled, 256 new tokens, thinking off; then one fresh
~4K-token prompt (unique tag, no cache help) for prompt speed. Reports token-weighted decode tok/s per round from the
engine's own timings, draft acceptance, and appends one line to /work/flashnext/bench/RESULTS.md."""
import json, sys, time, urllib.request, random
url, label = sys.argv[1], sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
key = sys.argv[4] if len(sys.argv) > 4 else None
H = {"Content-Type": "application/json"}
if key: H["Authorization"] = "Bearer " + key
P = ["Write a short story (about 200 words) about a lighthouse keeper who finds a message in a bottle.",
     "Write a Python function that parses a CSV file of transactions (date, amount, category) and returns the total per category per month. Include type hints and a short docstring.",
     "Explain how a refrigerator moves heat from inside to outside, step by step, for a curious teenager.",
     "Solve step by step: a train leaves at 9:40 at 72 km/h; a second train leaves the same station at 10:15 at 90 km/h on the same track. When and where does the second catch the first?",
     "List 12 practical tips for keeping a small vegetable garden healthy through a hot, dry summer, one sentence each.",
     "My friend keeps cancelling plans at the last minute. How can I bring it up kindly without starting a fight? Give me a few concrete things I could say."]
def req(body, timeout=900):
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), H)
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
def chat(p, n=256):
    return req({"model": "x", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                "reasoning_effort": "none"})
chat("Say hi.", 8)
res = []
for rd in range(rounds):
    tok = ms = acc = drf = 0
    for p in P:
        t = chat(p)["timings"]
        tok += t["predicted_n"]; ms += t["predicted_ms"]; acc += t.get("draft_n_accepted", 0); drf += t.get("draft_n", 0)
    res.append((tok / ms * 1000, acc / max(drf, 1), tok))
    print(f"round {rd+1}: decode {tok / ms * 1000:.1f} tok/s over {tok} tokens, drafts accepted {acc}/{drf} ({acc/max(drf,1):.1%})", flush=True)
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".split()
rnd = random.Random(time.time())
doc = " ".join(rnd.choice(words) for _ in range(3000))
t = req({"model": "x", "messages": [{"role": "user", "content": f"[{time.time()}] Count roughly how often 'echo' appears:\n{doc}"}],
         "max_tokens": 16, "temperature": 0, "reasoning_effort": "none"})["timings"]
pp = t["prompt_n"] / t["prompt_ms"] * 1000
dec = sorted(r[0] for r in res)
line = (f"| {time.strftime('%m-%d %H:%M')} | {label} (sampled) | {' / '.join(f'{r[0]:.1f}' for r in res)} | "
        f"{sum(r[1] for r in res)/len(res):.1%} | {t['prompt_n']} tok @ {pp:.0f} tok/s |")
print(line)
open("/work/flashnext/bench/RESULTS.md", "a").write(line + "\n")
