"""pbench.py URL N_WORDS: one fresh prompt of N_WORDS random words (unique tag), 8 tokens out; prints prompt tok/s."""
import json, random, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2])
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".split()
r = random.Random(time.time()); doc = " ".join(r.choice(words) for _ in range(n))
body = {"model": "x", "max_tokens": 8, "temperature": 0, "reasoning_effort": "none",
        "messages": [{"role": "user", "content": f"[{time.time()}] Which word comes first?\n{doc}"}]}
t = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
    {"Content-Type": "application/json"}), timeout=3000).read())["timings"]
print(f"prompt {t['prompt_n']} tokens at {t['prompt_n'] / t['prompt_ms'] * 1000:.0f} tok/s")
