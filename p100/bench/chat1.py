import json, sys, time, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
prompt = sys.argv[2] if len(sys.argv) > 2 else "In three sentences, explain why the sky is blue."
effort = sys.argv[3] if len(sys.argv) > 3 else "none"
body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0,
        "reasoning_effort": effort}
t = time.time()
r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                      {"Content-Type": "application/json"}), timeout=600).read())
m = r["choices"][0]["message"]
print(m.get("content")); print("reasoning:", (m.get("reasoning_content") or "")[:200])
print("timings:", json.dumps(r.get("timings")), "wall %.1fs" % (time.time() - t))
