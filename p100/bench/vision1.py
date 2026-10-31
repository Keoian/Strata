import base64, json, sys, time, urllib.request
url, img = sys.argv[1], sys.argv[2]; key = sys.argv[3] if len(sys.argv) > 3 else None
H = {"Content-Type": "application/json"}
if key: H["Authorization"] = "Bearer " + key
b64 = base64.b64encode(open(img, "rb").read()).decode()
body = {"model": "qwen3.8-27b", "max_tokens": 150, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image in two sentences."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
t = time.time()
r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), H), timeout=600).read())
print(r["choices"][0]["message"]["content"]); print("timings:", json.dumps(r.get("timings")), "wall %.1fs" % (time.time() - t))
