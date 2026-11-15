#!/usr/bin/env python3
"""Cancellation repros for pictures on a running Strata server (p100-results/vision/CANCEL.md).

Each case uses fresh pictures (a random 6-character code on a noisy background, so no earlier encode is cached) and
cancels a request the way a browser's Stop does: the socket is shut down mid-request.  It records when every request was
sent and answered, whether the answer reads the picture's code, the journal lines of the case and /v1/status after it.

    python p100-results/tools/cancel_repro.py --out p100-results/vision/cancel-baseline.json [--cases s1,s2,...]
"""
import argparse
import base64
import http.client
import io
import json
import os
import random
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HOST, PORT = "127.0.0.1", 8080
ENV = Path.home() / ".config/strata/bitlamas.env"
QUESTION = "What is the code written in the white box of the picture? Reply with the code only."
ALPHABET = "ACDEFHJKLMNPRTUVWXY34679"            # no look-alikes (0/O, 1/I, 5/S, 8/B, 2/Z)
BIG, MEDIUM = (2560, 1440), (1600, 900)          # 3600 image tokens (the cap, ~80 s on 16 threads), ~1400


def api_key() -> str:
    for line in ENV.read_text().splitlines():
        if line.startswith("STRATA_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no STRATA_API_KEY in " + str(ENV))


KEY = api_key()
T0 = time.monotonic()


def now() -> float:
    return round(time.monotonic() - T0, 2)


def picture(size, rnd: random.Random):
    """(code, data URL): a noisy picture with the code in a white box in the middle."""
    w, h = size
    im = Image.new("RGB", (w, h), tuple(rnd.randrange(60, 200) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(400):
        x, y = rnd.randrange(w), rnd.randrange(h)
        s = rnd.randrange(10, 90)
        d.rectangle([x, y, x + s, y + s], fill=tuple(rnd.randrange(256) for _ in range(3)))
    code = "".join(rnd.choice(ALPHABET) for _ in range(6))
    font = ImageFont.load_default(size=h // 7)
    box = d.textbbox((0, 0), code, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x0, y0 = (w - tw) // 2, (h - th) // 2
    d.rectangle([x0 - 40, y0 - 40, x0 + tw + 40, y0 + th + 60], fill=(255, 255, 255))
    d.text((x0 - box[0], y0 - box[1]), code, font=font, fill=(0, 0, 0))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return code, "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def user_turn(text, *urls):
    if not urls:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [{"type": "text", "text": text}] +
            [{"type": "image_url", "image_url": {"url": u}} for u in urls]}


def follow_up(url, first=QUESTION, second=QUESTION):
    """What the web page sends after a Stop: the picture's turn, the stopped (empty) answer, the new question."""
    return [user_turn(first, url), {"role": "assistant", "content": ""}, user_turn(second)]


class Req(threading.Thread):
    """One request in its own thread; cancel_after shuts its socket down that many seconds after it was sent."""

    def __init__(self, label, messages, cancel_after=None, stream=True, max_tokens=24, anthropic=False):
        super().__init__(daemon=True)
        self.label, self.messages, self.cancel_after = label, messages, cancel_after
        self.stream, self.max_tokens, self.anthropic = stream, max_tokens, anthropic
        self.r = {"label": label, "sent": None, "cancel_after": cancel_after}

    def run(self):
        r = self.r
        r["sent"] = now()
        t0 = time.monotonic()
        conn = http.client.HTTPConnection(HOST, PORT, timeout=900)
        if self.anthropic:
            path = "/v1/messages"
            body = {"model": "strata", "max_tokens": self.max_tokens, "messages": self.messages, "stream": self.stream,
                    "temperature": 0, "thinking": {"type": "disabled"}}
            headers = {"x-api-key": KEY, "content-type": "application/json", "anthropic-version": "2023-06-01"}
        else:
            path = "/v1/chat/completions"
            body = {"model": "strata", "messages": self.messages, "stream": self.stream, "max_tokens": self.max_tokens,
                    "temperature": 0, "reasoning_effort": "none"}
            headers = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
        timer = None
        try:
            conn.connect()
            sock = conn.sock                    # http.client drops conn.sock once the response headers are in
            if self.cancel_after is not None:
                def cut():
                    r["cancelled_at"] = now()
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                timer = threading.Timer(self.cancel_after, cut)
                timer.start()
            conn.request("POST", path, json.dumps(body), headers)
            resp = conn.getresponse()
            r["status"] = resp.status
            text = []
            if self.stream:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    try:
                        j = json.loads(data)
                    except ValueError:
                        continue
                    piece = "".join((c.get("delta") or {}).get("content") or "" for c in j.get("choices") or []) or \
                        ((j.get("delta") or {}).get("text") or "" if j.get("type") == "content_block_delta" else "")
                    if "first_token" not in r and piece:
                        r["first_token"] = round(time.monotonic() - t0, 2)
                    if j.get("error"):
                        r["error"] = j["error"]
                    for c in j.get("choices") or []:
                        text.append((c.get("delta") or {}).get("content") or "")
                    if j.get("type") == "content_block_delta":
                        text.append((j.get("delta") or {}).get("text") or "")
            else:
                j = json.loads(resp.read() or b"{}")
                if j.get("error"):
                    r["error"] = j["error"]
                if j.get("choices"):
                    text.append(j["choices"][0]["message"].get("content") or "")
                elif j.get("content"):
                    text.append("".join(b.get("text", "") for b in j["content"]))
            r["text"] = "".join(text)
        except Exception as e:  # noqa: BLE001 - a cancelled request ends in an error here
            r["exception"] = f"{type(e).__name__}: {e}"
        finally:
            if timer is not None:
                timer.cancel()
            conn.close()
            r["took"] = round(time.monotonic() - t0, 2)
            r["answered"] = now()


def reads(r, code) -> bool | None:
    if "text" not in r:
        return None
    return code in re.sub(r"[^A-Z0-9]", "", r["text"].upper())


def status() -> dict:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
    conn.request("GET", "/v1/status", headers={"Authorization": "Bearer " + KEY})
    j = json.loads(conn.getresponse().read())
    conn.close()
    return {"in_flight": (j.get("activity") or {}).get("in_flight"), "requests": (j.get("activity") or {}).get("requests"),
            "vision": j.get("vision")}


LOGS: list[Path] = []                   # --log: a server run by hand writes here instead of the service's journal
LOG_AT: dict = {}


def log_mark():
    for f in LOGS:
        LOG_AT[f] = f.stat().st_size if f.exists() else 0


UNIT = ["strata-bitlamas"]


def journal(since_epoch: float) -> list[str]:
    if LOGS:
        keep = ("[strata]", "strata serve: prompt", "strata prefill: stopped", "cancel", "ERR", "rror")
        lines = subprocess.run(["journalctl", "-u", UNIT[0], "--since", f"@{since_epoch:.0f}", "-o", "cat", "--no-pager"],
                               capture_output=True, text=True).stdout.splitlines()
        lines = [ln for ln in lines if any(k in ln for k in keep)]
        for f in LOGS:
            with open(f, "rb") as fh:
                fh.seek(LOG_AT.get(f, 0))
                lines += [ln for ln in fh.read().decode(errors="replace").splitlines() if any(k in ln for k in keep)]
        return lines[-60:]
    out = subprocess.run(["journalctl", "-u", UNIT[0], "--since", f"@{since_epoch:.0f}", "-o", "cat", "--no-pager"],
                         capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()][-40:]


def encoder_cpu() -> float:
    """CPU seconds used so far by the picture reader (strata-vision) processes."""
    total = 0.0
    for pid in subprocess.run(["pgrep", "-f", "bin/strata-vision"], capture_output=True, text=True).stdout.split():
        try:
            f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            pass
    return total


def wait_idle(limit=400):
    """Until the server has no request in flight and the picture reader is idle (between cases)."""
    end = time.monotonic() + limit
    while time.monotonic() < end:
        c0 = encoder_cpu()
        time.sleep(1)
        if not status().get("in_flight") and encoder_cpu() - c0 < 0.3:
            return


def start(*reqs, gaps=()):
    """Start the requests, the i-th after gaps[i-1] seconds; wait for all of them."""
    for i, q in enumerate(reqs):
        if i and gaps:
            time.sleep(gaps[i - 1])
        q.start()
    for q in reqs:
        q.join()
    return [q.r for q in reqs]


# ------------------------------------------------------------------------------------------------ the cases
def s1(rnd):
    """Cancel during a big picture's encode; a text request 0.5 s later; then the follow-up about the picture."""
    code, url = picture(BIG, rnd)
    a, b = Req("A picture, cancelled at 5 s", [user_turn(QUESTION, url)], cancel_after=5), Req("B text", [user_turn("Say OK.")])
    out = start(a, b, gaps=(5.5,))
    c = start(Req("C follow-up about A's picture", follow_up(url)))
    out += c
    return {"code": code, "requests": out, "C_reads_code": reads(c[0], code)}


def s2(rnd):
    """Cancel during a big picture's encode; the follow-up about it 5 s later (while the cancelled encode runs)."""
    code, url = picture(BIG, rnd)
    out = start(Req("A picture, cancelled at 5 s", [user_turn(QUESTION, url)], cancel_after=5),
                Req("B follow-up about A's picture", follow_up(url)), gaps=(10,))
    return {"code": code, "requests": out, "B_reads_code": reads(out[1], code)}


def s3(rnd):
    """Stop, resend, stop again: A cancelled at 3 s, B (A's picture again) sent at 4 s and cancelled 4 s later, C text at 9 s."""
    code, url = picture(BIG, rnd)
    out = start(Req("A picture, cancelled at 3 s", [user_turn(QUESTION, url)], cancel_after=3),
                Req("B A's picture again, cancelled after 4 s", follow_up(url), cancel_after=4),
                Req("C text", [user_turn("Say OK.")]), gaps=(4, 5))
    return {"code": code, "requests": out}


def s4(rnd):
    """A text request queued behind a medium picture's encode is cancelled; the picture's request must be unaffected."""
    code, url = picture(MEDIUM, rnd)
    out = start(Req("A picture (not cancelled)", [user_turn(QUESTION, url)]),
                Req("B text, cancelled at 3 s while queued", [user_turn("Say OK.")], cancel_after=3),
                Req("C text", [user_turn("Say OK.")]), gaps=(2, 3))
    return {"code": code, "requests": out, "A_reads_code": reads(out[0], code)}


def s5(rnd):
    """Cancel while the engine reads a picture prompt (the encode is cached), then text, then the same request in full."""
    code, url = picture(MEDIUM, rnd)
    warm = start(Req("W the picture, in full (encodes it)", [user_turn(QUESTION, url)]))
    q = "Look carefully at this picture. " + QUESTION
    out = start(Req("A the picture, new question, cancelled 1.5 s in (reading the prompt)", [user_turn(q, url)], cancel_after=1.5),
                Req("B text", [user_turn("Say OK.")]), gaps=(2,))
    again = start(Req("C A's request in full", [user_turn(q, url)]))
    return {"code": code, "requests": warm + out + again, "W_reads_code": reads(warm[0], code),
            "C_reads_code": reads(again[0], code)}


def s6(rnd):
    """Cancel while a picture's answer is written, then the same request in full."""
    code, url = picture(MEDIUM, rnd)
    long_q = "Describe this picture in detail: its colors, its shapes, and the code in the white box."
    first = start(Req("A long answer, cancelled at 32 s (encode ~17 s + reading ~9 s)", [user_turn(long_q, url)], cancel_after=32,
                      max_tokens=600))
    again = start(Req("B the code question about the same picture", [user_turn(QUESTION, url)]),
                  Req("C A's request in full", [user_turn(long_q, url)], max_tokens=600))
    return {"code": code, "requests": first + again, "B_reads_code": reads(again[0], code),
            "C_reads_code": reads(again[1], code)}


def s7(rnd):
    """Two medium pictures in one request, cancelled during the first; a text request 0.5 s later."""
    c1, u1 = picture(MEDIUM, rnd)
    c2, u2 = picture(MEDIUM, rnd)
    out = start(Req("A two pictures, cancelled at 4 s", [user_turn("What are the two codes?", u1, u2)], cancel_after=4),
                Req("B text", [user_turn("Say OK.")]), gaps=(4.5,))
    return {"codes": [c1, c2], "requests": out}


def s8(rnd):
    """Not streamed (stream false) and cancelled during a medium picture's encode; a text request 0.5 s later."""
    code, url = picture(MEDIUM, rnd)
    out = start(Req("A picture, not streamed, cancelled at 4 s", [user_turn(QUESTION, url)], cancel_after=4, stream=False),
                Req("B text", [user_turn("Say OK.")]), gaps=(4.5,))
    return {"code": code, "requests": out}


def s9(rnd):
    """The Anthropic endpoint (/v1/messages), cancelled during a medium picture's encode; a text request 0.5 s later."""
    code, url = picture(MEDIUM, rnd)
    b64 = url.split(",", 1)[1]
    msg = [{"role": "user", "content": [{"type": "text", "text": QUESTION},
                                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]}]
    out = start(Req("A picture via /v1/messages, cancelled at 4 s", msg, cancel_after=4, anthropic=True),
                Req("B text", [user_turn("Say OK.")]), gaps=(4.5,))
    return {"code": code, "requests": out}


def s10(rnd):
    """A long text prompt (~24K tokens, nothing cached) cancelled 6 s into its read; a text request 0.5 s later."""
    words = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
             "lagoon", "meadow", "nectar", "obsidian", "prairie", "quartz", "rivulet", "sierra", "tundra"]
    lines = [f"Line {i}: " + " ".join(rnd.choice(words) for _ in range(12)) + "." for i in range(1700)]
    long_prompt = "\n".join(lines) + "\n\nHow many lines mention quartz? Reply with a number."
    out = start(Req("A long text prompt, cancelled at 6 s", [user_turn(long_prompt)], cancel_after=6),
                Req("B text", [user_turn("Say OK.")]), gaps=(6.5,))
    return {"requests": out}


def s11(rnd):
    """No cancel (a reference for s6): the same picture request three times; the 2nd and 3rd reuse the 1st's prompt."""
    code, url = picture(MEDIUM, rnd)
    long_q = "Describe this picture in detail: its colors, its shapes, and the code in the white box."
    out = start(Req("R1 fresh", [user_turn(long_q, url)], max_tokens=200))
    out += start(Req("R2 the same request", [user_turn(long_q, url)], max_tokens=200))
    out += start(Req("R3 the same request", [user_turn(long_q, url)], max_tokens=200))
    texts = [r.get("text") for r in out]
    return {"code": code, "requests": out, "R1_eq_R2": texts[0] == texts[1], "R2_eq_R3": texts[1] == texts[2],
            "common_prefix_chars": [len(os.path.commonprefix([texts[0], t or ""])) for t in texts[1:]]}


WORDS = ["amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lagoon", "meadow", "nectar", "obsidian", "prairie", "quartz", "rivulet", "sierra", "tundra"]


def s12(rnd):
    """A long text prompt (~50K tokens, nothing cached) cancelled 16 s into its read, when on a two-card split the first
    card is in its second chunk and the second card in its first; a text request 0.5 s later."""
    lines = [f"Line {i}: " + " ".join(rnd.choice(WORDS) for _ in range(12)) + "." for i in range(1700)]
    long_prompt = "\n".join(lines) + "\n\nHow many lines mention quartz? Reply with a number."
    out = start(Req("A long text prompt, cancelled at 16 s", [user_turn(long_prompt)], cancel_after=16),
                Req("B text", [user_turn("Say OK.")]), gaps=(16.5,))
    return {"requests": out}


def s13(rnd):
    """The engine after a prompt read stopped in the middle: a ~29K-token prompt with a code on one line is cancelled 10 s
    in, then sent again in full (nothing of it was kept) and must give the code; then a short text request."""
    lines = [f"Line {i}: " + " ".join(rnd.choice(WORDS) for _ in range(12)) + "." for i in range(1000)]
    code = "".join(rnd.choice(ALPHABET) for _ in range(6))
    at = rnd.randrange(600, 900)
    lines[at] = f"Line {at}: the vault code is {code}."
    prompt = "\n".join(lines) + "\n\nWhat is the vault code? Reply with the code only."
    out = start(Req("A the code prompt, cancelled at 10 s", [user_turn(prompt)], cancel_after=10))
    out += start(Req("W the same prompt in full", [user_turn(prompt)]))
    out += start(Req("B text", [user_turn("Say OK.")]))
    return {"code": code, "line": at, "requests": out, "W_reads_code": code in (out[1].get("text") or "")}


CASES = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9, "s10": s10, "s11": s11, "s12": s12, "s13": s13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--unit", default="strata-bitlamas", help="the systemd unit whose journal has the server's lines")
    ap.add_argument("--log", default="", help="comma-separated log files of a server run by hand (instead of the journal)")
    a = ap.parse_args()
    LOGS.extend(Path(x) for x in a.log.split(",") if x)
    UNIT[0] = a.unit
    seed = a.seed if a.seed is not None else int(time.time())
    rnd = random.Random(seed)
    results = {"seed": seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}
    out = Path(a.out)
    for name in a.cases.split(","):
        wait_idle()
        epoch = time.time()
        log_mark()
        t = now()
        print(f"[{t:7.1f}] {name}: {CASES[name].__doc__.strip()}", flush=True)
        res = CASES[name](rnd)
        time.sleep(1.5)
        res["status_after"] = status()
        res["journal"] = journal(epoch)
        res["doc"] = CASES[name].__doc__.strip()
        results["cases"][name] = res
        for r in res["requests"]:
            extra = r.get("text", r.get("exception", ""))
            print(f"    {r['label']}: sent {r['sent']:.1f}, took {r['took']:.1f} s, "
                  f"first token {r.get('first_token')}, {extra[:60]!r}", flush=True)
        print("    " + ", ".join(f"{k}={v}" for k, v in res.items() if k.endswith("reads_code")), flush=True)
        out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
