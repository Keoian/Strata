#!/usr/bin/env python3
"""vision_compare.py - encode test images with strata-vision under two mmproj files, report tokens, grid, encode time,
peak RSS, and how close the embeddings are (per-row cosine, rel-L2), the BF16 projector being the reference.
  vision_compare.py --exe build-vision/bin/strata-vision --model <shard 1> --ref <bf16> --test <q8> --threads 16
                    --max-tokens 3600 --out DIR img1 [img2 ...]"""
import argparse, json, pathlib, struct, subprocess, sys, time
import numpy as np

def run(exe, mmproj, model, threads, max_tokens, imgs, outdir, tag):
    cmd = ["/usr/bin/time", "-f", "MAXRSS_KB %M", exe, "--mmproj", mmproj, "--model", model, "--threads", str(threads),
           "--max-tokens", str(max_tokens)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ready = p.stdout.readline().strip()
    res = {}
    for img in imgs:
        out = outdir / f"{pathlib.Path(img).stem}.{tag}.sve"
        t0 = time.time()
        p.stdin.write(f"ENC {img} {out}\n"); p.stdin.flush()
        line = p.stdout.readline().strip()
        res[img] = {"line": line, "wall_s": round(time.time() - t0, 2), "file": str(out)}
    p.stdin.write("QUIT\n"); p.stdin.flush()
    _, err = p.communicate(timeout=60)
    rss = [l for l in err.splitlines() if l.startswith("MAXRSS_KB")]
    return ready, res, (int(rss[-1].split()[1]) // 1024 if rss else None), err

def load(path):
    b = pathlib.Path(path).read_bytes()
    magic, n, nx, ny, d = struct.unpack("<5i", b[:20])
    return np.frombuffer(b[20:], dtype=np.float32).reshape(n, d), nx, ny

ap = argparse.ArgumentParser()
for k in ("exe", "model", "ref", "test", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--max-tokens", type=int, default=3600)
ap.add_argument("imgs", nargs="+")
a = ap.parse_args()
outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
report = {}
for tag, mm in (("bf16", a.ref), ("q8", a.test)):
    ready, res, rss, err = run(a.exe, mm, a.model, a.threads, a.max_tokens, a.imgs, outdir, tag)
    report[tag] = {"ready": ready, "peak_rss_mib": rss, "images": res}
    (outdir / f"{tag}.stderr.txt").write_text(err)
    print(tag, ready, "peak RSS", rss, "MiB"); [print("  ", pathlib.Path(k).name, v["line"], f"wall {v['wall_s']} s") for k, v in res.items()]
for img in a.imgs:
    try:
        r, nx, ny = load(report["bf16"]["images"][img]["file"]); t, _, _ = load(report["q8"]["images"][img]["file"])
    except Exception as e:
        print(img, "compare failed:", e); continue
    cos = (r * t).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(t, axis=1))
    rel = float(np.linalg.norm(r - t) / np.linalg.norm(r))
    report.setdefault("compare", {})[img] = {"tokens": int(r.shape[0]), "grid": [nx, ny], "cos_mean": float(cos.mean()),
                                             "cos_min": float(cos.min()), "rel_l2": rel}
    print(f"{pathlib.Path(img).name}: {r.shape[0]} tokens ({nx}x{ny} grid), cosine mean {cos.mean():.5f} min {cos.min():.5f}, rel-L2 {rel:.4f}")
(outdir / "report.json").write_text(json.dumps(report, indent=1))
