"""variant.py BASE.json NAME [--args a b ...] [--env K=V ...] [--set key=jsonvalue ...]: writes strata-NAME.json beside
BASE with extra engine args, env and top-level keys (the base config is not changed)."""
import json, sys, pathlib
base = pathlib.Path(sys.argv[1]); name = sys.argv[2]; cfg = json.loads(base.read_text())
mode = None; args = []; env = dict(cfg.get("env", {}))
for a in sys.argv[3:]:
    if a in ("--args", "--env", "--set"): mode = a; continue
    if mode == "--args": args.append(a)
    elif mode == "--env": k, v = a.split("=", 1); env[k] = v
    elif mode == "--set": k, v = a.split("=", 1); cfg[k] = json.loads(v)
cfg["args"] = list(cfg["args"]) + args
if env: cfg["env"] = env
cfg["log"] = str(base.parent / f"strata-{name}.log")
out = base.parent / f"strata-{name}.json"; out.write_text(json.dumps(cfg, indent=1)); print(out)
