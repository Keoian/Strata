# The experimental speed projection, loaded and off unless a request asks for it (2026-11-15)

**What it is.** Strata's `data/experimental-speed-projection/` vector (480 KB) for the original Qwen3.8-Flash-Next. Its
package describes it as a refusal-direction projection: with it on the model declines far fewer requests, and the
owner is responsible for what it writes then (`docs/DETAILS.md`, "Experimental speed projection"). Owner request:
enable it on this box.

**How it is served.** Four engine arguments load it (`--control-vector-scaled <file>:1.0 --control-vector-layer-range
4 44 --cvec-mode project --cvec-dir per-layer`) and `"sampling": {"experimental_speed_projection": false}` in the
config makes off the default. A request turns it on with `"experimental_speed_projection": true` in its body; the web
page has the switch in Chat settings.

**Measured here** (served config with the new cache limits, fresh engines, both cards at the 150 W limit in all 412
samples of `gpus.log`):

- It loads on the two-card layer split: `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`.
  VRAM free with everything loaded is the same with and without it (789 / 597 MiB).
- Loaded and off is the stock model. With `--adapt-every 0` on both engines, five greedy 8-token requests (three
  short chats, an agent's 9,806-token first request with tools, its next turn of 12,602): largest logprob difference
  between the engine without the vector and the engine with it loaded and off 0.000000, same tokens (`table.md`).
- On is another model state: the same five requests with it on had other tokens on both agent requests (largest
  logprob difference 0.46).
- Agents with it on and off can take turns. Agent X (off) and agent Y (on), three turns each in turn: every return
  reused its own parked conversation and read only the new turn (2,385-2,798 tokens). The same prompt asked with the
  other setting is read whole (17,397 tokens), and the copy read with the first setting stays parked (5 tokens read
  on the way back).
- Speed, one 400-token greedy reply four times: off 42.6 and 43.0 tok/s, on 43.2 and 42.4 tok/s. A fresh 9,806-token
  prompt: 311.8 tok/s off, 306.1 tok/s on (one read each). No difference outside the spread of these few samples.

**Not measured.** How much less the model declines (the package's own claim is 1 of 50 against 50 of 50 on its test
set), and what it does to answer quality on ordinary prompts: the paired PPL / KL gates of this project were not run
with it on, because the default path, which they guard, is unchanged.

**Limits.** A client that cannot add a field to the request body cannot turn it on. One conversation asked with both
settings holds two parked copies. Every client with the API key can turn it on.

Files: `ab.out`, `table.md`, `stock.json` / `off.json` / `on.json` (exactness), `live.json` (the agents in turn and
the speed), `engine-*.log`, `gpus.log`, `mem.log`. Tools: `p100-results/tools/projection_ab.sh`, `projection_ab.py`.
