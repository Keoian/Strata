# A Stop while the engine reads a long prompt (2026-11-07)

Measured on this rig: 2x Xeon E5-2696 v3, 2x P100 at 150 W, the served 256K config, a layer split of 0-23 / 24-47,
8192-token prompt chunks. The prototype is on branch `p100-engine-stop` (commit `877dd9bf`, worktree
`~/projects/strata-engine-stop`). **Deployed 2026-11-07** as commit `395fcb23` on `p100-bitlamas-iq4nl` (the same
engine source as `877dd9bf`), built into `build-p100/strata`; the engine from before is kept as
`build-p100/strata.before-stop-fix`.

## What was slow

- The engine checked for STOP only before each prompt chunk, and only on the first card.
- The second card read every chunk handed to it to the end.
- So a cancel 6 s into a 49,890-token prompt took effect at 24 s, when both cards had finished the first chunk, and the
  next request waited 18 s.

## The two fixes (both can be turned off for A/B)

1. **The second card checks for STOP too** before each chunk it is handed (`STRATA_STOP_STAGES=0`: as before).
   Mid-prompt checkpoints are safe: a checkpoint is kept only when every card made its part.
2. **Every card checks between layers inside a chunk** (`STRATA_STOP_LAYERS=0`: as before). It stops the expert copies,
   waits for its card to finish queued work, and returns "cancelled".
   - Before this, leaving a chunk early only happened on errors, and those end the engine.
   - A cancelled batched read already keeps none of its live session (`pp_exact`), only checkpoints made before it.

## Results (one binary, three settings, same prompts: `stop_ab.sh`, seed 8107)

| | both off | fix 1 | fixes 1+2 |
|---|---|---|---|
| s10: text after a 49,890-token prompt cancelled 6 s in | 17.9 s | 6.3 s | **1.1 s** |
| s12: text after the same kind of prompt cancelled 16 s in | 20.1 s | 8.4 s | **1.2 s** |
| the cancelled read ended (engine log) at | 23.7 s / 36.2 s | 12.3 s / 24.5 s | 7.0 s / 17.4 s |
| s13: a 28,963-token prompt cancelled 10 s in, then sent in full: code right | yes | yes | yes |
| s13: that full read | 491.0 tok/s | 489.3 tok/s | 490.4 tok/s |
| s13: text afterwards | 0.8 s, OK | 0.8 s, OK | 0.8 s, OK |

Fixes 1+2 log where each card stopped, for example `stopped in the chunk from token 0 before layer 34` (GPU1) and
`... from token 8192 before layer 10` (GPU0) for s12. No CUDA error appeared in any run. Power limit 150 W
throughout, no throttle reasons, at most 60 C (`gpus.log`).

**The trade-off:** with both off, s12's cancelled read went on to 16,384 tokens and saved one mid-prompt checkpoint,
which a resend of the same prompt could reuse. With the fixes, the read stops before any checkpoint is complete, so a
resend reads from the start. That is the same work the old behaviour did after the cancel; it now happens only when the
prompt is sent again.

## Exactness after a stop in the middle of a chunk (`tools/stop_exact.sh`, seed 8107)

The same requests on fresh engines (the prototype binary, `STRATA_STOP_*` 0/0 = the old stop, 1/1 = both fixes),
greedy with logprobs (top 5):
- T: a short text, 64 tokens.
- P: the 28,881-token code prompt, 5 tokens.
- L1 and L2: two ~50K-token prompts cancelled at 6 s and 16 s.
- P again, then Q (a new 20,195-token code prompt, 4 tokens), then T again.
- G: 200 tokens of code, text only.

With both fixes the stops were in the middle of chunks on both cards: L1 on GPU0 before layer 12; L2 on GPU1 before
layer 33 and on GPU0 (chunk from 8192) before layer 9. The largest logprob difference is shown; "=" is identical to
the last digit the API gives (6 decimals) in every row and every top-5 alternative.

| comparison | T1, P1 (before any cancel) | P2, Q (after the cancels) | T2, T3 (after the cancels) | G |
|---|---|---|---|---|
| old stop, run 1 vs run 2 | = | 0.0008, 0.0018 | 0.10, 0.08 (wording changes) | = |
| old stop vs both fixes | = | 0.0023, 0.0016 | 0.10, 0.13 (wording changes) | = |
| adaptation off (`--adapt-every 0`): old stop vs both fixes | = | **=** | 0.22, 0.32 | = |
| adaptation off: old stop, run 1 vs run 2 | T1 0.0069 (row 59), P1 = | = | 0.22, 0.36 | = |

Read together:
- **What the fix could change, it does not.** With the adaptive expert cache off, P and Q come out identical after
  the mid-chunk stops and after the old stop. Their logprobs come almost all from reading the prompt.
- **Generated tokens are not repeatable on this config, with or without the fix.** The old stop differs from its own
  second run as much as it differs from the fixes. With the adaptation off, T1 differs between two runs before any
  cancel.
  - A likely cause, from reading the code (not tested): `STRATA_SPEC_COST=1` sizes the draft windows from measured
    timings (`generate.cpp`, "the MTP window by expected tokens per ms at this machine's measured row costs"), and
    windows of other sizes change the last digits.
  - With the adaptation on, a decode that differs also swaps other experts in, so P2 and Q differ too.
- The answers were right in every run (codes CDV3J3 and A9YPVF), and G's 200 tokens were the same in all six.

Power: the limit stayed 150 W on both cards. During prompt reads the driver held them at it: the power-cap reason in
78 of 774 five-second samples, clocks down to 1012-1227 MHz. Single readings went up to 191 W. At most 79 C
(`exact-gpus.log`).

**Live check after the deploy** (the service, 22:07, `live-s10-s13.json`):
- s10: text sent 0.5 s after a cancel 6 s into a ~50K-token prompt took 1.1 s.
- s13: the code prompt sent in full after a stop in the middle of a chunk gave the code; the text after it took 0.8 s.
- The engine log shows the stops ("before layer 13", "before layer 22") and no error.

**Not checked:**
- The layer-major prompt path (off here).
- A single card (fix 2 applies there; fix 1 needs a split).
- HIP and SYCL builds.
