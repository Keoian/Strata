# `--prompt-cache-spread` with coding harnesses (2026-11-13)

Measured on this rig: 2x Xeon E5-2696 v3, 2x P100 at 150 W, the served 256K config
(`strata-bitlamas-p100-best-256k-k8v4.json`), engine `395fcb23`. Tool: `tools/agent_cache_ab.py` and
`tools/agent_cache_ab.sh`, seed 8107, three fresh engines in a row: base (the served config), spread (the same plus
`--prompt-cache-spread`), and base2 (the served config again, for the run-to-run floor).

## Why

The engine keeps up to 6 conversation checkpoints. By default the one used least recently leaves, the end of the
system prompt stays, and a checkpoint is made at the start of each assistant turn. So the cache holds the system prompt
and the last few turns. A harness that goes back further reads everything after the system prompt again.

The served engine's log shows this happening with a coding harness (engine start 25, before 2026-11-13):
- A 24,915-token prompt that had been sent before was sent again after the session had grown to 35,542 tokens.
- It resumed at 3,416 tokens (the system prompt) and read 21,499 again, which took 51.5 s.
- A 16,365-token prompt then resumed at 3,416 too, 36.0 s.

`--prompt-cache-spread` drops instead the checkpoint whose removal leaves the smallest gap. The kept points are dense
near the end and thinner further back.

## The workload

One conversation, sent the way a coding harness sends it (`/v1/chat/completions` with 8 tools):
- **Messages:** the system prompt and tools (9,802 tokens), then 36 turns of assistant reasoning, a read call and its
  result (7,000 characters of this repo's source with line numbers, about 2,400 tokens). There are three user tasks.
- **Each request:** greedy, `max_tokens` 4, logprobs, so its time is almost all the prompt read. The session ends at
  99,434 tokens.

The harness goes back four times:

| event | what is sent | how far back from the tip |
|---|---|---|
| J4 (after turn 16) | turn 12's prompt again, then the session goes on from its tip | 10K tokens |
| J6 (after turn 22) | turn 16's prompt again | 16K tokens |
| J10 (after turn 30) | turn 20's prompt again | 25K tokens |
| E9 (turn 34) | turn 25's tool result replaced by `[Old tool result content cleared]` and kept so | 20K tokens |

E9 is what OpenCode's prune and Claude Code's clearing of old tool results do. At the end, two codes are asked for:
one planted in the first user message, one in turn 5's tool result.

## Results

Tokens read again and the engine's prompt time. The token counts are the same in base and base2 to within 8 tokens.

| | base | spread | base2 |
|---|---|---|---|
| J4: turn 12's prompt again | 5 tokens, 1.2 s | 5 tokens, 1.2 s | 5 tokens, 1.2 s |
| **J6: turn 16's prompt again** | 10,202 tokens, 36.1 s | **2,476 tokens, 14.1 s** | 10,203 tokens, 41.7 s |
| J10: turn 20's prompt again | 10,846 tokens, 42.4 s | 10,846 tokens, 46.3 s | 10,846 tokens, 47.6 s |
| **E9: an old tool result cleared** | 32,983 tokens, 123.6 s | **28,003 tokens, 108.2 s** | 32,983 tokens, 131.9 s |
| the 36 normal turns and the answer check | the same reuse in all three runs | | |
| all 41 requests | 153,354 tokens, 704 s | **140,651 tokens, 688 s** | 153,358 tokens, 742 s |

- **Better:** J6 saves 22-28 s and E9 saves 15-24 s. Spread reads 8.3% fewer prompt tokens over the session.
- **Never worse:** no request read more tokens with spread. J4 is an exact hit both ways. J10 is the same both ways:
  after J6 the chain only holds points up to 47,862, and the tip comes back from the parked copy of the conversation,
  which brings no checkpoints between those points back. That is the conversation cache, not the eviction rule.
- **Whole-session time:** within the run-to-run spread. Base and base2 differ by 38 s (5%), because the normal turns
  in base2 and spread are 0.5-1.5 s slower than in base. Base began with cooler cards (43/49 C against 55/60 C), and
  the thermal-slowdown samples grow from run to run (23, 31, 48). The token counts are the reliable measure here.

## Answers and exactness

- The two codes (CDV3J3, A9YPVF) came back right in all three runs.
- Logprobs are not repeatable on this config, even on the first token of a reply, which comes from the prompt read.
  - Base against base2: 3 replies of 40 start with another token, and the largest difference in the first token's top 5
    is 4.67.
  - Base against spread: 3 start with another token, and the largest difference is 3.26.
  - So the spread run differs from base by no more than base differs from itself. There is no sign that the flag
    changes what the model computes; it only chooses which checkpoint a prompt resumes from.

## Power

The limit was 150 W on both cards for every measurement. It read 250 W only from 15:33:46 to 15:34:11, while the
first engine was loading at 35-44 W. Those were the last minutes before the cap was put back after the unattended
upgrade: RUNBOOK section 2, "The cap after a service restart".

Over 1,118 samples at 5 s:
- Mean draw 77 / 79 W.
- 32 single readings above 165 W, the highest 210.5 W, all with the driver's power-cap reason set.
- Power-cap reason in 82 samples. Thermal slowdown in 102, more in each run (23, 31, 48), at most 80 C in each.
  The exactness runs of 2026-11-07 (`engine-stop/exact-gpus.log`) had it in 2 of 776 samples.
- Clocks 405-1328 MHz, at most 80 C.
- File: `gpus.log`.

## Not covered

- One seed and one synthetic session, through the OpenAI chat API only. The real harness sessions were not replayed:
  the log has no token streams.
- Sessions longer than ~100K tokens. With more turns the default keeps only the last few, so a jump of the same
  depth would fall further back. This was not measured.
- Parallel requests (the server serves one at a time), images, `/v1/messages`.

Files: `base.json`, `spread.json`, `base2.json` (every request with its timings and logprobs), `engine-*.log`,
`agent-*.json` (the configs), `ab.out`, `table.md` (every request), `gpus.log`.
