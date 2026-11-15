# Cache limits for several agents: 16 parked conversations, 12 checkpoints, a 16 GiB budget (2026-11-15)

**Problem.** The box will serve several agents in turn (owner: a chat bot plus coding harnesses). With the defaults the
engine parks 4 conversations in at most 8192 MiB and keeps 6 checkpoints, so a fifth agent pushes the oldest out and
that agent reads its whole prompt again when it returns.

**Result.** Served config (split borrow and `--prompt-cache-spread` on), fresh engines, both cards at the 150 W limit
in every sample (`gpus.log`), `base` = as served, `big` =
`--prompt-cache 12 --conversation-cache-slots 16 --conversation-cache-mib 16384`:

| session | base | big |
|---|---|---|
| eight agents with different system prompts in turn, two rounds, two code checks (18 requests) | 196,234 tokens read, 792 s | 105,080 tokens read, 454 s |
| one coding agent, 36 turns, three jumps back and one cleared tool result (40 requests) | 140,599 tokens read, 645 s | 130,549 tokens read, 648 s |

- Rotation: in `base` every returning agent read its whole prompt again (12,150-12,683 tokens, 50-52 s); in `big` it
  read its new turn (2,339-2,872 tokens, 13-15 s). Evictions: 19 in `base`, 0 in `big`.
- Coding agent: the two rewrites far back resumed nearer the edit (10,846 -> 8,210 and 28,003 -> 20,597 tokens read),
  and an ordinary turn took 0.2-0.8 s longer (parking wrote 17.2 s in 32 snapshots in `base`, 23.9 s in 31 in `big`).
  Total time is the same.
- Memory in `big`: the cache held at most 14.5 GiB, the lowest `MemAvailable` was 11.2 GiB (`mem.log`), no parking
  was skipped, swap use did not change (321 MiB before and after).
- The planted codes came back right in all four sessions.

**Not an exactness check.** Adaptation was on, and the two variants read different amounts of each prompt, so their
4-token replies are not comparable: in the rotation seven replies had other tokens (largest logprob difference 1.18),
in the coding-agent session two (0.27). Parking and restoring are the release's own code and were already in use with
4 slots; only the counts changed.

**Limits.** One run of each variant, no second `base` run. Eight agents of 12-16 K tokens; agents with much longer
histories fill 16 GiB sooner, and then the oldest is dropped as before. The budget is RAM: with it full about 11 GiB
stayed available on this 125 GiB box.

Files: `ab.out` (the run), `table.md`, `base-*.json` / `big-*.json`, `engine-*.log`, `gpus.log`, `mem.log`.
Tools: `p100-results/tools/cache_limits_ab.sh`, `cache_limits_ab.py`.
