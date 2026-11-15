# Layer-split borrow (upstream #5612), measured 2026-11-14

Rig: 2x Xeon E5-2696 v3, 2x P100 at 150 W (every GPU sample at the limit), the served 256K config with the engine of
`~/projects/strata-split-borrow` (upstream main 13eedf55 + b232e857, which includes the first-stage fix made after run 1).
Tools: `tools/split_borrow_ab.py`, `tools/split_borrow_ab.sh`, seed 8107, fresh engines: stock, borrow
(`STRATA_SPLIT_BORROW=1`), stock2.

| request | stock | borrow | stock2 |
|---|---|---|---|
| M13: main agent returns after sub-agent 1 (40,411 tokens) | 18,684 read, 49.6 s | 2,755 read, 13.8 s | 18,684 read, 53.9 s |
| M14: main agent returns after sub-agent 2 (43,086 tokens) | 33,317 read, 104.8 s | 2,672 read, 13.3 s | 33,317 read, 109.7 s |
| all 22 requests | 100,555 read, 385 s | 53,981 read, 252 s | 100,555 read, 399 s |

Every other request reused and read the same token counts in all three runs. Both planted codes came back right in
all three. `table.md` has every request.

## Does the switch change the output? (`adapt0/`, the three runs again with `--adapt-every 0`)

Token counts as above (all requests: 94,975 / 53,928 / 94,975 read; 337 / 253 / 351 s). No reply starts with another
token in any pair. Largest logprob difference per request over the 4-token replies:

| requests | stock vs stock2 | borrow vs stock2 |
|---|---|---|
| M1-M8, O1, S1, O2, T1, T2 | 0 | 0 |
| M9-M12, S2, S3 | 0.0016 to 0.5946 (the first stock run left the other two at M9, before any sub-agent) | 0 |
| M13, M14: the main agent returns | 0.0001, 0 | 0.0204, 0.0052 |

- The sub-agents' own requests (S1, S2, T1, T2, S3) are identical with and without the switch.
- M13 and M14 differ by at most 0.02. There the two runs compute different things: without the switch the engine
  reads 18,684 / 33,317 tokens again in the batched prompt path; with it the conversation comes back as it was read,
  turn by turn. A re-read is not bit-identical to the first read on this config.
- Two runs of the same engine without the switch differ by up to 0.59 (M11), more than the switch does anywhere.
- With adaptation on (the first table's runs): stock and stock2 identical, borrow other tokens on T2 and S3, largest
  difference 0.546. Not separated further.

## Served since 2026-11-14 22:01 UTC

The patch on the served branch (9ca96758, `build-p100/strata`; the engine before it is
`build-p100/strata.before-split-borrow`), with `STRATA_SPLIT_BORROW=1` and `--prompt-cache-spread` in the served
config. Live check on the service (`live/served.json`): 2 borrows logged, M13 read 2,755 and M14 2,672 tokens, both
codes right.

Run 1 (`ab-run1.out`, `borrow-failed.json`, `engine-borrow-failed.log`): the first patch failed on the first borrow
("a layer split's image restored as a single session") and the engine stopped; its stock and stock2 runs are the ones
above. The borrow run above is run 2 (`ab.out`, `gpus.log`).
