# Cancelling a chat while a picture is read (2026-11-07)

Owner report: "canceling a chat while loading an image can bork everything". Measured on this rig (2x Xeon E5-2696 v3,
2x P100 at 150 W, the served 256K config, the CPU picture reader with 16 threads and the 3600-token cap), through the
LAN API with `p100-results/tools/cancel_repro.py`. Every case uses fresh pictures (a random code in a white box on a
noisy background, so nothing was cached) and cancels the way a browser's Stop does (the socket is shut down).

## What was wrong

Nothing crashed and nothing was logged. The server froze for the rest of the picture's encode:

- The picture reader (strata-vision, llama.cpp mtmd) cannot stop in the middle of a picture.
- The encode holds the engine's single turn, on purpose, so the reader and the engine never run at once.
- The server checked for a cancel only before each picture.

So a cancelled 1440p picture kept the turn for its whole ~80 s encode, and every request waited behind it, text
included. The encode's result was then cached, so the server recovered by itself.

## The fix (`serve/server.py`, this branch)

A picture whose client left is read on only while that is useful:

- **Another request waits for the turn** (text, another picture, `/unload?wait`): the reader is killed within 0.1 s of
  noticing, started again at once (0.9-1.2 s on the CPU), and the turn is handed on. The half-written output is deleted.
- **A waiting request sends the same picture** (the follow-up question about it, a retry, Regenerate): the encode goes on
  and that request uses it, so the picture is not read twice.
- **Nobody waits:** the encode goes on and is kept, so a follow-up typed later finds the picture already read.
- A request that did not cancel is never stopped.

If the reader does not start again, the next picture's turn starts it. The web page sends a stopped message's picture
again with a follow-up, so an unread picture is read then. Log lines name each outcome:

```
[strata] images: a picture's client left and another request waits for the turn: its encode was stopped 0.5 s after the client left (the encoder starts again with the next request)
[strata] images: the image encoder was started again in 0.9 s
[strata] images: a picture whose client left was read to the end (73.8 s after it left): a waiting request sends the same picture
```

How it works:
- `Service.fifo` is a `Turn`: a Lock that also counts who waits for it.
- A request waiting for the turn registers its pictures (`Vision.wanting`, by the hash of their bytes).
- The encode's reply is polled every 0.1 s (`Vision._readline(stop=...)`).

Upstream: [PR #5495](https://github.com/Niko1221/Strata/pull/5495) (the `serve` change, its tests and a note in DETAILS.md).

Tests: `serve/test_vision_cancel.py` (7 tests). The whole `serve` suite passes (test_server, robustness, logprobs: 415
before and after, the same 2 thread warnings before and after; the rest: 993 passed, 49 skipped).

## Results: before (`cancel-baseline.json`) and after (`cancel-fixed.json`, `cancel-fixed-smoke.json`)

Times are seconds from sending to the end of the answer. BIG = 2560x1440 (3600 image tokens, ~80 s encode); MEDIUM =
1600x900 (~1430 tokens, ~17 s encode).

| case | what | before | after |
|---|---|---|---|
| s1 | text request 0.5 s after a cancelled BIG picture | **73.2** | **1.9** |
| s1 | then the follow-up about that picture (now read again: it was stopped for the text) | 15.0 (cached) | 94.1 (encode + read), code read right |
| s2 | follow-up about a cancelled BIG picture, sent 5 s later | 83.4 | 83.0 / 83.8 (uses the running encode), code read right |
| s3 | Stop, resend, Stop again, then text | **69.3** | **1.6** |
| s7 | text after a cancelled request with 2 MEDIUM pictures | 13.7 | 1.4 |
| s8 | the same, not streamed | 13.6 | 1.3 / 1.8 |
| s9 | the same, Anthropic `/v1/messages` | 13.4 | 1.3 |
| s4 | text queued behind a MEDIUM picture, cancelled while queued | gives its place up, picture unaffected | same |
| s5 | text after a cancel while a cached picture's prompt is read | 7.1 | 7.1 (engine, see below) |
| s6 | the same request after a cancel during the answer | prompt reused (first token 0.38 s), code read right | same |
| s10 | text after a cancelled 49,889-token text prompt (cancelled 6 s in) | 18.2 | 17.9 (engine, see below) |

The trade-off is in s1's second line. When another request ends a cancelled picture's encode, a later question about that
picture reads it again from the start.

## Other cancellation behaviour found (not changed)

- **A cancel while the engine reads a prompt waits for the end of the current chunk.** The engine checks for STOP between
  prompt chunks. s10: a 49,889-token prompt cancelled at 6 s stopped at the end of its first 8,192-token chunk, 24.2 s in
  (`8192 of 49889 read ... (cancelled)`), so the next request waited 18 s. s5: a 1,433-token picture prompt is one
  chunk and is read to the end (9 s). The part read is kept: the same request right after took 0.4 s. Shortening this
  needs a STOP check inside the chunk, an engine (C++) change that was not made.
- **Answers at temperature 0 are not byte-identical from run to run on this server, cancel or not**
  (`cancel-residue-check.json`). The same picture request sent twice with no cancel split after 115-177 characters. The
  request sent after a cancelled one matched the request before it for 473-515 characters. The cancel leaves no state
  behind. (The yardstick's teacher-forced log-probabilities stay byte-identical; the drift is in free generation.)
- **Web page, not about cancelling:** after a page reload, or when a saved chat is opened again, a picture shows as a
  chip with its name. The page stores pictures as names only and leaves them out of the request without saying so, so a
  follow-up then goes without the picture.
