| request | prompt tokens | stock: reused / read, s | off: reused / read, s | on: reused / read, s |
|---|---|---|---|---|
| C1: short chat | 65 | 0 / 65, 1.3 | 0 / 65, 1.3 | 0 / 65, 1.6 |
| C2: short chat | 63 | 0 / 63, 1.3 | 0 / 63, 1.3 | 0 / 63, 1.3 |
| C3: short chat | 64 | 0 / 64, 1.3 | 0 / 64, 1.3 | 0 / 64, 1.2 |
| G1: agent, first request | 9,806 | 0 / 9,806, 31.6 | 0 / 9,806, 31.8 | 0 / 9,806, 31.9 |
| G2: agent, next turn | 12,602 | 9,801 / 2,801, 13.1 | 9,801 / 2,801, 13.2 | 9,801 / 2,801, 13.1 |
| end | 0 | 0 / 0, 0.0 | 0 / 0, 0.0 | 0 / 0, 0.0 |
| **all requests** | | **12,799 read, 49 s** | **12,799 read, 49 s** | **12,799 read, 49 s** |

stock: codes [] in the answer None, answer ''
off: codes [] in the answer None, answer ''
on: codes [] in the answer None, answer ''
stock vs off: largest logprob difference over the 4-token replies 0.000000; replies with other tokens: none
stock vs on: largest logprob difference over the 4-token replies 0.462170; replies with other tokens: ['G1: agent, first request', 'G2: agent, next turn']
