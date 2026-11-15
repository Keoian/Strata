| request | prompt tokens | stock: reused / read, s | borrow: reused / read, s | stock2: reused / read, s |
|---|---|---|---|---|
| M1: main agent | 9,807 | 0 / 9,807, 31.4 | 0 / 9,807, 31.8 | 0 / 9,807, 31.7 |
| M2: main agent | 12,198 | 9,802 / 2,396, 11.6 | 9,802 / 2,396, 11.7 | 9,802 / 2,396, 11.6 |
| M3: main agent | 14,932 | 12,193 / 2,739, 12.9 | 12,193 / 2,739, 12.9 | 12,193 / 2,739, 13.0 |
| M4: main agent | 17,175 | 14,935 / 2,240, 11.0 | 14,935 / 2,240, 11.1 | 14,935 / 2,240, 11.0 |
| M5: main agent | 19,557 | 17,178 / 2,379, 11.4 | 17,178 / 2,379, 11.6 | 17,178 / 2,379, 11.6 |
| M6: main agent | 21,732 | 19,560 / 2,172, 11.0 | 19,560 / 2,172, 11.0 | 19,560 / 2,172, 11.0 |
| M7: main agent | 24,745 | 21,735 / 3,010, 13.4 | 21,735 / 3,010, 13.6 | 21,735 / 3,010, 13.6 |
| M8: main agent | 27,308 | 24,740 / 2,568, 12.9 | 24,740 / 2,568, 13.3 | 24,740 / 2,568, 13.3 |
| M9: main agent | 29,429 | 27,311 / 2,118, 11.0 | 27,311 / 2,118, 11.1 | 27,311 / 2,118, 11.1 |
| M10: main agent | 32,386 | 29,432 / 2,954, 13.5 | 29,432 / 2,954, 13.6 | 29,432 / 2,954, 13.6 |
| M11: main agent | 34,850 | 32,389 / 2,461, 12.1 | 32,389 / 2,461, 12.1 | 32,389 / 2,461, 12.2 |
| M12: main agent | 37,608 | 34,853 / 2,755, 12.8 | 34,853 / 2,755, 13.1 | 34,853 / 2,755, 13.0 |
| O1: unrelated chat | 78 | 0 / 78, 3.2 | 0 / 78, 3.4 | 0 / 78, 3.4 |
| S1: sub-agent 1 (the main agent's system prompt and tools) | 9,797 | 9,769 / 28, 0.8 | 9,769 / 28, 0.9 | 9,769 / 28, 0.8 |
| S2: sub-agent 1 (the main agent's system prompt and tools) | 12,571 | 9,792 / 2,779, 12.6 | 9,792 / 2,779, 12.8 | 9,792 / 2,779, 12.8 |
| M13: main agent returns after sub-agent 1 | 40,366 | 24,740 / 15,626, 42.4 | 37,611 / 2,755, 13.8 | 24,740 / 15,626, 43.0 |
| O2: unrelated chat again | 102 | 73 / 29, 1.6 | 73 / 29, 1.5 | 73 / 29, 1.6 |
| T1: sub-agent 2 | 9,799 | 9,769 / 30, 1.1 | 9,769 / 30, 1.1 | 9,769 / 30, 1.1 |
| T2: sub-agent 2 | 12,626 | 9,794 / 2,832, 13.1 | 9,794 / 2,832, 13.2 | 9,794 / 2,832, 13.2 |
| M14: main agent returns after sub-agent 2 | 43,041 | 12,193 / 30,848, 70.8 | 40,369 / 2,672, 13.5 | 12,193 / 30,848, 80.3 |
| S3: sub-agent 1 returns | 14,980 | 12,566 / 2,414, 12.6 | 12,566 / 2,414, 12.3 | 12,566 / 2,414, 13.3 |
| N: the main agent's two planted codes | 45,756 | 43,044 / 2,712, 14.0 | 43,044 / 2,712, 13.7 | 43,044 / 2,712, 14.9 |
| **all requests** | | **94,975 read, 337 s** | **53,928 read, 253 s** | **94,975 read, 351 s** |

stock: codes ['CDV3J3', 'A9YPVF'] in the answer [True, True], answer 'CDV3J3, A9YPVF'
borrow: codes ['CDV3J3', 'A9YPVF'] in the answer [True, True], answer 'CDV3J3, A9YPVF'
stock2: codes ['CDV3J3', 'A9YPVF'] in the answer [True, True], answer 'CDV3J3, A9YPVF'
stock vs borrow: largest logprob difference over the 4-token replies 0.594647; replies with other tokens: none
stock vs stock2: largest logprob difference over the 4-token replies 0.594647; replies with other tokens: none
