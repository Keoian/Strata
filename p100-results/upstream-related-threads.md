# Upstream Strata threads checked for duplicate work (2026-11-06, GitHub search API)

Searches: P100, IQ4_NL, bitlamas (0 hits), NUMA, k8v4, "stop rule".

Relevant, to read before the matching phase:
- #5125 [Benchmark] six NVIDIA cards with three Tesla P100s (sm_60), layer split, 0.2.10: MTP and DFlash2 (closed)
- #3976 [Engine] Layer-major prefill on a CUDA layer split: 32K prompts 2.0x faster (open)
- #5333 [Engine] Dual-socket CPU without AVX: both sockets decode 10-38% faster (open)
- #5372 [Engine] Idle GPU spin: both cards at 100% / ~110 W with no request (open) - matters under a 150 W cap
- #4610 [Benchmark] record host side: PCIe per card, NUMA, memlock, GPU temperature (open)
- #1135 k8v4 deterministic token repetition (RTX 5090, 0.1.40) (closed)
- #4624 / #4625 STRATA_POOL_SPIN_US=0 semantics (open)
No open PR found for: bitlamas, Q4_K + IQ4_NL expert pair, learned MTP stop rule, q4p attention.
