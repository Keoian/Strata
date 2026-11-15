# Source revisions recorded before any change (2026-11-06)

| Source | Revision |
|---|---|
| Strata upstream (github.com/Niko1221/Strata, main) | 9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e (2026-11-05 14:54 +0100), engine 0.1.57 / app 0.2.11 per AGENTS.md |
| Strata working branch | p100-bitlamas-iq4nl (from the SHA above) |
| llama.cpp pinned by Strata (third_party/ggml/VERSION.txt) | 3cf03257f219afbe7334045ff7c6a06ac68c627d |
| Reference fork Keoian/dual-p100-qwen-3-8-27b, branch p100-optimizations-keoian | 886a4b743637fb0815d6ec05f3eeadb390b0d40f (2026-11-01T01:32:32Z) |
| Model bitlamas/Qwen3.8-Flash-Next-GGUF | 408a8e6cceced7706d18ccba07e56e37e722a0a8 (lastModified 2026-10-17), folder Q4_K_XL-IQ4NL |
| NVIDIA driver | 580.178.04 (Ubuntu nvidia-headless-no-dkms-580-server + linux-modules-nvidia-580-server-7.0.0-43-generic) |
| CUDA toolkit | 12.8.1 runfile (pending install); 12.9 avoided: glibc 2.43 header clash per Strata TROUBLESHOOTING.md |
| Host compiler for nvcc | g++-14 14.3.0 (system default gcc 15.2 is newer than CUDA 12.8 supports) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-43-generic, glibc 2.43 |
