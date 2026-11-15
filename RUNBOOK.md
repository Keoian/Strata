# RUNBOOK: Strata + bitlamas Qwen3.8-Flash-Next Q4_K_XL-IQ4NL on 2x Tesla P100

Host: `p100-box`. 2x Xeon E5-2696 v3 (18 cores each, 2 NUMA nodes), 8x 16 GB DDR3-1600 RDIMM
(one per channel, 4 channels per socket), 2x Tesla P100-PCIE-16GB (one per socket), Ubuntu 26.04.1, kernel 7.0.0-43.
Status of each section is stated in it. Nothing here claims a speed that is not in `OPTLOG.md` with its measurement.

## 1. NVIDIA driver (done 2026-11-06)

Ubuntu's 580 server branch with the prebuilt, signed kernel modules (no DKMS compile; the `-generic` meta package
follows kernel updates). 580 is the last branch that supports Pascal. The open kernel modules do not support Pascal.

```sh
sudo apt-get install -y linux-modules-nvidia-580-server-generic nvidia-headless-no-dkms-580-server nvidia-utils-580-server
```

The package blacklists `nouveau` with `alias nouveau off` (`/usr/lib/modprobe.d/nvidia-graphics-drivers.conf`), so
`modprobe -r nouveau` silently does nothing on a running system. To switch without a reboot: `sudo rmmod nouveau &&
sudo modprobe nvidia`. At boot nouveau is no longer loaded. The Radeon HD 7450 (display) keeps `radeon`.

Verify: `nvidia-smi -L` lists both P100s:

| index | UUID | PCI | NUMA node | link |
|---|---|---|---|---|
| 0 | GPU-00000000-0000-0000-0000-000000000000 | 0000:04:00.0 | 0 | PCIe 3.0 **x8** (card supports x16) |
| 1 | GPU-00000000-0000-0000-0000-000000000001 | 0000:81:00.0 | 1 | PCIe 3.0 x16 |

Rollback: `sudo apt-get purge 'nvidia-*580*' 'linux-modules-nvidia-580*' && sudo reboot` (nouveau returns).

### Planned hardware change (owner, after this effort): GPU0 from x8 to x16

GPU0 sits behind CPU0 root port `00:02.2`, which is x8 only (port 2 is bifurcated x8x8; `00:02.0` is empty). CPU0's
only x16 port, `00:03.0`, holds the Radeon HD 7450 display card (which needs one lane). Plan: swap the two cards
(P100 into the Radeon's slot, Radeon into the P100's old slot), so both P100s run x16 and GPU0 stays on NUMA node 0.
Blocker: in that (top) slot the P100's cooling shroud hits the RAM; the owner will design and print a new cooler
first. Until then every measurement here is with GPU0 at x8. After the swap:

1. `nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.width.current --format=csv` shows x16 on both cards.
2. `p100-power-cap.service` needs no change (it matches UUIDs, not bus IDs); run the post-boot check in section 2.
3. Re-run `p100-results/tools/pcie_bw` per GPU and node (OPTLOG Phase 1 table) and re-tune `"layer_split"`: the
   split chosen under x8 gives GPU0 fewer layers on purpose and is likely wrong at x16.

## 2. Mandatory 150 W power cap on BOTH P100s, at every boot (done 2026-11-06; reboot persistence VERIFIED 2026-11-06 18:55 UTC)

Both cards report min 125 W, default 250 W, max 250 W, so 150 W is in range.

Files (copies in `p100-results/tools/`):

- `/etc/p100-power-cap.conf` - `LIMIT_W=150`, the two UUIDs above, `WAIT_SECS=180`.
- `/usr/local/sbin/p100-power-cap` (root, 0755) - waits up to `WAIT_SECS` for the driver and for BOTH UUIDs, refuses a
  card whose name is not a Tesla P100 or a limit outside the card's range, enables persistence mode, runs
  `nvidia-smi -i <UUID> -pl 150` per card, then reads `power.limit` and `enforced.power.limit` back. Exits 1 with an
  `ERROR:` line unless every listed card is verified at the limit.
- `/etc/systemd/system/p100-power-cap.service` - `Type=oneshot`, `RemainAfterExit=yes`, after
  `systemd-modules-load` and `nvidia-persistenced`, `WantedBy=multi-user.target`, enabled.

Install (as done):

```sh
sudo install -m 0644 -o root -g root p100-results/tools/p100-power-cap.conf /etc/p100-power-cap.conf
sudo install -m 0755 -o root -g root p100-results/tools/p100-power-cap /usr/local/sbin/p100-power-cap
sudo install -m 0644 -o root -g root p100-results/tools/p100-power-cap.service /etc/systemd/system/p100-power-cap.service
sudo systemctl daemon-reload
sudo systemctl enable --now p100-power-cap.service
```

**Post-boot verification command** (both lines must show 150.00 W, and the service `active (exited)`):

```sh
systemctl is-enabled p100-power-cap.service; systemctl status p100-power-cap.service --no-pager | head -5
journalctl -u p100-power-cap.service -b --no-pager | tail -4
nvidia-smi --query-gpu=index,uuid,pci.bus_id,persistence_mode,power.limit,enforced.power.limit --format=csv
```

Tested on 2026-11-06: the service ran, both cards report `power.limit=150.00 W enforced=150.00 W`. Failure paths
tested with a copy of the script: a missing UUID exits 1 (`after 4s still missing: GPU-0000...`), a 300 W limit exits 1
(`300 W outside supported range 125.00-250.00 W`); the real limits stayed at 150 W.

**Reboot persistence: NOT YET VERIFIED** (the reboot waits for an owner-approved checkpoint).

Rollback: `sudo systemctl disable --now p100-power-cap.service && sudo rm /etc/systemd/system/p100-power-cap.service
/usr/local/sbin/p100-power-cap /etc/p100-power-cap.conf && sudo systemctl daemon-reload && sudo nvidia-smi -pl 250`
(only with the owner's say-so: the 150 W cap is a standing requirement for every experiment).

Any future Strata boot service must order itself `After=p100-power-cap.service` and `Requires=p100-power-cap.service`.

### The cap after a service restart (added 2026-11-13)

**What happened:** at 06:19 on 2026-11-13 the daily unattended upgrade installed a new libc and restarted the services
that use it, among them `nvidia-persistenced` and `strata-bitlamas`, at the same second.
- Stopping `nvidia-persistenced` turns persistence mode off on both cards. With the engine stopped too, nothing held
  them, and the driver reset both to the default 250 W.
- `strata-bitlamas` came back at 06:19:51. Its `Requires=p100-power-cap.service` was already satisfied, because that
  oneshot stays active from boot, so the cap was not applied again.
- The service ran with a 250 W limit until 15:34. The cap was put back with `sudo /usr/local/sbin/p100-power-cap`.
  No power log ran in that window.

**The two drop-ins that close it** (copies in `p100-results/tools/systemd-dropins/`):
- `/etc/systemd/system/strata-bitlamas.service.d/p100-power-cap.conf`: `ExecStartPre=+/usr/local/sbin/p100-power-cap`.
  Every start of the service applies the cap and checks it first. If it fails, the service does not start.
- `/etc/systemd/system/nvidia-persistenced.service.d/p100-power-cap.conf`:
  `ExecStartPost=-+/usr/local/sbin/p100-power-cap`. After every start of the daemon, persistence mode and the cap are
  applied again. A failure is logged but does not stop the daemon.

Install (after `sudo mkdir -p` of both `.d` folders), from the repo:

```sh
sudo install -m 0644 p100-results/tools/systemd-dropins/strata-bitlamas.service.d-p100-power-cap.conf /etc/systemd/system/strata-bitlamas.service.d/p100-power-cap.conf
sudo install -m 0644 p100-results/tools/systemd-dropins/nvidia-persistenced.service.d-p100-power-cap.conf /etc/systemd/system/nvidia-persistenced.service.d/p100-power-cap.conf
sudo systemctl daemon-reload
```

Rollback: `sudo rm /etc/systemd/system/{strata-bitlamas,nvidia-persistenced}.service.d/p100-power-cap.conf && sudo systemctl daemon-reload`.

### Boot-time record of the cap (`p100-boot-check.service`, enabled)

After every boot `/usr/local/sbin/p100-boot-check` (copy in `p100-results/tools/`) writes
`p100-results/boot-checks/boot-<UTC time>.txt`: both cards' power.limit / enforced.power.limit (PASS only at 150 W),
`iommu=pt` and no `crashkernel` on the kernel line, the GPU IOMMU group types, the scratch mount and sshd. The first
reboot after this effort is the persistence test: `cat ~/projects/strata/p100-results/boot-checks/boot-*.txt`.
Result of that reboot (`boot-20261106T185510Z.txt`): cap 150 W on both cards 20 s after boot (PASS), `iommu=pt`
PASS, no crashkernel PASS, both GPU IOMMU groups `identity`, scratch mounted, sshd active.

## 2b. IOMMU passthrough and no kdump reservation (done 2026-11-06, live and for the next boot)

Measured (OPTLOG Phase 1): with the IOMMU translating (`DMA-FQ`) GPU1's pinned copies were 11.7 GB/s from its own
node and 10.3 GB/s from the other; in passthrough both are 12.4 GB/s. kdump was already off (`USE_KDUMP=0`, no crash
kernel loaded) but its `crashkernel=` reservation held 4.25 GiB, 4 GiB of it on NUMA node 1.

Live, without a reboot (as done; each GPU is alone in its IOMMU group, 51 for 04:00.0 and 1 for 81:00.0):

```sh
sudo systemctl stop nvidia-persistenced
for d in 0000:04:00.0 0000:81:00.0; do echo $d | sudo tee /sys/bus/pci/drivers/nvidia/unbind; done
for g in 51 1; do echo identity | sudo tee /sys/kernel/iommu_groups/$g/type; done
for d in 0000:04:00.0 0000:81:00.0; do echo $d | sudo tee /sys/bus/pci/drivers/nvidia/bind; done
sudo systemctl start nvidia-persistenced
sudo systemctl restart p100-power-cap.service        # a driver rebind resets the limit to 250 W: re-cap first
echo 0 | sudo tee /sys/kernel/kexec_crash_size       # MemTotal 121.5 -> 125.7 GiB; nodes 62.80 / 62.93 GiB
```

For every later boot: `/etc/default/grub.d/zz-p100-strata.cfg` (sorts after `kdump-tools.cfg`) removes
`crashkernel=` and adds `iommu=pt`, then `sudo update-grub`. Check after a boot: `cat /proc/cmdline` has `iommu=pt`
and no `crashkernel`; `cat /sys/kernel/iommu_groups/*/type | sort | uniq -c` shows `identity`.
Rollback: `sudo rm /etc/default/grub.d/zz-p100-strata.cfg && sudo update-grub && sudo reboot`.

## 3. CUDA 12.8.1 toolkit (done 2026-11-06)

CUDA 13 cannot compile sm_60. Strata's docs say CUDA 12.9 clashes with glibc 2.43's math headers; on this Ubuntu
(glibc 2.43, system headers) **12.8 clashes too** (`cospi`, `sinpi`, `rsqrt` and their `f` forms: glibc >= 2.41 declares
them `noexcept`). The 2x P100 community report that built with 12.8 used a conda GCC with its own glibc headers.

1. The runfile's installer needs `libxml2.so.2`, which Ubuntu 26.04 no longer ships (only `libxml2.so.16`). So the
   runfile is unpacked without running it and the toolkit components are merged by hand (no driver is installed):

   ```sh
   sh cuda_12.8.1_570.124.06_linux.run --noexec --keep --target /mnt/scratch/downloads/cuda-12.8.1-extract   # md5 e281945bf4ea42788db29457ebc1e6ae
   X=/mnt/scratch/downloads/cuda-12.8.1-extract/builds; D=/usr/local/cuda-12.8
   for c in cuda_cccl cuda_cudart cuda_cuobjdump cuda_cupti cuda_cuxxfilt cuda_nvcc cuda_nvdisasm cuda_nvml_dev cuda_nvprune \
            cuda_nvrtc cuda_nvtx cuda_profiler_api cuda_sanitizer_api cuda_gdb libcublas libcufft libcurand libcusolver \
            libcusparse libnvjitlink libnvfatbin; do sudo rsync -aK $X/$c/ $D/; done
   sudo rsync -a $X/nsight_systems/ $D/nsight-systems/ && sudo ln -sf $D/nsight-systems/bin/nsys $D/bin/nsys
   ```

2. Header patch, reversible: `/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h` gets
   `noexcept(true)` (C++ only, through a macro) on those six declarations. The original is beside it as
   `math_functions.h.orig`; `sudo cp -a math_functions.h.orig math_functions.h` undoes it.
3. Host compiler: `g++-14` (CUDA 12.8 supports GCC <= 14; the system default is GCC 15.2).

Result: `nvcc` 12.8.93. `p100-results/tools/pcie_bw.cu` compiles for sm_60 and runs on both cards.

## 4. Strata build (done 2026-11-06; branch `p100-bitlamas-iq4nl`)

```sh
cd ~/projects/strata
cmake -S . -B build-p100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DSTRATA_GGML_DIR=$HOME/projects/llama.cpp-strata-pin \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build-p100 --target strata -j 40
```

`~/projects/llama.cpp-strata-pin` is llama.cpp at Strata's pinned `3cf03257f219afbe7334045ff7c6a06ac68c627d`
(`third_party/ggml/VERSION.txt`); its `gguf-py` is installed editable in Strata's `.venv`.

## 5. Model files

`~/models/bitlamas-Qwen3.8-Flash-Next-GGUF/Q4_K_XL-IQ4NL/` (SSD), revision `408a8e6cceced7706d18ccba07e56e37e722a0a8`,
downloaded with `hf download ... --revision 408a8e6c... --include "Q4_K_XL-IQ4NL/*"`; all four shards match the repo's
`SHA256SUMS` (`p100-results/model/sha256-check.txt`).

- Strata pack: `~/models/strata-packs/bitlamas-q4kxl-iq4nl`, made with
  `STRATA_GGUF_PY=~/projects/llama.cpp-strata-pin/gguf-py .venv/bin/python tools/iq_pack.py --gguf <shard 1> --out <pack> --compat-bf16`
  (46 s; no experts.bin: the arena is filled from the GGUF shards at each start).
- MTP draft layer: `~/models/strata-mtp/rt` (from the original Qwen checkpoint's MTP tensors; hashes match the
  community 2x P100 report, `p100-results/model/mtp-rt-sha256.txt`).
- Expert profile learned on this model: `data/expert-profile-bitlamas-corpus.bin` (routing of the 45 prompts of
  `bench/profile_corpus.json`, `p100-results/tools/collect_routing_serve.py` + `tools/make_profile.py --reorder`).

## 6. Storage

- SSD (`/`, LVM on the PNY CS900 500 GB): OS, source, model, PLE table, results.
- HDD 1 TB (`/mnt/scratch`, ext4, label `scratch`, UUID [removed], `nofail` in
  `/etc/fstab`): downloads and cold storage only. Never active PLE reads from it. (The owner had it wiped on
  2026-11-06 before the project spec arrived; the spec's "do not format disks" is followed from then on.)

## 7. The tuned server (best measured configuration)

Config: `p100-configs/strata-bitlamas-p100-best.json` (32K context, KV int8) - the stock setup config plus:

| setting | what it does | evidence (OPTLOG.md) |
|---|---|---|
| `STRATA_ARENA_NUMA=1` (env) | each expert's row halves on the NUMA node of the CPU pool slice that reads them, pinned per chunk before the touch; the PLE table half per node (new code on this branch); needs RLIMIT_MEMLOCK unlimited for its pre-pin | CPU pool 28.6 -> 15 ms/window; byte-identical output; 100% / 100% placed on every start, cold or warm (2026-11-07) |
| `--pcie-frac 0` | no expert is copied over PCIe at decode; the CPU pool takes the misses | 41.1 -> 51.4 tok/s (code) on the placed arena |
| `STRATA_Q8_SM60=1` (env) | the Pascal GP100 Q8_0 decode GEMV for the dense projections and the head | +16% code; nearer llama.cpp than stock (KL 0.041 vs 0.045) |
| `STRATA_Q8_SM60_EXACT=1` (env) | the Q8_SM60 kernel with the exact kernels' addition order (new code) | bitwise the stock kernels; about -1% vs plain Q8_SM60 |
| `STRATA_DOWN_WS=1` (env) | the weight-staged IQ4_NL down kernel (sm_60) | +2-5%; byte-identical output |
| `--trim-stage-weights` + `"layer_split": "24"` | each card loads only its own layers' dense weights (an explicit split lets CUDA0 trim too) | 6392 -> 6648 expert slots; 61.5 (auto split) -> 64.0 tok/s |
| `STRATA_SPEC_COST=1` (env) | the MTP draft window chosen by measured cost and acceptance (exact verification) | +8% prose, +38% chat, never below without it |

Measured (final interleaved run, 4 fresh servers per arm, `OPTLOG.md` Phase 4): decode **x1.73 geomean over the stock
config** across 8 cases - code 37.2 -> 62.3, edit 33.6 -> 61.8, prose 28.9 -> 46.8, chat 27.4 -> 51.0, tool 23.5 ->
46.5, thinking 31.8 -> 56.6, 15K-token prompt 33.3 -> 56.4, 32K-token prompt 39.0 -> 55.5 tok/s; at 128K context
(k8v4) 64K / 120K-token prompts 32.7 -> 51.9 / 36.3 -> 49.8. Prompt reading is unchanged (~313 tok/s at 15K, ~493 at
32K, ~504 at 120K). Quality: PPL within 0.15% of llama.cpp; HumanEval 158/164 vs 159/164 stock.

Run it by hand (the memlock limit is needed for the arena's pre-pin and the PLE lock; without it the engine still
starts and says what it could not do):

```sh
cd ~/projects/strata
sudo prlimit --memlock=unlimited:unlimited --pid $$        # once per shell (or use the service below)
.venv/bin/python serve/server.py --engine strata --config p100-configs/strata-bitlamas-p100-best.json --port 8080 --no-api-key
```

By hand like this the server listens on 127.0.0.1:8080 only (OpenAI- and Anthropic-compatible API, `GET /health`,
`GET /metrics`). The service below serves the LAN with an API key instead.
Load time is about 3 minutes (63.5 GiB of experts filled from the GGUF through the file cache, placed per node).
Long context: `p100-configs/strata-bitlamas-p100-best-128k-k8v4.json` (131072 tokens, KV k8v4, 32768 cells resident) and
`p100-configs/strata-bitlamas-p100-best-256k-k8v4.json` (262144 tokens, the same; what the service runs since 2026-11-07).

## 8. Start on boot (enabled 2026-11-06; on the LAN with an API key and at 256K context since 2026-11-07)

`/etc/systemd/system/strata-bitlamas.service` (copy in `p100-results/tools/strata-bitlamas.service`): runs the server
above as `user` with the **256K config** (`strata-bitlamas-p100-best-256k-k8v4.json`: every switch of the tuned
config, 262144 tokens, KV k8v4 with 32768 cells per QSA layer in VRAM and the rest in 1.2 GiB of pinned RAM),
`Requires=` and `After=` `p100-power-cap.service` (the 150 W cap is in force before any GPU work),
`LimitMEMLOCK=infinity`, restarts on failure. Back to 32K: put `strata-bitlamas-p100-best.json` in `ExecStart`, then
`sudo systemctl daemon-reload && sudo systemctl restart strata-bitlamas`. A 254K-token prompt takes ~12 min to read
(~360 tok/s, power-capped); decode after it 48 tok/s.

It listens on `0.0.0.0:8080` and needs an API key on `/v1/*` (`Authorization: Bearer <key>` or `x-api-key`):

- Web page (chat): `http://<server-ip>:8080/` (or `http://p100-box:8080/` where the name
  resolves). The page loads without the key; put the key in its settings ("API key") once per browser.
- API: `http://<server-ip>:8080/v1` (OpenAI), the same base for Anthropic's `/v1/messages`.
- The key: `~/.config/strata/bitlamas.env` (mode 600, one line `STRATA_API_KEY=...`; systemd's `EnvironmentFile=`).
  New key: edit that line, then `sudo systemctl restart strata-bitlamas`.
- Local only again: `--host 127.0.0.1` in `ExecStart`, `sudo systemctl daemon-reload && sudo systemctl restart
  strata-bitlamas`. No key on the LAN: remove `EnvironmentFile=` and add `--no-api-key` (the server warns; anyone on
  the LAN can then use it).
- No firewall is active (`ufw` inactive); nothing else is needed for the LAN.

```sh
sudo systemctl enable --now strata-bitlamas.service     # start now and at every boot
journalctl -u strata-bitlamas -f                        # its output; the engine log is ~/projects/strata/strata-bitlamas.log
sudo systemctl disable --now strata-bitlamas.service    # back to off
```

## 8b. Pictures (vision, since 2026-11-07)

The served 256K config has a `"vision"` section: the CPU picture reader `build-vision/bin/strata-vision` (Strata's
`tools/vision`, llama.cpp `3cf03257` as pinned by setup.py, built with
`cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$HOME/projects/llama.cpp-strata-pin -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=OFF`),
the encoder `~/models/strata-vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` (ggml-org, revision 91fb0035, sha256 b2e9b5e4...;
the BF16 reference beside it, ISTA-DASLab ed59f920, sha256 b1a82259..., as setup.py pins it), `"threads": 16`,
`"max_tokens": 3600` (a picture larger than 2560x1440's pixel area is scaled down to it, aspect kept; smaller ones are
not scaled up). The server adds `--vision` to the engine (an image-position table, 3 MiB per GPU: 6505 instead of 6507
expert slots). The reader lives in RAM (1.3 GiB resident, 0% CPU when idle) and takes 16 threads only while it reads.
Measured (`p100-results/vision`): a 4K picture -> 3600 tokens in ~80 s on the CPU, 3638-token prompt 14.7 s, 98 s to the
answer; a 1024x768 chart -> 768 tokens in ~7.6 s. Text output with `--vision` byte-identical (yardstick md5 989d6f4f).
Pictures off again: remove the `"vision"` section and `--vision` from the config's args, restart the service.

A Stop while a picture is read (fixed 2026-11-07 in `serve/server.py`, `p100-results/vision/CANCEL.md`):
- **Another request comes:** the reader is ended and started again (~1 s), and that request runs. Text after a stopped
  1440p picture took 1.9 s, against 73 s before.
- **A request with the same picture comes, or none comes:** the picture is read to the end and kept for the follow-up.
- If a follow-up asks about a picture whose read was ended, the picture is read again from the start (~80 s for 1440p).
- A Stop while a long prompt is read (fixed 2026-11-07 in the engine, `p100-results/engine-stop/STOP.md`): every card
  checks it before each chunk and between layers, so the next request runs about 1 s after the Stop (18-20 s before).
  A prompt whose read was stopped is read again from the start when it is sent again.
- Repro: `.venv/bin/python p100-results/tools/cancel_repro.py --out <file.json> [--cases s1,s2,...]`.

## 9. Rollback

- Engine and tools: `git -C ~/projects/strata checkout main` (upstream as cloned) and rebuild (section 4); every change
  of this effort is on branch `p100-bitlamas-iq4nl`. The new code is opt-in (`STRATA_ARENA_NUMA`); the stock config
  `p100-configs/strata-bitlamas-stock-2gpu-32k-int8.json` runs the default path.
- The engine's Stop during a prompt read (commit `395fcb23`): turn it off with `"STRATA_STOP_STAGES": "0"` and
  `"STRATA_STOP_LAYERS": "0"` in the config's `env`, or run the engine from before it, `build-p100/strata.before-stop-fix`
  (copy it over `build-p100/strata`), then `sudo systemctl restart strata-bitlamas`.
- Sub-agents and the conversation cache (served since 2026-11-14, commit `9ca96758`): `"STRATA_SPLIT_BORROW": "1"` in
  the config's `env` and `--prompt-cache-spread` in its `args`. Remove either line to turn it off, or run the engine
  from before the patch, `build-p100/strata.before-split-borrow` (copy it over `build-p100/strata`), then
  `sudo systemctl restart strata-bitlamas`. Measurements: `p100-results/split-borrow/BORROW.md`,
  `p100-results/prompt-cache-spread/SPREAD.md`.
- Cache limits (served since 2026-11-15): `--prompt-cache 12 --conversation-cache-slots 16 --conversation-cache-mib
  16384` in the config's `args`. Remove the six entries for the defaults (6 checkpoints, 4 parked conversations,
  8192 MiB), then `sudo systemctl restart strata-bitlamas`. Measurement: `p100-results/cache-limits/LIMITS.md`.
- The experimental speed projection (served since 2026-11-15, off unless a request sends
  `"experimental_speed_projection": true`): remove `--control-vector-scaled <file>:1.0 --control-vector-layer-range 4
  44 --cvec-mode project --cvec-dir per-layer` from the config's `args` and `experimental_speed_projection` from its
  `sampling`, then restart. To make it on for every request instead, set that `sampling` value to `true`.
  Measurement: `p100-results/projection/PROJECTION.md`.
- Service: `sudo systemctl disable --now strata-bitlamas.service && sudo rm /etc/systemd/system/strata-bitlamas.service`.
- System: section 2 (power cap: keep it), 2b (IOMMU passthrough / kdump: steps to undo there), CUDA header patch
  (section 3: `math_functions.h.orig`).

## 10. The NAS share and the backup (2026-11-08)

- `\\NAS-HOST\nas-share` (Unraid, <nas-ip>) is mounted at `/mnt/nas-share` on first access:
  `cifs-utils` installed; `/etc/fstab` line with `credentials=/root/.smbcredentials-nas-share` (root, mode 600, made by
  the owner), `nofail` and `x-systemd.automount`, so boot and SSH never wait for the NAS. The fstab from before is
  `/etc/fstab.before-nas-share`. Guests can only read the share.
- This machine's backup is `/mnt/nas-share/p100-box/` (other boxes back up to their own folders
  on the same share; never write there). `p100-results/tools/backup-to-nas.sh stage|push|verify|all`:
  - stage: checksums and archives on `/mnt/scratch/nas-staging`.
  - push: rsync, resumable.
  - verify: reads every file back.
  The first run (2026-11-08) ran as the transient unit `nas-backup` (`journalctl -u nas-backup`).
- Speed: the link is 100 Mb/s (the switch or router port offers only 10/100; the RTL8111 does 1000), so ~9.6 MB/s,
  about 3 h for the 105 GB of models and 3 h more to verify.
- Not backed up: the API key and the CUDA 12.8.1 installer (see the README in the backup folder).
