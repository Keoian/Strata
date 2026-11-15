#!/bin/bash
# Back up this machine's model files, projects and system changes to the Unraid share \\NAS-HOST\nas-share
# (mounted at /mnt/nas-share, see RUNBOOK.md section 10).
#
#   backup-to-nas.sh stage    local only: checksums, projects.tar.zst, the git bundle, the system files (/mnt/scratch/nas-staging)
#   backup-to-nas.sh push     copy models/, the staged files and the scratch results to the share (rsync, resumable)
#   backup-to-nas.sh verify   read every file back from the share and check it against MANIFEST.sha256
#
# Not backed up: the API key (~/.config/strata/bitlamas.env: anyone who can read the share could read it), the CUDA
# 12.8.1 installer on /mnt/scratch/downloads (a public NVIDIA download) and its extract.
set -euo pipefail
ST=/mnt/scratch/nas-staging
DEST=/mnt/nas-share/p100-box
NICE="nice -n 19 ionice -c3"
SYSTEM_FILES=(/etc/fstab /etc/fstab.before-nas-share /etc/default/grub /etc/default/grub.d/zz-p100-strata.cfg
  /etc/p100-power-cap.conf /etc/systemd/system/p100-power-cap.service /etc/systemd/system/p100-boot-check.service
  /etc/systemd/system/strata-bitlamas.service /usr/local/sbin/p100-power-cap /usr/local/sbin/p100-boot-check
  /etc/systemd/system/strata-bitlamas.service.d/p100-power-cap.conf
  /etc/systemd/system/nvidia-persistenced.service.d/p100-power-cap.conf
  /usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h
  /usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h.orig)
say() { echo "== $(date +%T) $*"; }

stage() {
  mkdir -p $ST/system
  say "checksums of ~/models (paths as on the share)"
  (cd ~ && find models -type f ! -name '*.lock' -print0 | sort -z | $NICE xargs -0 -n1 -P6 sha256sum) | sort -k2 > $ST/models.sha256
  say "the GGUF shards against the model repo's SHA256SUMS"
  while read -r h f; do
    got=$(grep -F " models/bitlamas-Qwen3.8-Flash-Next-GGUF/${f#\*}" $ST/models.sha256 | cut -d' ' -f1)
    [ "$got" = "$h" ] && echo "   OK ${f#\*}" || { echo "   MISMATCH ${f#\*}"; exit 1; }
  done < <(grep "Q4_K_XL-IQ4NL/" ~/models/bitlamas-Qwen3.8-Flash-Next-GGUF/SHA256SUMS)
  say "projects.tar.zst (every repo, worktree and build, with permissions and links)"
  $NICE tar -C ~ -cf - projects | $NICE zstd -T8 -3 -q -f -o $ST/projects.tar.zst
  say "strata-all-branches.bundle"
  git -C ~/projects/strata bundle create $ST/strata-all-branches.bundle --all 2>&1 | tail -1
  say "system files"
  local have=()
  for f in "${SYSTEM_FILES[@]}"; do [ -e "$f" ] && have+=("$f"); done
  sudo tar -cpf $ST/system-files.tar "${have[@]}" 2>/dev/null
  sudo chown "$(id -u):$(id -g)" $ST/system-files.tar
  for f in "${have[@]}"; do sudo cp --parents -p "$f" $ST/system/; done
  sudo chown -R "$(id -u):$(id -g)" $ST/system
  dpkg-query -W -f '${Package} ${Version}\n' > $ST/system/dpkg-packages.txt
  write_readme
  say "the manifest"
  { cat $ST/models.sha256
    (cd $ST && sha256sum README.md projects.tar.zst strata-all-branches.bundle system-files.tar && find system -type f -print0 | sort -z | xargs -0 sha256sum)
    (cd /mnt/scratch && find sass-diff-dp4a -type f -print0 | sort -z | xargs -0 sha256sum | sed 's#  #  scratch/#')
  } > $ST/MANIFEST.sha256
  say "staged: $(wc -l < $ST/MANIFEST.sha256) files"; du -sh $ST/projects.tar.zst $ST/strata-all-branches.bundle $ST/system-files.tar
}

write_readme() {
  cat > $ST/README.md <<EOF
# p100-box backup ($(date +%F))

2x Xeon E5-2696 v3, 128 GB DDR3, 2x Tesla P100 16 GB (150 W), Ubuntu 26.04.1. Strata serving bitlamas'
Qwen3.8-Flash-Next Q4_K_XL-IQ4NL at 256K context. How it was set up: RUNBOOK.md inside the strata repo
(branch p100-bitlamas-iq4nl, also on the private GitHub repo <private backup repo>).

| Here | What | Restore |
|---|---|---|
| models/ | ~/models as it was: the GGUF (4 shards), the Strata pack, the MTP draft layer, the vision encoders | copy back to ~/models |
| projects.tar.zst | ~/projects: the strata repo with its worktrees and builds, llama.cpp at Strata's pin, ref tools | tar -C ~ --zstd -xf projects.tar.zst |
| strata-all-branches.bundle | every branch of the strata repo | git clone -b p100-bitlamas-iq4nl strata-all-branches.bundle strata |
| system-files.tar, system/ | /etc/fstab, the grub drop-in, the power-cap and boot-check scripts and units, the service unit, the patched CUDA header (and the original); dpkg package list | sudo tar -C / -xpf system-files.tar (read RUNBOOK first) |
| scratch/sass-diff-dp4a | SASS comparisons from the P100 kernel work | - |
| MANIFEST.sha256 | sha256 of every file here | sha256sum -c MANIFEST.sha256 |

Not here: the API key (~/.config/strata/bitlamas.env; make a new one with strata launch --new-key or a new random
value), the CUDA 12.8.1 installer (cuda_12.8.1_570.124.06_linux.run from NVIDIA's archive).
EOF
}

push() {
  mountpoint -q /mnt/nas-share || ls /mnt/nas-share >/dev/null   # the automount mounts it on first access
  mkdir -p $DEST/models $DEST/scratch
  local R="rsync -rtl --modify-window=2 --partial --info=stats1"
  for d in strata-packs strata-vision strata-mtp bitlamas-Qwen3.8-Flash-Next-GGUF; do
    say "models/$d ($(du -sh ~/models/$d | cut -f1))"
    $NICE $R --exclude '*.lock' ~/models/$d/ $DEST/models/$d/
  done
  say "staged files"
  $NICE $R $ST/README.md $ST/projects.tar.zst $ST/strata-all-branches.bundle $ST/system-files.tar $ST/MANIFEST.sha256 $DEST/
  $NICE $R $ST/system/ $DEST/system/
  say "scratch/sass-diff-dp4a"
  $NICE $R /mnt/scratch/sass-diff-dp4a/ $DEST/scratch/sass-diff-dp4a/
  say "pushed"; du -sh $DEST
}

verify() {
  say "reading every file back from the share"
  (cd $DEST && $NICE sha256sum --quiet -c MANIFEST.sha256) && say "all $(wc -l < $DEST/MANIFEST.sha256) files match"
}

case "${1:-}" in
  stage) stage ;; push) push ;; verify) verify ;;
  all) stage; push; verify ;;
  *) echo "usage: $0 stage|push|verify|all"; exit 2 ;;
esac
