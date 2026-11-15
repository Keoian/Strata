#!/bin/bash
# open_upstream_pr.sh - opens the STRATA_ARENA_NUMA draft PR on Niko1221/Strata from the owner's account.
#   1. stops unless `gh` is logged in (run `gh auth login` first)
#   2. lists upstream issues/PRs that mention the arena NUMA placement (CONTRIBUTING.md: search first) and stops if
#      any open PR already covers it (re-run with FORCE=1 after reading them)
#   3. forks Niko1221/Strata to the logged-in account (once), pushes branch engine-arena-numa-placement, opens a DRAFT
#      PR with the prepared description (~/projects/strata-pr-numa/PR_BODY.md)
set -euo pipefail
UP=Niko1221/Strata
WT=$HOME/projects/strata-pr-numa
BR=engine-arena-numa-placement
TITLE="[Engine] opt-in STRATA_ARENA_NUMA: place the expert arena per NUMA node on two-socket Linux"
gh auth status >/dev/null 2>&1 || { echo "gh is not logged in: run 'gh auth login' first"; exit 1; }
echo "== upstream PRs/issues mentioning the arena's NUMA placement:"
gh search prs --repo "$UP" "NUMA arena" --limit 20 --json number,title,state,url --jq '.[] | "PR #\(.number) [\(.state)] \(.title) \(.url)"' || true
gh search issues --repo "$UP" "NUMA arena placement" --limit 20 --json number,title,state,url --jq '.[] | "issue #\(.number) [\(.state)] \(.title) \(.url)"' || true
OPEN=$(gh search prs --repo "$UP" "STRATA_ARENA_NUMA OR \"not placed per node\"" --state open --json number --jq 'length' || echo 0)
if [ "${OPEN:-0}" != 0 ] && [ "${FORCE:-0}" != 1 ]; then echo "an open upstream PR mentions this: read it, then re-run with FORCE=1"; exit 2; fi
cd "$WT"
git log --oneline -1 "$BR"
gh repo fork "$UP" --remote --remote-name fork --clone=false 2>/dev/null || true
git remote get-url fork >/dev/null 2>&1 || git remote add fork "https://github.com/$(gh api user --jq .login)/Strata.git"
git push -u fork "$BR"
gh pr create --repo "$UP" --draft --base main --head "$(gh api user --jq .login):$BR" --title "$TITLE" --body-file "$WT/PR_BODY.md"
