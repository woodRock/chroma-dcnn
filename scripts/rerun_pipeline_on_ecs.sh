#!/usr/bin/env bash
# Rerun the full pretrain -> fine-tune -> saliency pipeline on an ECS GPU, for when a paper
# revision changes code or configs and the reported numbers/figures need regenerating.
#
# What this does, and why each step is here (this is the sequence a human had to rediscover
# by hand the first time, including the mistakes):
#   1. `task sync` pushes this repo's working tree (commit first; uncommitted changes need
#      --allow-dirty, see `task sync --help`) to the shared ECS volume. It refuses if the
#      remote copy has uncommitted changes of its own (from a previous manual run, a
#      different repo state on the server, etc.) -- resolve those by hand before rerunning
#      this script rather than stashing/discarding blindly.
#   2. If data/pretraining/raw/ doesn't exist on the server, 01_download_pretrain_data.py
#      re-downloads it there (MoNA from committed parts, MassBank from GitHub -- this alone
#      can take 15-30 minutes for ~140k MassBank records over the shared volume).
#   3. `task submit` queues steps 2-4 as ONE chained job (`&&`, `set -e`) so they run in
#      order on a single GPU and a failure partway stops the rest rather than silently
#      continuing on stale intermediate output.
#   4. `task wait` blocks until it finishes (success, failure, or timeout) -- don't poll
#      `task info`/`task logs` in a loop; this is what `task wait` is for.
#   5. `task fetch` copies results/ and figs/*.png and *.pdf back so `taura publish
#      --no-commit` (see AGENTS.md) can sync them into the paper, and `taura check` can see
#      the new run.
#
# Usage:
#   scripts/rerun_pipeline_on_ecs.sh [HOST]          # HOST default: cuda-small0
#   scripts/rerun_pipeline_on_ecs.sh --skip-download  # corpus already present on the server
#
# After it finishes:
#   cd ~/Desktop/chroma-dcnn && taura check              # see what changed
#   taura publish --no-commit                             # sync into the paper mid-revision
#   taura publish                                          # ... or commit for real once the
#                                                           #     paper repo is clean

set -euo pipefail

HOST=cuda-small0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --skip-download) SKIP_DOWNLOAD=1 ;;
    *) HOST="$arg" ;;
  esac
done

REPO_NAME=chroma-dcnn
# "<remote_root>/<repo name>" per ~/.gpu-spooler/hosts.toml's [groups.ecs] remote_root
# (git repos submitted from this laptop run from there) -- update if that changes.
REMOTE_DIR="/vol/ecrg-solar/woodj4/${REPO_NAME}"

echo "==> Checking this repo is committed (task sync needs that, or --allow-dirty)"
if [ -n "$(git status --porcelain)" ]; then
  echo "    Working tree is dirty. Commit, or edit this script to pass --allow-dirty to sync." >&2
  exit 1
fi

echo "==> Syncing repo to $HOST"
task sync

if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
  echo "==> Checking whether the pretraining corpus needs (re)downloading on $HOST"
  if ! task exec -H "$HOST" -- "test -s ${REMOTE_DIR}/data/pretraining/raw/combined_ei_spectra.json"; then
    echo "    Not present -- downloading (can take 15-30 min over the shared volume) ..."
    task exec -H "$HOST" -- "cd ${REMOTE_DIR} && python3 scripts/01_download_pretrain_data.py" || {
      echo "    task exec may have hit its own client-side timeout; the download keeps running" \
           "detached on the server regardless -- check with:" >&2
      echo "      task exec -H $HOST -- \"pgrep -fa 01_download_pretrain_data\"" >&2
      exit 1
    }
  else
    echo "    Already present, skipping (pass no --skip-download to force a recheck)."
  fi
fi

echo "==> Submitting pretrain -> fine-tune -> saliency as one chained GPU job (any free GPU)"
TASK_ID=$(task submit -q -n "chroma-revision-rerun" -- bash -lc \
  "cd ${REMOTE_DIR} && set -e && \
   python3 scripts/02_preprocess_pretrain_data.py && \
   python3 scripts/03_pretrain_chroma.py && \
   python3 scripts/04_finetune_evaluate.py && \
   python3 scripts/11_saliency.py")
echo "    Queued task $TASK_ID. Waiting for it to finish (this is the long part: ~10 min"
echo "    pretraining + fine-tuning/CV + saliency on a GPU) ..."

task wait "$TASK_ID"

echo "==> Fetching results and figures back"
task fetch results figs/gc-ms.png figs/synthetic-dataset.png figs/saliency-heatmaps.png \
             figs/saliency-1d.png figs/saliency-3d.png ledger/runs

echo "==> Done. Next: taura check, then taura publish [--no-commit]"
