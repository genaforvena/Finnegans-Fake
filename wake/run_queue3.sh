#!/usr/bin/env bash
# The first BPE-4096 run predates the save-final fix, so its overfit weights —
# the playable ones — were never written to disk. Re-run it.
set -u
cd "$(dirname "$0")/.."
while pgrep -f 'wake/run_queue.sh' >/dev/null; do sleep 20; done
echo "=== BPE-4096 REDO (keeps final weights) $(date -u +%FT%TZ) ==="
~/.venv-ai/bin/python wake/train_scratch.py --vocab 4096 --block 256 --batch 32 \
    --iters 6000 --out wake-bpe4096
echo "=== QUEUE3 DONE $(date -u +%FT%TZ) ==="
