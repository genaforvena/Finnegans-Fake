#!/usr/bin/env bash
# Variant sweep. Waits for the first pair to finish so the 3060 is never shared.
set -u
cd "$(dirname "$0")/.."
PY=~/.venv-ai/bin/python
while pgrep -f 'wake/run_both.sh' >/dev/null; do sleep 20; done
echo "=== BPE-8192 (coarser pieces, expect faster memorisation) $(date -u +%FT%TZ) ==="
$PY wake/train_scratch.py --vocab 8192 --block 256 --batch 32 --iters 3000 --out wake-bpe8192
echo "=== BPE-4096 SMALL+REGULARISED (4L/256d, dropout .35) $(date -u +%FT%TZ) ==="
$PY wake/train_scratch.py --vocab 4096 --block 256 --batch 32 --iters 3000 \
    --layers 4 --heads 4 --embd 256 --dropout 0.35 --out wake-bpe4096-small
echo "=== QUEUE DONE $(date -u +%FT%TZ) ==="
