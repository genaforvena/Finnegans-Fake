#!/usr/bin/env bash
# Bet 2, two tokenizations, same architecture budget. Sequential so the 3060
# is never shared and the wall-clock numbers stay comparable.
set -u
cd "$(dirname "$0")/.."
PY=~/.venv-ai/bin/python
echo "=== BPE-4096 (block 256) $(date -u +%FT%TZ) ==="
$PY wake/train_scratch.py --vocab 4096 --block 256 --batch 32 --iters 6000 --out wake-bpe4096
echo "=== CHAR-257 (block 512) $(date -u +%FT%TZ) ==="
$PY wake/train_scratch.py --vocab 257 --block 512 --batch 24 --iters 6000 --out wake-char257
echo "=== DONE $(date -u +%FT%TZ) ==="
