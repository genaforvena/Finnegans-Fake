#!/usr/bin/env bash
# Full sweep, post label-shift fix. Sequential: the 3060 is never shared, so the
# wall-clock and loss numbers stay comparable across variants.
set -u
cd "$(dirname "$0")/.."
PY=~/.venv-ai/bin/python
run () { echo "=== $1 $(date -u +%FT%TZ) ==="; shift; $PY wake/train_scratch.py "$@"; }
run "BPE-4096"        --vocab 4096 --block 256 --batch 32 --iters 6000 --out wake-bpe4096
run "BPE-8192"        --vocab 8192 --block 256 --batch 32 --iters 6000 --out wake-bpe8192
run "CHAR-257"        --vocab 257  --block 512 --batch 24 --iters 6000 --out wake-char257
run "BPE-4096-SMALL"  --vocab 4096 --block 256 --batch 32 --iters 6000 \
                      --layers 4 --heads 4 --embd 256 --dropout 0.35 --out wake-bpe4096-small
echo "=== SWEEP DONE $(date -u +%FT%TZ) ==="
