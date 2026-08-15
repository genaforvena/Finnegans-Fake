#!/usr/bin/env bash
# Chain: wait for the running from-scratch char run to EXIT, assert it produced weights,
# then LoRA the Base model. The wait is on the process, and the go/no-go is on the ARTIFACT
# — a directory existing is not a finished run (see POSTMORTEM: the download waiter).
set -u
cd "$(dirname "$0")/.."
PID="${1:?usage: run_lora_after_scratch.sh <pid-of-train_scratch>}"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
echo "=== scratch pid $PID exited $(date -u +%FT%TZ) ==="
CK=wake/wake-char257-long/final
[ -d "$CK" ] || { echo "NO final weights at $CK — scratch run did not finish cleanly; NOT starting LoRA"; exit 1; }
BASE="$HOME/models/Qwen3.5-0.8B-Base"
SZ=$(stat -c%s "$BASE/model.safetensors-00001-of-00001.safetensors")
[ "$SZ" -gt 1700000000 ] || { echo "base weights only $SZ bytes — refusing"; exit 1; }
echo "=== LoRA base=$BASE $(date -u +%FT%TZ) ==="
exec ~/.venv-ai/bin/python wake/train_lora.py --base "$BASE" --out wake-lora-base --epochs 3
