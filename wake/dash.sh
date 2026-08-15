#!/usr/bin/env bash
# Live data pane for the `wake` window. Refreshes what the window is FOR: which
# variant is training, where its loss is, and what the last finished model says.
# Everything here is read-only — it never writes into the run's artefacts.
cd "$(dirname "$0")/.." || exit 1
W=$(cd "$(dirname "$0")" && pwd)

while :; do
  clear
  printf '\033[1mFINNEGANS FAKE\033[0m  %s   a post-mortem, written while the patient trains\n' "$(date -u +%H:%M:%SZ)"
  printf '%.0s─' $(seq 1 78); echo

  if pgrep -f 'train_scratch\.py' >/dev/null; then
    printf '\033[32mTRAINING\033[0m  '
    pgrep -af 'train_scratch\.py' | head -1 | grep -o -- '--out [a-z0-9-]*' | head -1
  else
    printf '\033[33midle — no run in flight\033[0m\n'
  fi

  echo
  echo "current run:"
  grep -aE '^(===|iter|best|final)' "$W/sweep.log" 2>/dev/null | tail -6 | sed 's/^/  /'

  echo
  echo "finished variants (train / best-val / params):"
  for d in "$W"/wake-*/; do
    [ -f "$d/trainlog.json" ] || continue
    ~/.venv-ai/bin/python - "$d" <<'PY' 2>/dev/null
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); j = json.loads((d / "trainlog.json").read_text())
ft, bv = j.get("final_train"), j.get("best_val")
print(f"  {d.name:22s} train {ft:.3f}  best-val {bv:.3f}  {j['params']/1e6:.1f}M"
      if ft else f"  {d.name:22s} (incomplete)")
PY
  done

  echo
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /'

  if [ -f "$W/last-sample.txt" ]; then
    echo; echo "last sample:"; fold -s -w 76 "$W/last-sample.txt" | tail -8 | sed 's/^/  /'
  fi

  sleep 30
done
