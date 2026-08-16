#!/usr/bin/env bash
# Live data pane for the `wake` window. Refreshes what the window is FOR: which
# variant is training, where its loss is, and what the last finished model says.
# Everything here is read-only — it never writes into the run's artefacts.
#
# THE FRAME MUST FIT THE PANE. A frame taller than its window scrolls its own
# head off, and `tmux capture-pane -p` is VISIBLE-ONLY: the liveness watchdogs
# hash what is on screen, not what was printed. This dash used to render 29+
# lines into a 27-row pane, so its clock — the only per-frame-changing token —
# never entered the hash, and the surviving tail changed only when nvidia-smi's
# temp digit moved. Result: a pane re-rendering every 30s for 7h read FROZEN,
# twice flagged and twice auto-"recovered" (2026-08-16 01:07, 02:17). Two rules
# come out of that and both are load-bearing:
#   1. size every section against the LIVE pane height, never a constant;
#   2. the changing token goes in the LAST line, the one that cannot scroll off.
cd "$(dirname "$0")/.." || exit 1
W=$(cd "$(dirname "$0")" && pwd)

frame=0
while :; do
  frame=$(( frame + 1 ))

  # Live geometry — the pane can be resized under us, so re-read it every frame.
  rows=$(tput lines 2>/dev/null)
  [ -n "$rows" ] && [ "$rows" -gt 8 ] 2>/dev/null || rows=27
  cols=$(tput cols 2>/dev/null)
  [ -n "$cols" ] && [ "$cols" -gt 24 ] 2>/dev/null || cols=80
  # 1 row for the footer, 1 so the cursor's own landing row never forces a scroll.
  budget=$(( rows - 2 ))
  rule=$(( cols < 78 ? cols : 78 ))
  wrap=$(( cols - 2 < 76 ? cols - 2 : 76 ))

  body=$(
    printf '\033[1mFINNEGANS FAKE\033[0m  a post-mortem, written while the patient trains\n'
    printf '%.0s─' $(seq 1 "$rule"); echo

    if pgrep -f 'train_scratch\.py' >/dev/null; then
      printf '\033[32mTRAINING\033[0m  '
      pgrep -af 'train_scratch\.py' | head -1 | grep -o -- '--out [a-z0-9-]*' | head -1
    else
      printf '\033[33midle — no run in flight\033[0m\n'
    fi

    echo
    echo "current run:"
    # cut, not just tail: sweep.log carries absolute paths >100 chars, and a line
    # that WRAPS costs two rows while the budget below counts one. Every plain
    # (ANSI-free) section is clipped to the pane width so rows == lines.
    grep -aE '^(===|iter|best|final)' "$W/sweep.log" 2>/dev/null | tail -4 |
      sed 's/^/  /' | cut -c1-"$cols"

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
    done | cut -c1-"$cols"

    echo
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /' | cut -c1-"$cols"
  )

  # The sample is the elastic section: it gets exactly the rows nothing else
  # claimed. Below two spare rows it is dropped entirely rather than shown as a
  # single orphan line that pushes the head off the top.
  used=$(printf '%s\n' "$body" | wc -l)
  rem=$(( budget - used - 2 ))          # 2 = the blank + the "last sample:" label
  sample=""
  if [ -f "$W/last-sample.txt" ] && [ "$rem" -ge 1 ]; then
    sample=$(fold -s -w "$wrap" "$W/last-sample.txt" | tail -n "$rem" | sed 's/^/  /')
  fi

  clear
  {
    printf '%s\n' "$body"
    [ -n "$sample" ] && { echo; echo "last sample:"; printf '%s\n' "$sample"; }
  } | head -n "$budget"                 # backstop: the body can never overrun
  # THE LIVENESS LINE. Last row, always printed, and every field in it moves on
  # every render — this is what a visible-only capture hash sees change.
  printf '\033[2m── %s   frame %d   %dx%d ──\033[0m\n' \
    "$(date -u +%H:%M:%SZ)" "$frame" "$cols" "$rows"

  sleep 30
done
