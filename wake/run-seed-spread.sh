#!/usr/bin/env bash
# run-seed-spread.sh — how big is a rung's RUN-TO-RUN spread on the content metric?
#
# Every comparison this lane has published rests on differences of ~0.002 nats/char between rungs
# trained ONCE each. Until 2026-08-18 train_fold.py never seeded torch, so the LoRA init, the
# dropout masks and the batch order were an unseeded draw on every run while trainlog.json recorded
# "seed": 0 for all of them. No two rungs were ever replicates, and the spread that would tell us
# whether -0.0018 +/- 0.0022 means anything had never been measured.
#
# n=144 is deliberate: pool[:144] takes ALL 144 training rows regardless of the shuffle, so the DATA
# is identical across seeds by construction and the only thing varying is the training draw. Same
# 40 frozen eval windows, same 157-token budget, same 6 foreign donors as db416ad, so the spread is
# directly comparable to the gap it has to be read against.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venv-ai/bin/python"
export CONDENT_WINDOWS="$PWD/wake/eval-windows.json" CONDENT_FOLDS="$PWD/wake/eval-folds.json"
LOG="$PWD/wake/seed-spread.log"
SEEDS="${SEEDS:-1 2 3}"
: > "$LOG"
say(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

say "seed-spread: n=144 (all rows, data fixed by construction), 3 epochs, seeds: $SEEDS"
for s in $SEEDS; do
  out="wake/fold-lora-144s$s"; var="ft144s$s-e3"
  if [ -d "$out/ep3" ]; then say "seed $s: adapter exists, skipping train"; else
    say "seed $s: train n=144 epochs=3 -> $out"
    # --out is resolved against train_fold.py's OWN directory (outdir = HERE / a.out), NOT the cwd.
    # Passing the repo-relative "wake/fold-lora-144s1" therefore wrote the adapter to
    # wake/wake/fold-lora-144s1 and the generate step died 17 minutes later on a missing
    # adapter_config.json. Strip the prefix for the flag; every other path here is cwd-relative.
    "$PY" wake/train_fold.py --n 144 --epochs 3 --seed "$s" --out "${out#wake/}" 2>&1 | tee -a "$LOG"
    [ -d "$out/ep3" ] || { say "seed $s: FAILED -- no $out/ep3 after training"; exit 1; }
  fi
  say "seed $s: generate rung $var from $out/ep3 over the 40 eval windows"
  "$PY" wake/make_folds.py --rung --adapter "$out/ep3" --variant "$var" 2>&1 | tail -45 | tee -a "$LOG"
  say "seed $s: score $var (budget 157, 6 foreign donors)"
  "$PY" wake/condent.py --pairs "$var" --budget 157 --n-foreign 6 \
      --out "wake/recs-$var.json" 2>&1 | tail -25 | tee -a "$LOG"
done
say "seed-spread: all seeds done; rungs on disk: $(ls wake/recs-ft144s*-e3.json 2>/dev/null | tr '\n' ' ')"
