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
# N and DATA_SEED parameterize which rung is replicated. At N=144 the row draw is the whole pool
# whatever the shuffle, so DATA_SEED is irrelevant and --seed alone varies only the run; at any
# smaller N it is NOT (seeds 0 and 7 share 15 of their 47 rows), so DATA_SEED pins the rows and
# --seed varies the run. TAG names the family in the artifacts.
SEEDS="${SEEDS:-1 2 3}"
N="${N:-144}"
DATA_SEED="${DATA_SEED:-}"
TAG="${TAG:-$N}"
DS_ARG=""; [ -n "$DATA_SEED" ] && DS_ARG="--data-seed $DATA_SEED"
# PREFIX restricts the corpus to rows whose id starts with it, written to its own data file. This is
# how the 47-fold rung is replicated exactly: batch 1 is the 55 't' windows, of which the frozen
# split holds 8 as val and 47 as train, so PREFIX=t + N=47 trains on precisely the rows the original
# ft47 rung saw -- and, as at N=144, the drawn set is then the whole pool whatever the shuffle, so
# the data is fixed by construction and --seed varies only the run.
DATA_ARG=""
if [ -n "${PREFIX:-}" ]; then
  DF="$PWD/wake/teacher-folds-$PREFIX.json"
  "$PY" - "$PREFIX" "$DF" <<'PYEOF'
import json,pathlib,sys
pre,out=sys.argv[1],sys.argv[2]
rows=json.loads(pathlib.Path("wake/teacher-folds.json").read_text())
sub=[r for r in rows if r["id"].startswith(pre)]
pathlib.Path(out).write_text(json.dumps(sub,ensure_ascii=False,indent=1))
n_val=len([r for r in sub if r.get("split")=="val"])
print(f"[subset] prefix {pre!r}: {len(sub)} rows ({len(sub)-n_val} train / {n_val} val) -> {out}")
PYEOF
  DATA_ARG="--data $DF"
fi
[ "${APPEND:-0}" = 1 ] || : > "$LOG"
say(){ printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

say "seed-spread: n=$N tag=$TAG ${DS_ARG:-(no data-seed: rows follow --seed)}, 3 epochs, seeds: $SEEDS"
for s in $SEEDS; do
  out="wake/fold-lora-${TAG}s$s"; var="ft${TAG}s$s-e3"
  if [ -d "$out/ep3" ]; then say "seed $s: adapter exists, skipping train"; else
    say "seed $s: train n=$N epochs=3 $DS_ARG -> $out"
    # --out is resolved against train_fold.py's OWN directory (outdir = HERE / a.out), NOT the cwd.
    # Passing the repo-relative "wake/fold-lora-144s1" therefore wrote the adapter to
    # wake/wake/fold-lora-144s1 and the generate step died 17 minutes later on a missing
    # adapter_config.json. Strip the prefix for the flag; every other path here is cwd-relative.
    "$PY" wake/train_fold.py --n "$N" --epochs 3 --seed "$s" $DS_ARG $DATA_ARG --out "${out#wake/}" 2>&1 | tee -a "$LOG"
    [ -d "$out/ep3" ] || { say "seed $s: FAILED -- no $out/ep3 after training"; exit 1; }
  fi
  say "seed $s: generate rung $var from $out/ep3 over the 40 eval windows"
  "$PY" wake/make_folds.py --rung --adapter "$out/ep3" --variant "$var" 2>&1 | tail -45 | tee -a "$LOG"
  say "seed $s: score $var (budget 157, 6 foreign donors)"
  "$PY" wake/condent.py --pairs "$var" --budget 157 --n-foreign 6 \
      --out "wake/recs-$var.json" 2>&1 | tail -25 | tee -a "$LOG"
done
say "seed-spread: all seeds done; rungs on disk: $(ls wake/recs-ft${TAG}s*-e3.json 2>/dev/null | tr '\n' ' ')"
