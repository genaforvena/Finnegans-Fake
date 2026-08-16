#!/usr/bin/env python3
"""Paired difference on SIGNAL/novel between two condent runs over the same windows.

The aggregate condent prints is a mean over pairs with its own spread; comparing
two rungs by their aggregates throws away the pairing, and the windows differ
enormously in how much signal any summary can carry. Pairing by window id is what
makes a 0.003 nats/char difference readable at all.

Records come from `condent.py --out`. Only windows present in BOTH runs are used,
and the count of what each side dropped is printed rather than absorbed — a rung
whose folds fall below the shared budget is scored on fewer windows, and that
mildly favours it.

  python paired.py a.json b.json          # b - a
"""
import argparse, json, pathlib, statistics, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import condent


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    recs = d["records"] if isinstance(d, dict) else d
    out = {}
    for r in recs:
        v = condent.deltas(r).get("signal.novel")
        if v is not None:
            # names are "<variant>/<window id>"; the window is the pairing key
            out[r["name"].split("/")[-1]] = v
    return out, d


def describe(v, label):
    if not v:
        print(f"  {label:<28} —")
        return
    m = statistics.mean(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    sem = sd / len(v) ** 0.5 if len(v) > 1 else 0.0
    pos = sum(1 for x in v if x > 0)
    print(f"  {label:<28} {m:+.4f} ± {sem:.4f} sem  ({m/sem:.1f} sem, {pos}/{len(v)} positive)"
          if sem else f"  {label:<28} {m:+.4f}  (n={len(v)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    a = ap.parse_args()

    A, da = load(a.a)
    B, db = load(a.b)
    la = a.label_a or pathlib.Path(a.a).stem
    lb = a.label_b or pathlib.Path(a.b).stem
    both = sorted(set(A) & set(B))
    print(f"[windows] {la}: {len(A)}  {lb}: {len(B)}  shared: {len(both)}"
          + (f"  (only in {la}: {sorted(set(A)-set(B))})" if set(A) - set(B) else "")
          + (f"  (only in {lb}: {sorted(set(B)-set(A))})" if set(B) - set(A) else ""))
    if isinstance(da, dict) and isinstance(db, dict):
        print(f"[budget]  {la}: {da.get('budget')}  {lb}: {db.get('budget')}")
        if da.get("budget") != db.get("budget"):
            print("  WARNING: different budgets — not comparable")

    print("\nSIGNAL/novel, nats/char:")
    describe([A[k] for k in both], f"{la} (shared windows)")
    describe([B[k] for k in both], f"{lb} (shared windows)")
    print("\nPAIRED:")
    describe([B[k] - A[k] for k in both], f"{lb} − {la}")


if __name__ == "__main__":
    main()
