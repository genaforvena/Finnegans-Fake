#!/usr/bin/env python3
"""Paired difference on SIGNAL/novel between two RUNGS, each replicated.

The aggregate condent prints is a mean over pairs with its own spread; comparing
two rungs by their aggregates throws away the pairing, and the windows differ
enormously in how much signal any summary can carry. Pairing by window id is what
makes a 0.003 nats/char difference readable at all.

But pairing is not the only term. Measured in `condent-results-2026-08-18-seed-spread.md`:
the TRAINING draw alone is sd 0.00140 on a 144 rung and sd 0.00178 on a 47 rung,
so a difference between two singly-trained rungs carries +/-0.00226 from the
training draw before a single window is sampled. Folded with the window sem of
about 0.0022 the honest floor on a one-run-vs-one-run comparison is +/-0.0032 --
against effects this lane cares about of 0.002 to 0.008. That is why `4a07481` and
`db416ad`'s "144 folds lose to 47" (-0.0018 +/- 0.0022, read from one training per
side) was withdrawn in `8e183de`: with both sides replicated the difference is
-0.0002 and its sign is 6 of 12.

So this refuses to print a between-rung delta unless BOTH sides carry at least
MIN_TRAININGS distinct trainings, and says why instead. The count is over distinct
TRAININGS, not over files: two epochs of one run are one draw. Provenance comes
from `rung-provenance.json` (written by make_folds.py at generation time); a rung
this tool cannot trace is refused rather than assumed -- missing evidence is not a
pass. A rung with no training draw at all (the untrained base) is exempt, because
there is nothing to replicate.

  python paired.py --a recs-ft47-e3.json recs-ft47rs1-e3.json recs-ft47rs2-e3.json \
                   --b recs-ft144b-e3.json recs-ft144s1-e3.json recs-ft144s2-e3.json
  python paired.py a.json b.json          # legacy 1-vs-1 form: refused, by design
  python paired.py --test                 # self-test: drives the gate red AND green
"""
import argparse, itertools, json, pathlib, statistics, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import condent

HERE = pathlib.Path(__file__).resolve().parent
PROVENANCE = HERE / "rung-provenance.json"

# 3 a side divides the training term by sqrt(3), from +/-0.00226 to +/-0.0013 --
# below the smallest effect (0.002) this lane reports. Two a side leaves 0.0016,
# which does not clear it, and a two-run sd is not a spread measurement anyway
# (seed-spread.md: three agreeing runs were falsified by the fourth).
MIN_TRAININGS = 3
ONE_RUN_FLOOR = 0.0032


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


def provenance():
    if not PROVENANCE.exists():
        return {}
    return json.loads(PROVENANCE.read_text()).get("variants", {})


class Side:
    """One rung: its scored sets, and the distinct trainings behind them."""

    def __init__(self, paths, label, prov):
        self.paths = [str(p) for p in paths]
        self.label = label
        self.sets = []          # (variant, {window: value})
        self.trainings = {}     # training id -> [variants]
        self.untraceable = []   # variants with no provenance entry
        self.no_draw = []       # variants that carry no training draw at all
        for p in self.paths:
            vals, meta = load(p)
            var = meta.get("pairs") if isinstance(meta, dict) else None
            var = var or pathlib.Path(p).stem.replace("recs-", "")
            self.sets.append((var, vals))
            if var not in prov:
                self.untraceable.append(var)
                continue
            t = prov[var].get("training")
            if t is None:
                self.no_draw.append(var)
            else:
                self.trainings.setdefault(t, []).append(var)

    @property
    def k(self):
        return len(self.trainings)

    def gate(self):
        """(ok, reason). Fail closed: an untraceable rung is never counted."""
        if self.untraceable:
            return False, (f"{len(self.untraceable)} scored set(s) have no entry in "
                           f"{PROVENANCE.name} ({', '.join(self.untraceable)}) — the training "
                           f"behind them is unknown, so they cannot be counted as replicates")
        if self.no_draw and not self.trainings:
            return True, (f"no training draw ({', '.join(self.no_draw)}) — nothing to replicate, "
                          f"exempt")
        if self.no_draw:
            return False, (f"mixes trained rungs with draw-free ones ({', '.join(self.no_draw)}) — "
                           f"they are not replicates of each other")
        if self.k < MIN_TRAININGS:
            return False, (f"{self.k} distinct training(s) "
                           f"({'; '.join(f'{t.split(chr(47))[-1]}: {v}' for t, v in self.trainings.items())})"
                           f" — needs {MIN_TRAININGS}")
        return True, f"{self.k} distinct trainings ({self.k} replicates)"

    def per_window(self, windows):
        """Family mean per window, averaged over the side's replicates."""
        return {w: statistics.mean([v[w] for _, v in self.sets]) for w in windows}

    def aggregates(self, windows):
        return [statistics.mean([v[w] for w in windows]) for _, v in self.sets]

    def sem_train(self, windows):
        """The training-draw term on this side's family mean: sd/sqrt(k)."""
        a = self.aggregates(windows)
        if len(a) < 2:
            return 0.0, 0.0
        sd = statistics.stdev(a)
        return sd, sd / len(a) ** 0.5


def describe(v, label):
    if not v:
        print(f"  {label:<34} —")
        return
    m = statistics.mean(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    sem = sd / len(v) ** 0.5 if len(v) > 1 else 0.0
    pos = sum(1 for x in v if x > 0)
    print(f"  {label:<34} {m:+.4f} ± {sem:.4f} sem  ({m/sem:.1f} sem, {pos}/{len(v)} positive)"
          if sem else f"  {label:<34} {m:+.4f}  (n={len(v)})")


def run(paths_a, paths_b, label_a=None, label_b=None):
    prov = provenance()
    A = Side(paths_a, label_a or pathlib.Path(paths_a[0]).stem, prov)
    B = Side(paths_b, label_b or pathlib.Path(paths_b[0]).stem, prov)

    windows = None
    for side in (A, B):
        for _, v in side.sets:
            windows = set(v) if windows is None else windows & set(v)
    windows = sorted(windows or [])
    print(f"[windows] shared by all {len(A.sets) + len(B.sets)} scored set(s): {len(windows)}")
    for side in (A, B):
        for var, v in side.sets:
            drop = sorted(set(v) - set(windows))
            print(f"  {var:<14} {len(v):>3} scored" + (f"  (dropped: {drop})" if drop else ""))

    budgets = set()
    for side in (A, B):
        for p in side.paths:
            d = json.loads(pathlib.Path(p).read_text())
            if isinstance(d, dict):
                budgets.add(d.get("budget"))
    print(f"[budget]  {sorted(budgets, key=lambda x: (x is None, x))}")
    if len(budgets) > 1:
        print("  WARNING: different budgets — not comparable")

    print("\nSIGNAL/novel, nats/char, on the shared windows:")
    for side in (A, B):
        for var, v in side.sets:
            describe([v[w] for w in windows], f"  {var}")
        if len(side.sets) > 1:
            sd, semt = side.sem_train(windows)
            describe([side.per_window(windows)[w] for w in windows], f"{side.label} FAMILY MEAN")
            print(f"  {'':<34} training draw: sd {sd:.5f} over {len(side.sets)} runs"
                  f" → ±{semt:.5f} on the family mean")

    oka, why_a = A.gate()
    okb, why_b = B.gate()
    print(f"\n[replication] {A.label}: {'OK' if oka else 'INSUFFICIENT'} — {why_a}")
    print(f"[replication] {B.label}: {'OK' if okb else 'INSUFFICIENT'} — {why_b}")

    if not (oka and okb):
        print(f"\nPAIRED: REFUSED — a between-rung delta needs >= {MIN_TRAININGS} distinct trainings "
              f"per side.")
        print(f"  The floor on a one-run-vs-one-run comparison is ±{ONE_RUN_FLOOR:.4f} nats/char "
              f"(training draw ±0.00226 folded with window sem ~0.0022), against effects of "
              f"0.002–0.008.")
        print(f"  That floor is why 4a07481/db416ad's '144 folds lose to 47' (−0.0018 ± 0.0022) was "
              f"withdrawn in 8e183de: replicated, the difference is −0.0002 and negative in 6 of 12 "
              f"pairings.")
        print(f"  The per-rung aggregates above are single-rung readings and stand; their DIFFERENCE "
              f"does not, and is not printed.")
        print(f"  To get one: PREFIX=... N=... TAG=... SEEDS=\"1 2 3\" ./wake/run-seed-spread.sh "
              f"per side, then pass all of them with --a/--b.")
        print(f"  See condent-results-2026-08-18-seed-spread.md, "
              f"'The error bar this lane should have been using'.")
        return 2

    fa, fb = A.per_window(windows), B.per_window(windows)
    d = [fb[w] - fa[w] for w in windows]
    m = statistics.mean(d)
    sem_win = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
    _, sta = A.sem_train(windows)
    _, stb = B.sem_train(windows)
    bar = (sem_win ** 2 + sta ** 2 + stb ** 2) ** 0.5
    pos = sum(1 for x in d if x > 0)
    print(f"\nPAIRED: {B.label} − {A.label}  (family means, {len(windows)} windows)")
    print(f"  {m:+.4f} ± {bar:.4f}   ({m/bar:+.1f} sem, {pos}/{len(d)} windows positive)")
    print(f"  bar = window sem {sem_win:.5f} ⊕ training draw {sta:.5f} ({A.label}) "
          f"⊕ {stb:.5f} ({B.label})")

    pairings = [statistics.mean([vb[w] - va[w] for w in windows])
                for (_, va), (_, vb) in itertools.product(A.sets, B.sets)]
    neg = sum(1 for x in pairings if x < 0)
    print(f"  one-run-vs-one-run pairings: {len(pairings)}, mean {statistics.mean(pairings):+.4f}, "
          f"range [{min(pairings):+.4f}, {max(pairings):+.4f}], negative in {neg}/{len(pairings)}")
    return 0


def selftest():
    """Drive the gate RED and GREEN on real scored sets. A gate not seen to fail
    is not a gate."""
    def paths(*v):
        return [str(HERE / f"recs-{x}.json") for x in v]

    py = sys.executable
    cases = []
    one_a, one_b = paths("ft47-e3"), paths("ft144b-e3")
    three_a = paths("ft47rs1-e3", "ft47rs2-e3", "ft47rs3-e3")
    three_b = paths("ft144s1-e3", "ft144s2-e3", "ft144s3-e3")
    missing = [p for p in one_a + one_b + three_a + three_b if not pathlib.Path(p).exists()]
    if missing or not PROVENANCE.exists():
        print(f"[test] SKIP (exit 2) — scored sets absent on this node: "
              f"{[pathlib.Path(m).name for m in missing] or PROVENANCE.name}")
        return 2

    cases.append(("1-vs-1 must REFUSE", ["--a"] + one_a + ["--b"] + one_b, 2, "PAIRED: REFUSED", None))
    cases.append(("legacy positional 1-vs-1 must REFUSE", one_a + one_b, 2, "PAIRED: REFUSED", None))
    cases.append(("3-vs-1 must REFUSE", ["--a"] + three_a + ["--b"] + one_b, 2, "PAIRED: REFUSED", None))
    cases.append(("3-vs-3 must PRINT", ["--a"] + three_a + ["--b"] + three_b, 0, "PAIRED: ", "REFUSED"))
    cases.append(("untraceable rung must REFUSE",
                  ["--a"] + three_a + ["--b"] + three_b + ["--_test_forget", "ft144s1-e3"],
                  2, "no entry in", None))
    # the untrained base carries no training draw: exempt, but the trained side still gated
    cases.append(("base (no draw) vs 3 must PRINT", ["--a"] + paths("model") + ["--b"] + three_b,
                  0, "PAIRED: ", "REFUSED"))
    cases.append(("base (no draw) vs 1 must REFUSE", ["--a"] + paths("model") + ["--b"] + one_b,
                  2, "PAIRED: REFUSED", None))

    bad = 0
    for name, argv, want_rc, want, forbid in cases:
        r = subprocess.run([py, str(HERE / "paired.py")] + argv, capture_output=True, text=True)
        ok = r.returncode == want_rc and want in r.stdout and (forbid is None or forbid not in r.stdout)
        print(f"[test] {'ok  ' if ok else 'FAIL'} {name}  (rc={r.returncode}, want {want_rc})")
        if not ok:
            bad += 1
            print("  ---", r.stdout.strip()[-600:] or r.stderr.strip()[-600:])
    print(f"[test] {len(cases) - bad}/{len(cases)} passed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy", nargs="*", help="legacy positional A B (one run a side: refused)")
    ap.add_argument("--a", nargs="+", default=None, help="scored sets for rung A (>=3 trainings)")
    ap.add_argument("--b", nargs="+", default=None, help="scored sets for rung B (>=3 trainings)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--test", action="store_true", help="self-test: drive the gate red and green")
    ap.add_argument("--_test_forget", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.test:
        return selftest()
    if a._test_forget:                      # self-test hook: hide one rung's provenance
        global provenance
        _real = provenance
        provenance = lambda: {k: v for k, v in _real().items() if k != a._test_forget}

    pa, pb = a.a, a.b
    if pa is None and pb is None:
        if len(a.legacy) != 2:
            ap.error("give --a <sets...> --b <sets...> (or two positional files)")
        pa, pb = [a.legacy[0]], [a.legacy[1]]
    if not (pa and pb):
        ap.error("--a and --b must both be given")
    return run(pa, pb, a.label_a, a.label_b)


if __name__ == "__main__":
    sys.exit(main())
