#!/usr/bin/env python3
"""Re-cut the extra teacher windows so they have the shape of the eval set.

The first extra batch (97 windows, ids u000..u096) was cut by deleting every board
LINE already claimed by the training, eval or test sets and then cutting from what
was left. That guarantees zero overlap and it also guarantees the leftovers are the
gaps between claimed windows -- median 6 lines and 5 voices against 13 and 8 for
eval and test. Two thirds of the training set became a narrower task than the one
the student is scored on, and the 144-fold rung lost to the 47-fold rung.

Two things made that cut fragmentary, and only the second one was intended:

  content blocking   board lines recur verbatim (roll-calls, refusals, liveness
                     pings). Blocking by line TEXT matched 2647 of 5302 rows -- half
                     the corpus, most of it never in any claimed window. Blocking by
                     the located index SPAN of each frozen window removes 1343.
  no ordinary cut    the leftovers were cut as leftovers. Here the ordinary
                     contiguous cutter runs over the maximal unclaimed runs, i.e.
                     the new windows are cut the way batch 1 and eval were cut.

The eval and test windows stay exactly where they are. They are frozen on disk and
the 47-fold rung has already been scored against them; re-cutting them would make
the two rungs incomparable, which is the one thing this run has to preserve. So the
selection order is applied to what is left after the frozen sets are removed, and
the frozen sets are removed by span rather than by text.

  python recut_windows.py --n 97 --out recut-windows.json
"""
import argparse, json, pathlib, statistics as st, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import board_corpus

HERE = pathlib.Path(__file__).resolve().parent


def board_lines(rows):
    """The exact string each board row contributes to a window."""
    return [f"{win}: {body}" for _ts, win, body in rows]


def locate(text, lines, index):
    """(i, j) span of rows whose joined text is exactly `text`, or None.

    Board lines repeat, so a first-line hit is a candidate and not an answer; the
    whole window has to match before the span is believed.
    """
    want = text.split("\n")
    for i in index.get(want[0], []):
        j, acc = i, []
        while j < len(lines) and len("\n".join(acc + [lines[j]])) <= len(text):
            acc.append(lines[j])
            j += 1
            if "\n".join(acc) == text:
                return (i, j)
    return None


def frozen_spans(rows, lines, index):
    """Spans of every window already claimed, with what could not be found reported.

    The oldest board snapshots have rotated out since the frozen sets were cut, so a
    few early windows no longer exist in the union at all. Those cannot be re-cut
    either, which is why unlocatable is safe rather than silent.
    """
    ev = json.loads((HERE / "eval-windows.json").read_text())
    te = json.loads((HERE / "test-windows.json").read_text())
    tf = json.loads((HERE / "teacher-folds.json").read_text())
    b1 = [r for r in tf if r["id"].startswith("t")]

    groups, report = {}, {}
    for label, items, key in (("eval", ev, "text"), ("test", te, "text"),
                              ("batch1", b1, "source")):
        spans, missing = [], []
        for it in items:
            s = locate(it[key], lines, index)
            (spans.append(s) if s else missing.append(it["id"]))
        groups[label] = spans
        report[label] = {"located": len(spans), "of": len(items), "missing": missing}
    return groups, report


def cut(rows, lines, blocked, target_chars=4200, max_lines=22):
    """board_corpus.cut_windows over the unblocked runs, same shape parameters."""
    out, i, k = [], 0, 0
    while i < len(rows):
        if blocked[i]:
            i += 1
            continue
        buf, chars, j = [], 0, i
        while (j < len(rows) and chars < target_chars
               and len(buf) < max_lines and not blocked[j]):
            buf.append(lines[j])
            chars += len(lines[j]) + 1
            j += 1
        if len(buf) >= 4 and chars >= 1800:
            out.append({"id": f"v{k:03d}", "text": "\n".join(buf), "n_lines": len(buf),
                        "start": rows[i][0], "span": [i, j]})
            k += 1
        i = max(j, i + 1)
    return out


def voices(text):
    return len({l.split(":")[0] for l in text.split("\n") if ":" in l})


def describe(texts, label):
    L = [t.count("\n") + 1 for t in texts]
    V = [voices(t) for t in texts]
    C = [len(t) for t in texts]
    print(f"  {label:26s} n={len(texts):3d}  lines med={st.median(L):5.1f} "
          f"p10={sorted(L)[len(L)//10]:3d}  voices med={st.median(V):4.1f}  "
          f"chars med={st.median(C):6.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=97)
    ap.add_argument("--min-lines", type=int, default=4,
                    help="4 is the cutter's own floor, i.e. no extra selection. Raising "
                         "it selects on window width, which is the axis under test -- "
                         "board order already reproduces the eval shape without it")
    ap.add_argument("--out", default="recut-windows.json")
    a = ap.parse_args()

    rows = board_corpus.union_board()
    lines = board_lines(rows)
    index = {}
    for i, s in enumerate(lines):
        index.setdefault(s, []).append(i)
    print(f"union board: {len(rows)} entries, {rows[0][0]} .. {rows[-1][0]}")

    groups, report = frozen_spans(rows, lines, index)
    for label, r in report.items():
        print(f"  frozen {label:7s} {r['located']}/{r['of']} located"
              + (f", not in corpus: {r['missing']}" if r["missing"] else ""))

    blocked = [False] * len(rows)
    for spans in groups.values():
        for i, j in spans:
            for k in range(i, j):
                blocked[k] = True
    print(f"blocked rows: {sum(blocked)} of {len(rows)}")

    cand = cut(rows, lines, blocked)
    print(f"candidates cut: {len(cand)}")

    # leakage audit. The index spans cannot overlap, so a shared line can only be a
    # RECURRENCE -- the mesh reposts roll-calls, merge refusals and liveness pings
    # verbatim, and those recur across the whole corpus. Counting shared lines
    # therefore drops long windows for being long, which is the original mistake in
    # a different coat. What would actually be leakage is a candidate reproducing a
    # RUN of an eval window, so the audit is on contiguous runs.
    ev = json.loads((HERE / "eval-windows.json").read_text())
    te = json.loads((HERE / "test-windows.json").read_text())
    RUN = 4
    claimed_runs = set()
    for w in ev + te:
        ls = w["text"].split("\n")
        for i in range(len(ls) - RUN + 1):
            claimed_runs.add(tuple(ls[i:i + RUN]))

    def reruns(text):
        ls = text.split("\n")
        return sum(1 for i in range(len(ls) - RUN + 1)
                   if tuple(ls[i:i + RUN]) in claimed_runs)

    heavy = [w for w in cand if reruns(w["text"])]
    print(f"candidates reproducing a {RUN}-line run of eval/test: {len(heavy)} (dropped)")
    cand = [w for w in cand if not reruns(w["text"])]
    shared = [sum(1 for l in w["text"].split("\n")
                  if any(l in x["text"] for x in ev)) for w in cand]
    print(f"  isolated recurring lines kept: median {st.median(shared):.0f} per window")

    wide = [w for w in cand if w["n_lines"] >= a.min_lines]
    print(f"candidates >= {a.min_lines} lines: {len(wide)}")

    print("\nshape:")
    describe([w["text"] for w in cand], "all candidates")
    describe([w["text"] for w in wide], f">= {a.min_lines} lines")
    describe([w["text"] for w in ev], "eval (target)")
    describe([w["text"] for w in te], "test")
    tf = json.loads((HERE / "teacher-folds.json").read_text())
    describe([r["source"] for r in tf if r["id"].startswith("t")], "batch1")
    describe([r["source"] for r in tf if r["id"].startswith("u")], "batch2 (being replaced)")

    if len(wide) < a.n:
        sys.exit(f"only {len(wide)} candidates, need {a.n}")
    pick = wide[:a.n]                       # board order: no selection on content
    print(f"\nselected: {len(pick)}")
    describe([w["text"] for w in pick], "selected")

    out = HERE / a.out
    out.write_text(json.dumps([{k: w[k] for k in ("id", "text", "n_lines", "start")}
                               for w in pick], indent=1))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
