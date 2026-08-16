#!/usr/bin/env python3
"""Print recut windows for hand-folding, and collect the folds written back."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ws = json.loads((HERE / "recut-windows.json").read_text())
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for w in ws[lo:hi]:
    print(f"===== {w['id']}  ({w['n_lines']} lines, {len(w['text'])} chars, {w['start']}) =====")
    print(w["text"])
    print()
