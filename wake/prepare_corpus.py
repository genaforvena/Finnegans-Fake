#!/usr/bin/env python3
"""Clean a scanned Finnegans Wake text into a training corpus.

Input : data/wake_reconstructed_raw.txt  (not in the repo — see README)
Output: data/wake_clean.txt

The load-bearing step is the first one. A scan breaks Joyce's words across
lines — 'passen-' newline 'core' — and a tokenizer fed that sees two ordinary
English fragments instead of the coinage 'passencore'. Rejoining them back into
single words is what preserves the wordplay the whole project is about.
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "wake_reconstructed_raw.txt"
DST = ROOT / "data" / "wake_clean.txt"

if not SRC.exists():
    sys.exit(f"missing corpus: {SRC}\nsee README — the text is not distributed with this repo")

raw = SRC.read_text(encoding="utf-8", errors="replace")
t = re.sub(r"-\s*\n\s*", "", raw)          # rejoin words split across scan lines
t = re.sub(r"[ \t]+", " ", t)              # collapse the scan's ragged spacing
t = "\n".join(l.strip() for l in t.split("\n"))
t = re.sub(r"\n{2,}", "\x00", t)           # protect paragraph breaks
t = t.replace("\n", " ")                   # unwrap lines within a paragraph
t = t.replace("\x00", "\n\n")
t = re.sub(r" {2,}", " ", t)
t = re.sub(r"\n{3,}", "\n\n", t).strip()
DST.write_text(t + "\n", encoding="utf-8")

words = re.findall(r"[A-Za-z']+", t)
types = {w.lower() for w in words}
print(f"{len(raw)} -> {len(t)} chars, {len(words)} words, {len(types)} types, "
      f"{t.count(chr(10)*2)+1} paragraphs -> {DST}")
