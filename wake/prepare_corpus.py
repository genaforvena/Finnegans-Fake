#!/usr/bin/env python3
"""Clean a scanned Finnegans Wake text into a training corpus.

Input : data/wake_reconstructed_raw.txt  (not in the repo — see README)
Output: data/wake_clean.txt

The load-bearing step is the first one. A scan breaks Joyce's words across
lines — 'passen-' newline 'core' — and a tokenizer fed that sees two ordinary
English fragments instead of the coinage 'passencore'. Rejoining them back into
single words is what preserves the wordplay the whole project is about.

`clean()` is exported rather than inlined because a SECOND corpus now goes
through it (wake/prepare_english.py, the plain-English expert). Two corpora that
are meant to differ only in content must not drift apart in formatting, and the
only way to guarantee that is to have one function, not two copies of one.
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def clean(raw: str) -> str:
    """The whole normalisation pipeline, verbatim. Do not fork this."""
    t = re.sub(r"-\s*\n\s*", "", raw)          # rejoin words split across scan lines
    t = re.sub(r"[ \t]+", " ", t)              # collapse the scan's ragged spacing
    t = "\n".join(l.strip() for l in t.split("\n"))
    t = re.sub(r"\n{2,}", "\x00", t)           # protect paragraph breaks
    t = t.replace("\n", " ")                   # unwrap lines within a paragraph
    t = t.replace("\x00", "\n\n")
    t = re.sub(r" {2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def stats(raw_len: int, t: str, dst: pathlib.Path) -> None:
    words = re.findall(r"[A-Za-z']+", t)
    types = {w.lower() for w in words}
    print(f"{raw_len} -> {len(t)} chars, {len(words)} words, {len(types)} types, "
          f"{t.count(chr(10)*2)+1} paragraphs -> {dst}")


if __name__ == "__main__":
    SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "wake_reconstructed_raw.txt"
    DST = ROOT / "data" / "wake_clean.txt"
    if not SRC.exists():
        sys.exit(f"missing corpus: {SRC}\nsee README — the text is not distributed with this repo")
    raw = SRC.read_text(encoding="utf-8", errors="replace")
    t = clean(raw)
    DST.write_text(t + "\n", encoding="utf-8")
    stats(len(raw), t, DST)
