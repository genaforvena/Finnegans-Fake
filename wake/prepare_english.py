#!/usr/bin/env python3
"""Build data/english_clean.txt — the plain-English half of the product of experts.

The point of this corpus is to be the *other* lexicon. The Wake expert says what
Joyce would say; this one says what ordinary narrative English would say; the
product sampler (wake/product_sample.py) emits only characters both find
probable, which is a portmanteau's condition of existence.

Two things therefore matter and nothing else does:

  1. CONTENT must be ordinary. Public-domain 19th/early-20th century narrative
     prose, several authors mixed so no single voice dominates, and deliberately
     NOT Joyce — A Portrait is on Gutenberg and including it would blur exactly
     the line the product is drawn along.

  2. FORMATTING must be identical to the Wake's. The two corpora have to differ
     in content, not in whitespace convention, or the models learn a formatting
     difference and the product measures that instead. So this file imports
     `clean()` from prepare_corpus.py rather than reimplementing it.

Between the download and `clean()` there is a Gutenberg-specific de-boilerplating
pass, which is a property of the *source*, not of the cleaning:

  - the ``*** START/END OF THE PROJECT GUTENBERG EBOOK ***`` band
  - front matter (title page, dedication, contents, preface) — dropped by seeking
    a per-book first-sentence marker, listed in BOOKS below
  - ``[Illustration: ...]`` blocks, ``_italic_`` underscores, chapter headings
  - curly quotes/dashes folded to the ASCII forms the Wake corpus already uses.
    wake_clean.txt contains " and ' and — and no “ ” ‘ ’; leaving Gutenberg's
    curly forms in would be a formatting difference wearing content's clothes.

Size is matched to data/wake_clean.txt by taking a PREFIX of whole paragraphs
from each book until its quota (target / n_books) is filled. Deterministic: no
sampling, no shuffling, same bytes every run.

  python wake/prepare_english.py            # from the local downloads
  python wake/prepare_english.py --fetch    # download any that are missing first
"""
import argparse, pathlib, re, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from prepare_corpus import clean, stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAWDIR = ROOT / "data" / "gutenberg"
DST = ROOT / "data" / "english_clean.txt"
WAKE = ROOT / "data" / "wake_clean.txt"

# (gutenberg id, short name, first sentence of the actual narrative)
# The marker exists to skip front matter. It must be a string that occurs in the
# body and NOT in the table of contents — a chapter TITLE would match the TOC
# line first and drag the whole contents list into the corpus.
BOOKS = [
    (1342, "austen-pride-and-prejudice",   "It is a truth universally acknowledged"),
    (98,   "dickens-tale-of-two-cities",   "It was the best of times"),
    (1260, "bronte-jane-eyre",             "There was no possibility of taking a walk"),
    (120,  "stevenson-treasure-island",    "Squire Trelawney, Dr. Livesey, and the rest"),
    (174,  "wilde-dorian-gray",            "The studio was filled with the rich odour of roses"),
    (219,  "conrad-heart-of-darkness",     "The Nellie, a cruising yawl"),
    (2701, "melville-moby-dick",           "Call me Ishmael"),
    (145,  "eliot-middlemarch",            "Miss Brooke had that kind of beauty"),
]

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch(bid: int, path: pathlib.Path) -> None:
    # --http1.1 is not decoration: over this node's egress the HTTP/2 fetch of a
    # ~1MB body dies mid-stream with curl 56 while HEAD returns 200, which reads
    # as "the file is unreachable" when it is only the protocol.
    print(f"  fetching {bid} -> {path.name}", flush=True)
    subprocess.run(
        ["curl", "-sS", "--http1.1", "-m", "300", "--retry", "3", "--retry-delay", "3",
         "--retry-all-errors", "-o", str(path), URL.format(id=bid)],
        check=True,
    )
    if path.stat().st_size < 50_000:
        path.unlink(missing_ok=True)
        sys.exit(f"download for {bid} came back too small — refusing to build on it")


def strip_gutenberg(raw: str, marker: str, name: str) -> str:
    """Everything that is Project Gutenberg rather than the book."""
    m = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    if not m:
        sys.exit(f"{name}: no START banner — is this a Gutenberg plain text?")
    t = raw[m.end():]
    m = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    if not m:
        sys.exit(f"{name}: no END banner")
    t = t[:m.start()]

    i = t.find(marker)
    if i < 0:
        sys.exit(f"{name}: body marker {marker!r} not found — the edition changed")
    t = t[i:]

    t = re.sub(r"\[Illustration:.*?\]", "", t, flags=re.S)
    t = re.sub(r"\[Illustration\]", "", t)
    t = re.sub(r"^\s*(CHAPTER|Chapter|BOOK|Book the|VOLUME|PART|PRELUDE|FINALE)\b.*$",
               "", t, flags=re.M)
    t = re.sub(r"^\s*[IVXLC]+\.?\s*$", "", t, flags=re.M)      # bare roman numerals
    t = t.replace("_", "")                                      # italic markers
    # Fold to the ASCII punctuation the Wake corpus already uses.
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
                 ("–", "-"), (" ", " "), ("﻿", "")]:
        t = t.replace(a, b)
    return t


def paragraphs(t: str):
    return [p for p in re.split(r"\n\s*\n", t) if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download missing raws first")
    ap.add_argument("--target", type=int, default=0,
                    help="target chars (default: match data/wake_clean.txt)")
    a = ap.parse_args()

    if not WAKE.exists() and not a.target:
        sys.exit(f"missing {WAKE} — run prepare_corpus.py first, or pass --target")
    target = a.target or len(WAKE.read_text(encoding="utf-8"))
    quota = target // len(BOOKS)

    RAWDIR.mkdir(parents=True, exist_ok=True)
    pieces, report, hyphen_joins = [], [], 0
    for bid, name, marker in BOOKS:
        path = RAWDIR / f"pg{bid}.txt"
        if not path.exists():
            if not a.fetch:
                sys.exit(f"missing {path} — run with --fetch to download it")
            fetch(bid, path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = strip_gutenberg(raw, marker, name)
        hyphen_joins += len(re.findall(r"[A-Za-z]-\s*\n\s*[A-Za-z]", body))

        # Whole paragraphs from the front of the book until the quota is met.
        taken, n = [], 0
        for p in paragraphs(body):
            taken.append(p)
            n += len(p) + 2
            if n >= quota:
                break
        piece = "\n\n".join(taken)
        pieces.append(piece)
        report.append((name, len(body), len(piece), len(taken)))

    t = clean("\n\n".join(pieces))
    DST.write_text(t + "\n", encoding="utf-8")

    print(f"target {target} chars (= data/wake_clean.txt), quota {quota}/book, {len(BOOKS)} books")
    for name, avail, took, npar in report:
        print(f"  {name:34s} body {avail:8d}  took {took:7d}  ({npar} paragraphs)")
    # The Wake cleaner rejoins a hyphen at end-of-line, because in a SCAN that is
    # a broken word. In a Gutenberg text it is usually a real compound, so this
    # step does damage here ('sea-\nreach' -> 'seareach'). Counted rather than
    # special-cased: identical formatting is the requirement, and the count is
    # the size of what it cost.
    print(f"  end-of-line hyphens rejoined by the shared cleaner: {hyphen_joins}")
    stats(sum(len(p) for p in pieces), t, DST)

    wake_n = len(WAKE.read_text(encoding="utf-8")) if WAKE.exists() else 0
    if wake_n:
        print(f"\nsize match (the artifact, not the claim):")
        print(f"  data/wake_clean.txt     {wake_n:9d} chars")
        print(f"  data/english_clean.txt  {len(t)+1:9d} chars")
        print(f"  ratio english/wake      {(len(t)+1)/wake_n:9.4f}")


if __name__ == "__main__":
    main()
