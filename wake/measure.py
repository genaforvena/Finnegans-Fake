#!/usr/bin/env python3
"""Describe each variant's output as properties, not as a ranking.

Loss ranks models on an objective nobody here is optimising for, and words like
"degenerate" or "speaks worse" smuggle a verdict in as an observation. These are
quantities instead — a reader decides what they want from them.

  self_repeat  fraction of 4-grams in the output that recur within the output
  verb3/verb5  fraction of output 3- and 5-grams found verbatim in the book. An
               8-gram threshold reads 0.000 for every model, which is
               indistinguishable from a broken measure; the copying is real but
               it lives in short spans (45% of word PAIRS come from the book)
  novel_words  fraction of output word types absent from the book
  ttr          type/token ratio of the output

  python wake/measure.py [--samples 8] [--chars 900]
"""
import argparse, json, pathlib, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen import generate_chars

ROOT = pathlib.Path(__file__).resolve().parent.parent
WAKE = ROOT / "wake"

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="riverrun, past Eve and Adam's,")
p.add_argument("--samples", type=int, default=8)
p.add_argument("--chars", type=int, default=900)
p.add_argument("--temp", type=float, default=0.85)
p.add_argument("--top-p", type=float, default=0.95)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--json", default="")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
book = (ROOT / "data" / "wake_clean.txt").read_text(encoding="utf-8").lower()
book_words = re.findall(r"[a-z']+", book)
book_vocab = set(book_words)
book_ng = {n: {tuple(book_words[i:i + n]) for i in range(len(book_words) - n + 1)}
           for n in (3, 5)}


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def describe(text, vocab=None):
    """vocab: what counts as 'already in the book'. For a slice OF the book it must
    exclude that slice, or novel_words is a tautology — the book scores 0.0 against
    itself by construction and cannot serve as a baseline for the models."""
    vocab = book_vocab if vocab is None else vocab
    w = re.findall(r"[a-z']+", text.lower())
    if len(w) < 12:
        return None
    g4 = ngrams(w, 4)
    uniq4 = len(set(g4))
    v = {}
    for n in (3, 5):
        g = ngrams(w, n)
        v[f"verb{n}"] = round(sum(x in book_ng[n] for x in g) / max(len(g), 1), 3)
    return {
        "words": len(w),
        "self_repeat": round(1 - uniq4 / max(len(g4), 1), 3),
        **v,
        "novel_words": round(len({x for x in w if x not in vocab}) / len(set(w)), 3),
        "ttr": round(len(set(w)) / len(w), 3),
    }


# The book itself, same measurements, same length — without it the numbers have no
# scale and "high repetition" is a feeling rather than a comparison. Held-out slices
# only, so `verbatim` stays a meaningful column for the models.
import random

rng = random.Random(a.seed)
raw = (ROOT / "data" / "wake_clean.txt").read_text(encoding="utf-8")
book_stats = []
for _ in range(max(a.samples, 8)):
    i = rng.randrange(0, max(len(raw) - a.chars - 1, 1))
    slice_ = raw[i:i + a.chars]
    rest = set(re.findall(r"[a-z']+", (raw[:i] + raw[i + a.chars:]).lower()))
    d = describe(slice_, vocab=rest)
    if d:
        book_stats.append(d)
rows = [{"run": "THE BOOK", "n": len(book_stats),
         **{k: round(sum(s[k] for s in book_stats) / len(book_stats), 3)
            for k in book_stats[0] if k != "words"}}]

for run in sorted(WAKE.glob("wake-*")):
    path = run / "final"
    if not (path / "config.json").exists():
        continue
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path)).to(dev).eval()
    stats, texts = [], []
    for s in range(a.samples):
        torch.manual_seed(a.seed + s)
        t = generate_chars(model, tok, a.prompt, a.chars, a.temp, a.top_p, dev)
        texts.append(t)
        d = describe(t)
        if d:
            stats.append(d)
    del model
    torch.cuda.empty_cache()
    if not stats:
        continue
    keys = [k for k in stats[0] if k != "words"]
    row = {"run": run.name, "n": len(stats),
           **{k: round(sum(s[k] for s in stats) / len(stats), 3) for k in keys}}
    log = run / "trainlog.json"
    if log.exists():
        j = json.loads(log.read_text())
        row["train"] = round(j["final_train"], 3)
        row["best_val"] = round(j["best_val"], 3)
    rows.append(row)

cols = ["run", "n", "self_repeat", "verb3", "verb5", "novel_words", "ttr", "train", "best_val"]
print(f"{a.samples} samples/model, {a.chars} chars each, temp {a.temp}, prompt {a.prompt!r}")
print("  ".join(f"{c:>18s}" if c == "run" else f"{c:>12s}" for c in cols))
for r in rows:
    print("  ".join(f"{str(r.get(c, '')):>18s}" if c == "run" else f"{str(r.get(c, '')):>12s}"
                    for c in cols))
print("\nself_repeat/verb3/verb5/novel_words/ttr are properties of the text, not a ranking.")
print("train and best_val are in different units across tokenisations — compare down a")
print("column within one tokenisation only, never across the char/BPE line.")
if a.json:
    pathlib.Path(a.json).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=2))
