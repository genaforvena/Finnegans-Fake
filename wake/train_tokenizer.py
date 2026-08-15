#!/usr/bin/env python3
"""Train a BPE tokenizer ON the Wake itself.

The whole point: 'penisolate', 'wielderfight', 'thuartpeatrick' become learned
units instead of being shredded into ordinary English fragments by a tokenizer
that never saw the book. This is what makes the from-scratch bet a word-play
bet rather than a small-model bet.
"""
import pathlib, sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "wake_clean.txt"

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
OUT = ROOT / "wake" / f"tokenizer-{VOCAB}.json"

tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()
tok.post_processor = processors.ByteLevel(trim_offsets=False)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)
tok.train([str(CORPUS)], trainer)
tok.save(str(OUT))

text = CORPUS.read_text(encoding="utf-8")
ids = tok.encode(text).ids
print(f"vocab={tok.get_vocab_size()} corpus_tokens={len(ids)} chars_per_token={len(text)/len(ids):.2f}")
for w in ["penisolate", "wielderfight", "thuartpeatrick", "passencore", "riverrun", "bababadalgharaghtakamminarronnkonn"]:
    enc = tok.encode(" " + w)
    print(f"  {w!r:40s} -> {enc.tokens}")
print("saved", OUT)
