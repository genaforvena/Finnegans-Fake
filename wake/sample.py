#!/usr/bin/env python3
"""Generate from a trained Wake model.

  python sample.py wake-bpe4096 --prompt "riverrun, past Eve and" -n 3
  python sample.py wake-char257 --chat        # interactive; the thing to play with
"""
import argparse, pathlib, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--prompt", default="riverrun, past Eve and Adam's,")
p.add_argument("-n", "--num", type=int, default=3)
p.add_argument("--tokens", type=int, default=200)
p.add_argument("--temp", type=float, default=0.9)
p.add_argument("--top-k", type=int, default=100)
p.add_argument("--top-p", type=float, default=0.95)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--chat", action="store_true")
a = p.parse_args()

path = pathlib.Path(a.model)
if not path.exists():
    path = ROOT / "wake" / a.model
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(str(path))
model = AutoModelForCausalLM.from_pretrained(str(path)).to(dev).eval()
if a.seed is not None:
    torch.manual_seed(a.seed)


def gen(prompt, n=1):
    ids = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(
        **ids, max_new_tokens=a.tokens, do_sample=True, temperature=a.temp,
        top_k=a.top_k, top_p=a.top_p, num_return_sequences=n,
        pad_token_id=tok.eos_token_id or 0,
    )
    return [tok.decode(o, skip_special_tokens=True) for o in out]


if a.chat:
    print(f"[{path.name}] type a seed phrase; empty line quits. Ctrl-C also works.")
    while True:
        try:
            line = input("\nwake> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            break
        print(gen(line, 1)[0])
else:
    for i, t in enumerate(gen(a.prompt, a.num), 1):
        print(f"--- sample {i} " + "-" * 50)
        print(t)
