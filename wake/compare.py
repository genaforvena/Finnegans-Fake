#!/usr/bin/env python3
"""Same seed phrase through every trained variant, side by side, unedited.

The question 'does any of this still resemble English' is settled by reading the
output, not by arguing about tokenisers — so this prints raw text and the loss
each model was selected at, and nothing else.

  python wake/compare.py --prompt "riverrun, past Eve and Adam's," --tokens 120
"""
import argparse, json, pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
WAKE = ROOT / "wake"

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="riverrun, past Eve and Adam's,")
p.add_argument("--tokens", type=int, default=120)
p.add_argument("--temp", type=float, default=0.9)
p.add_argument("--top-p", type=float, default=0.95)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--which", default="final", choices=["final", "best", "both"])
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"


def variants():
    for run in sorted(WAKE.glob("wake-*")):
        if not (run / "config.json").exists():
            continue
        log = {}
        f = run / "trainlog.json"
        if f.exists():
            log = json.loads(f.read_text())
        if a.which in ("best", "both"):
            yield f"{run.name} [best-val]", run, log
        if a.which in ("final", "both") and (run / "final" / "config.json").exists():
            yield f"{run.name}/final [overfit]", run / "final", log


for name, path, log in variants():
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path)).to(dev).eval()
    ids = tok(a.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=a.tokens, do_sample=True, temperature=a.temp,
            top_p=a.top_p, top_k=0, pad_token_id=tok.eos_token_id or 0,
        )
    n = sum(q.numel() for q in model.parameters()) / 1e6
    meta = ""
    if log:
        meta = (f"  train {log.get('final_train')}  val {log.get('final_val')}"
                f"  best-val {round(log.get('best_val', 0), 3)}")
    print(f"\n{'='*72}\n{name}  ({n:.1f}M params){meta}\n{'='*72}")
    print(tok.decode(out[0], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
