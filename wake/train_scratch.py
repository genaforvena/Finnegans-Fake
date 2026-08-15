#!/usr/bin/env python3
"""Bet 2 — train a small GPT from scratch on Finnegans Wake alone.

No pretraining, no English prior beyond what the book itself contains. The model
knows nothing except this one text, so everything it emits is Wake-shaped by
construction. It will be incoherent; that is the deliverable.

  python train_scratch.py [--iters N] [--vocab 4096] [--out wake-scratch]
"""
import argparse, math, pathlib, time, json
import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--vocab", type=int, default=4096)
p.add_argument("--block", type=int, default=256)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--heads", type=int, default=6)
p.add_argument("--embd", type=int, default=384)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--out", default="wake-scratch")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---- data ----------------------------------------------------------------
tokfile = ROOT / "wake" / f"tokenizer-{a.vocab}.json"
base = Tokenizer.from_file(str(tokfile))
text = (ROOT / "data" / "wake_clean.txt").read_text(encoding="utf-8")
ids = torch.tensor(base.encode(text).ids, dtype=torch.long)
n = int(0.95 * len(ids))
train_ids, val_ids = ids[:n], ids[n:]
print(f"tokens: train={len(train_ids)} val={len(val_ids)} vocab={base.get_vocab_size()}")

def batch(split):
    """labels ARE input_ids — transformers' ForCausalLMLoss shifts them itself.

    Handing it a pre-shifted y (the nanoGPT convention) shifts twice, so the model
    learns to predict token t+2 from position t. It still trains and the loss still
    falls; the damage only shows in the text, which comes out looking like every
    other character was deleted. Cost the char-level run a full pass to find.
    """
    d = train_ids if split == "train" else val_ids
    i = torch.randint(len(d) - a.block - 1, (a.batch,))
    x = torch.stack([d[j:j + a.block] for j in i]).to(dev, non_blocking=True)
    return x, x

# ---- model ---------------------------------------------------------------
cfg = GPT2Config(
    vocab_size=base.get_vocab_size(), n_positions=a.block, n_embd=a.embd,
    n_layer=a.layers, n_head=a.heads, resid_pdrop=a.dropout, embd_pdrop=a.dropout,
    attn_pdrop=a.dropout, bos_token_id=0, eos_token_id=0,
)
model = GPT2LMHeadModel(cfg).to(dev)
nparams = sum(q.numel() for q in model.parameters())
print(f"params: {nparams/1e6:.1f}M  device={dev}")

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

@torch.no_grad()
def evaluate(k=40):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(k)
        for i in range(k):
            x, y = batch(split)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                losses[i] = model(input_ids=x, labels=y).loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

best = math.inf
outdir = ROOT / "wake" / a.out
outdir.mkdir(parents=True, exist_ok=True)
hist = []
t0 = time.time()
for it in range(1, a.iters + 1):
    x, y = batch("train")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        loss = model(input_ids=x, labels=y).loss
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); sched.step()

    if it % a.eval_every == 0 or it == a.iters:
        m = evaluate()
        hist.append({"iter": it, **m, "secs": round(time.time() - t0, 1)})
        star = ""
        if m["val"] < best:
            best = m["val"]; star = " *"
            model.save_pretrained(outdir)
            PreTrainedTokenizerFast(
                tokenizer_file=str(tokfile), bos_token="<|endoftext|>",
                eos_token="<|endoftext|>", unk_token="<|endoftext|>",
                pad_token="<|endoftext|>",
            ).save_pretrained(outdir)
        print(f"iter {it:5d}  train {m['train']:.4f}  val {m['val']:.4f}  "
              f"{time.time()-t0:6.1f}s{star}", flush=True)

# Keep the FINAL weights too, not just the best-val ones. On a one-book corpus
# early stopping picks an undertrained checkpoint that emits noise: val loss is
# measuring generalisation to unseen Joyce, which is not what we are after. The
# overfit end-of-run model is the one that actually speaks Wakese.
final = outdir / "final"
final.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final)
PreTrainedTokenizerFast(
    tokenizer_file=str(tokfile), bos_token="<|endoftext|>", eos_token="<|endoftext|>",
    unk_token="<|endoftext|>", pad_token="<|endoftext|>",
).save_pretrained(final)

(outdir / "trainlog.json").write_text(json.dumps(
    {"args": vars(a), "params": nparams, "best_val": best,
     "final_val": hist[-1]["val"] if hist else None,
     "final_train": hist[-1]["train"] if hist else None, "history": hist}, indent=2))
print(f"\nbest val loss {best:.4f} (ppl {math.exp(best):.1f}) -> {outdir}")
print(f"final weights (overfit, the playable one) -> {final}")
