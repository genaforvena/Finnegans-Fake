#!/usr/bin/env python3
"""Distil the board-folding behaviour into Qwen3.5-0.8B-Instruct with LoRA.

The measurement that motivates this: a hand-written fold of a board window shows
+0.0225 nats/char on source tokens it never quotes, while the same model's own
unaided fold shows -0.0012 — fluent, quoting MORE of the source, carrying nothing.
This trains on the hand-written folds to try to close that gap.

Two things are deliberate:

  loss on the fold only  The prompt (a whole board window) is masked out. We are
                         not teaching it to model the board, we are teaching it
                         what to emit given one.
  same prompt as the     The instruction is imported from make_folds.py verbatim,
  baseline               so the trained model is asked exactly what the untrained
                         baseline was asked. Changing the prompt would confound
                         the training with a prompt change.

Selection is NOT done here. Every epoch is saved, because val loss is the wrong
selector for this — the whole finding is that fluent-and-empty scores well on
likelihood. The adapters are judged afterwards by wake/condent.py.

  python train_fold.py --n 47 --epochs 3 --out fold-lora-47
"""
import argparse, json, math, pathlib, random, time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import make_folds

HERE = pathlib.Path(__file__).resolve().parent
BASE = make_folds.INSTRUCT

p = argparse.ArgumentParser()
p.add_argument("--base", default=BASE)
p.add_argument("--data", default=str(HERE / "teacher-folds.json"))
p.add_argument("--out", default="fold-lora")
p.add_argument("--n", type=int, default=0, help="training pairs to use (0 = all but val)")
p.add_argument("--n-val", type=int, default=8)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--maxlen", type=int, default=2048)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

rows = json.loads(pathlib.Path(a.data).read_text())
rng = random.Random(a.seed)
if any("split" in r for r in rows):
    # frozen split: val must not move when the training set grows, or the
    # val curve across dataset sizes compares different held-out rows
    val = [r for r in rows if r.get("split") == "val"]
    pool = [r for r in rows if r.get("split") != "val"]
else:
    rng.shuffle(rows)
    val, pool = rows[:a.n_val], rows[a.n_val:]
rng.shuffle(pool)
train = pool[:a.n] if a.n else pool
print(f"[data] {len(train)} train / {len(val)} val (of {len(rows)} pairs)")

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode(row):
    """(input_ids, labels) with the prompt masked to -100."""
    msgs = [{"role": "user", "content": make_folds.PROMPT + row["source"][:6000]}]
    pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                  return_dict=True, enable_thinking=False)["input_ids"]
    tgt = tok(row["summary"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (pre + tgt)[:a.maxlen]
    labels = ([-100] * len(pre) + tgt)[:a.maxlen]
    return ids, labels


class Folds(Dataset):
    def __init__(self, rows):
        self.items = [encode(r) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, lab, att


model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

dl = DataLoader(Folds(train), batch_size=1, shuffle=True, collate_fn=collate)
vl = DataLoader(Folds(val), batch_size=1, shuffle=False, collate_fn=collate)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)


def loss_of(ids, lab, att):
    """CE over the fold tokens only.

    The prompt is a whole board window and its logits are pure waste: a 248k
    vocabulary over ~1600 masked positions is gigabytes. `logits_to_keep` slices
    the hidden states before the output head, so only the tail is ever projected.
    """
    ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
    keep = int((lab[0] != -100).sum()) + 1
    out = model(input_ids=ids, attention_mask=att, logits_to_keep=keep)
    logits = out.logits[0, :-1]                     # predicts the last keep-1 tokens
    target = lab[0, -(keep - 1):]
    return torch.nn.functional.cross_entropy(logits.float(), target), keep - 1


def evaluate():
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for ids, lab, att in vl:
            l, k = loss_of(ids, lab, att)
            tot += l.item() * k
            n += k
    model.train()
    return tot / max(n, 1)


outdir = HERE / a.out
log = {"args": vars(a), "n_train": len(train), "history": []}
print(f"[val] step-0 {evaluate():.4f}")
t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train()
    run = 0.0
    for i, (ids, lab, att) in enumerate(dl, 1):
        loss, _ = loss_of(ids, lab, att)
        (loss / a.accum).backward()
        run += loss.item()
        if i % a.accum == 0 or i == len(dl):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
    v = evaluate()
    print(f"== epoch {ep} train {run/len(dl):.4f} val {v:.4f} ({time.time()-t0:.0f}s)")
    log["history"].append({"epoch": ep, "train": run / len(dl), "val": v})
    # every epoch is kept: val loss is not the selector, condent is
    model.save_pretrained(str(outdir / f"ep{ep}"))

(outdir / "trainlog.json").write_text(json.dumps(log, indent=1))
print(f"[out] {outdir}")
