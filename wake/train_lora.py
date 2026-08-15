#!/usr/bin/env python3
"""Bet 1 — LoRA a small instruct model into Wakese.

Keeps the base model's ability to hold a conversation and bends its style toward
the book, so you get something you can actually TALK to. Plain torch loop on
purpose: no Trainer/SFTTrainer API surface to drift under us.

  python train_lora.py --base Qwen/Qwen3-0.6B --epochs 3
"""
import argparse, json, math, pathlib, random, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ROOT = pathlib.Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen3-0.6B")
p.add_argument("--data", default="data/finnegans_wake_dataset.jsonl")
p.add_argument("--out", default="wake-lora")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--maxlen", type=int, default=512)
p.add_argument("--val-frac", type=float, default=0.03)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

SYSTEM = ("You speak in the language of Finnegans Wake: portmanteau, pun, "
          "dreamspeech, rivering syntax. Never explain yourself in plain English.")


class WakeSet(Dataset):
    """prompt tokens masked to -100; only the Wakese continuation is learned."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        conv = self.rows[i]["conversations"]
        user = next(m["content"] for m in conv if m["role"] == "user")
        asst = next(m["content"] for m in conv if m["role"] == "assistant")
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        pid = tok(prompt, add_special_tokens=False).input_ids
        aid = tok(asst + tok.eos_token, add_special_tokens=False).input_ids
        ids = (pid + aid)[: a.maxlen]
        labels = ([-100] * len(pid) + aid)[: a.maxlen]
        return torch.tensor(ids), torch.tensor(labels)


def collate(b):
    n = max(len(x) for x, _ in b)
    pad = tok.pad_token_id
    ids = torch.full((len(b), n), pad, dtype=torch.long)
    lab = torch.full((len(b), n), -100, dtype=torch.long)
    att = torch.zeros((len(b), n), dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, : len(x)] = x
        lab[i, : len(y)] = y
        att[i, : len(x)] = 1
    return ids, lab, att


rows = [json.loads(l) for l in (ROOT / a.data).read_text().splitlines() if l.strip()]
random.Random(0).shuffle(rows)
nval = max(16, int(len(rows) * a.val_frac))
train_dl = DataLoader(WakeSet(rows[nval:]), batch_size=a.batch, shuffle=True, collate_fn=collate)
val_dl = DataLoader(WakeSet(rows[:nval]), batch_size=a.batch, collate_fn=collate)
print(f"train={len(rows)-nval} val={nval} base={a.base}")

model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
steps = math.ceil(len(train_dl) / a.accum) * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)


@torch.no_grad()
def val_loss():
    model.eval()
    tot = ntok = 0.0
    for ids, lab, att in val_dl:
        out = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev))
        k = (lab != -100).sum().item()
        tot += out.loss.item() * k
        ntok += k
    model.train()
    return tot / max(ntok, 1)


outdir = ROOT / "wake" / a.out
best, hist, t0, step = math.inf, [], time.time(), 0
print(f"step0 val {val_loss():.4f}", flush=True)
for ep in range(1, a.epochs + 1):
    for i, (ids, lab, att) in enumerate(train_dl, 1):
        loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
        (loss / a.accum).backward()
        if i % a.accum == 0:
            torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{steps} train {loss.item():.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
    v = val_loss()
    hist.append({"epoch": ep, "val": v, "secs": round(time.time() - t0, 1)})
    star = ""
    if v < best:
        best, star = v, " *"
        model.save_pretrained(outdir); tok.save_pretrained(outdir)
    print(f"== epoch {ep} val {v:.4f} (ppl {math.exp(v):.1f}){star}", flush=True)

(outdir / "trainlog.json").write_text(json.dumps(
    {"args": vars(a), "best_val": best, "history": hist, "system_prompt": SYSTEM}, indent=2))
print(f"\nbest val {best:.4f} -> {outdir}")
