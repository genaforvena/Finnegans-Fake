"""Shared sampling that can outrun the model's context window.

These models have a 256- or 512-token window. Asking `generate` for more than
that does not truncate politely — the position ids run past the embedding table
and CUDA aborts the process with an indexing assert, which is a hard crash a long
way from its cause. So the context is rolled by hand: keep the last window-1
tokens, sample the next one, repeat.

This also keeps a CHARACTER budget honest across tokenisations. A character-level
model needs roughly three times as many tokens as BPE-4096 for the same text, so
any token budget silently favours the coarse tokenizer — the very axis these
comparisons exist to measure.
"""
import torch


def generate_chars(model, tok, prompt, n_chars, temp=0.85, top_p=0.95, device="cuda"):
    window = model.config.n_positions
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids
    while True:
        text = tok.decode(out[0], skip_special_tokens=True)
        if len(text) >= len(prompt) + n_chars:
            return text[len(prompt):]
        ctx = out[:, -(window - 1):]
        with torch.no_grad():
            logits = model(input_ids=ctx).logits[:, -1, :] / max(temp, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        srt, idx = torch.sort(probs, descending=True)
        keep = (torch.cumsum(srt, dim=-1) - srt) < top_p
        srt = torch.where(keep, srt, torch.zeros_like(srt))
        srt = srt / srt.sum(dim=-1, keepdim=True)
        nxt = idx.gather(-1, torch.multinomial(srt, 1))
        out = torch.cat([out, nxt], dim=-1)
