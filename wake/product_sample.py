#!/usr/bin/env python3
"""Product of experts over two character models: say two things at once.

One model has read only Finnegans Wake. One has read only ordinary English. At
every step both are asked what character comes next, and their answers are
multiplied rather than averaged:

    combined = w * log_softmax(logits_A) + (1-w) * log_softmax(logits_B)
    combined -= logsumexp(combined)          # renormalise
    probs = softmax(combined / temperature)  # then temp, then top-p, then sample

The multiplication is the whole argument. An ARITHMETIC mean (0.5*p_A + 0.5*p_B)
is a mixture: it puts mass wherever EITHER expert puts mass, so the text
alternates between the two voices — a stretch of Joyce, a stretch of Austen. The
GEOMETRIC mean is a conjunction: a character survives only if BOTH experts think
it likely, because a near-zero from either drives the product to zero. That is a
portmanteau's condition of existence — a string that is simultaneously readable
as two lexicons — so the aim is to get coinage by construction rather than by
luck.

The two experts MUST index the same symbols or the addition above is adding up
unrelated numbers, silently and plausibly. Both hard-checks are below and both
abort; neither is a warning.

  python wake/product_sample.py wake-char257/final eng-char257/final --compare
  python wake/product_sample.py wake-char257/final eng-char257/final \
      --weights 0.3,0.5,0.7 --prompt "Preheat the oven to 180 degrees."
"""
import argparse, hashlib, pathlib, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def resolve(name: str) -> pathlib.Path:
    p = pathlib.Path(name)
    if not p.exists():
        p = ROOT / "wake" / name
    if not p.exists():
        sys.exit(f"no such model: {name}")
    return p


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(name_a: str, name_b: str, device: str):
    """Load both experts and refuse to proceed unless they share a vocabulary.

    Checked explicitly rather than assumed. Two models trained with --vocab 257
    from the same tokenizer file is the INTENT; the artifact is the two files on
    disk hashing equal and the two configs agreeing on vocab_size. If they do
    not, every logit index means something different in each model and the
    product is arithmetic on nonsense that still runs and still prints text.
    """
    pa, pb = resolve(name_a), resolve(name_b)
    tok_a, tok_b = AutoTokenizer.from_pretrained(str(pa)), AutoTokenizer.from_pretrained(str(pb))
    ma = AutoModelForCausalLM.from_pretrained(str(pa)).to(device).eval()
    mb = AutoModelForCausalLM.from_pretrained(str(pb)).to(device).eval()

    if ma.config.vocab_size != mb.config.vocab_size:
        sys.exit(f"VOCAB MISMATCH: {pa} has {ma.config.vocab_size} tokens, "
                 f"{pb} has {mb.config.vocab_size}. A product of experts over "
                 f"differently-indexed logits is meaningless. Retrain both with "
                 f"the same --vocab.")
    fa, fb = pa / "tokenizer.json", pb / "tokenizer.json"
    if not (fa.exists() and fb.exists()):
        sys.exit(f"missing tokenizer.json under {pa} or {pb} — cannot prove a shared vocabulary")
    ha, hb = sha(fa), sha(fb)
    if ha != hb:
        sys.exit(f"TOKENIZER MISMATCH: {fa} sha256 {ha[:16]} != {fb} sha256 {hb[:16]}.\n"
                 f"Equal vocab SIZES are not a shared vocabulary — the same index "
                 f"can be a different character. Train both from one tokenizer file.")

    window = min(ma.config.n_positions, mb.config.n_positions)
    return (ma, tok_a, pa), (mb, tok_b, pb), window, ha


@torch.no_grad()
def generate_product(ma, mb, tok, prompt, n_chars, weight=0.5, temp=0.9, top_p=0.95,
                     window=512, device="cuda"):
    """Roll the context by hand — see wake/gen.py; `generate` walks off the
    position embedding table and CUDA aborts a long way from the cause.

    weight=1.0 is expert A alone, weight=0.0 is expert B alone, and they go
    through THIS function rather than a separate path so the three outputs in
    --compare differ in the mixing weight and in nothing else.

    Returns (text_after_prompt, slid) where `slid` says whether the context ever
    had to be truncated. A slid-window continuation is a different artifact from
    a single-pass one and the caller has to be able to say which it is holding.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = ids
    slid = False
    while True:
        text = tok.decode(out[0], skip_special_tokens=True)
        if len(text) >= len(prompt) + n_chars:
            return text[len(prompt):], slid
        if out.shape[1] > window - 1:
            slid = True
        ctx = out[:, -(window - 1):]
        la = ma(input_ids=ctx).logits[:, -1, :]
        lb = mb(input_ids=ctx).logits[:, -1, :]
        comb = weight * F.log_softmax(la, dim=-1) + (1.0 - weight) * F.log_softmax(lb, dim=-1)
        comb = comb - torch.logsumexp(comb, dim=-1, keepdim=True)      # renormalise
        probs = torch.softmax(comb / max(temp, 1e-5), dim=-1)
        srt, idx = torch.sort(probs, descending=True)
        keep = (torch.cumsum(srt, dim=-1) - srt) < top_p
        srt = torch.where(keep, srt, torch.zeros_like(srt))
        srt = srt / srt.sum(dim=-1, keepdim=True)
        nxt = idx.gather(-1, torch.multinomial(srt, 1))
        out = torch.cat([out, nxt], dim=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_a", help="expert A (the Wake, by convention)")
    p.add_argument("model_b", help="expert B (ordinary English, by convention)")
    p.add_argument("--weight", type=float, default=0.5, help="w on expert A")
    p.add_argument("--weights", default="", help="sweep, e.g. 0.3,0.5,0.7")
    p.add_argument("--prompt", default="riverrun, past Eve and Adam's,")
    p.add_argument("--tokens", type=int, default=600, help="CHARACTERS to generate")
    p.add_argument("--temp", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true",
                   help="A alone, B alone, and the product from the same prompt and seed")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (ma, tok, pa), (mb, _, pb), window, h = load_pair(a.model_a, a.model_b, dev)
    print(f"A = {pa}\nB = {pb}\nshared tokenizer sha256 {h[:16]}  vocab {ma.config.vocab_size}  "
          f"window {window} (min of {ma.config.n_positions}/{mb.config.n_positions})")
    print(f"prompt {a.prompt!r}  chars {a.tokens}  temp {a.temp}  top_p {a.top_p}  seed {a.seed}\n")

    runs = []
    if a.compare:
        runs += [("A alone (w=1.0)", 1.0), ("B alone (w=0.0)", 0.0)]
    if a.weights:
        runs += [(f"product w={w}", float(w)) for w in a.weights.split(",")]
    elif a.compare:
        runs += [(f"product w={a.weight}", a.weight)]
    if not runs:
        runs = [(f"product w={a.weight}", a.weight)]

    for label, w in runs:
        torch.manual_seed(a.seed)
        text, slid = generate_product(ma, mb, tok, a.prompt, a.tokens, w, a.temp,
                                      a.top_p, window, dev)
        print("=" * 72)
        print(f"{label}{'   [window slid]' if slid else ''}")
        print("=" * 72)
        print(a.prompt + text)
        print()


if __name__ == "__main__":
    main()
