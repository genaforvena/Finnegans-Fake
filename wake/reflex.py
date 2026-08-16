#!/usr/bin/env python3
"""Answer the operator's Telegram message with one of the trained models, chosen at random.

The mesh already routes `/wake <text>` from Telegram; this is what sits at the end
of that route. The operator's message is not a question the model can answer — it
is a SEED the model continues. So the reply is a continuation, and it carries the
id of whichever model produced it, because that is the whole point: nine voices,
and you are told which one spoke.

Nothing here is a mind. This is a reflex in the sense mesh-url-watch uses the word:
a fixed mechanical path from a message to an action, no inference about intent.

  reflex.py "what is the mesh?"        pick at random, print a Telegram-ready reply
  reflex.py --id wake-bpe4096 "..."    force one model (for testing)
  reflex.py --list                     print the pool
  reflex.py --test                     smoke test: every id resolves on disk (no GPU)

Two hazards this file exists to route around, both already documented in the repo
and both silent when hit:

  1. `model.generate` walks off the position embedding table. These windows are 256
     or 512 tokens; a long Telegram message plus 300 characters of continuation runs
     the position ids past the table and CUDA aborts the process a long way from the
     cause (wake/gen.py). Every path here rolls the context by hand instead, so the
     reflex is safe against a message of any length.
  2. A product of experts over two differently-indexed vocabularies is arithmetic on
     nonsense that still runs and still prints plausible text (wake/product_sample.py).
     load_pair() refuses unless the two tokenizer files hash equal. That check is
     load-bearing here too, so it is not bypassed for speed.
"""
import argparse
import fcntl
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "wake"))

# The pool the operator chose: the five models that read the Wake, plus the product
# of experts at four weights. eng-char257 is deliberately absent as a voice of its
# own — it is the control, trained only on ordinary English, and its job here is to
# be expert B inside the product rather than to answer on its own.
SINGLE = [
    "wake-char257",
    "wake-char257-long",
    "wake-bpe4096",
    "wake-bpe4096-small",
    "wake-bpe8192",
]
PRODUCT_A = "wake-char257"   # the Wake expert
PRODUCT_B = "eng-char257"    # ordinary English
PRODUCT_WEIGHTS = [0.3, 0.5, 0.7, 0.9]

STATE = pathlib.Path(os.environ.get("MESH_HOME", pathlib.Path.home() / ".mesh"))
LAST = STATE / ".wake-reflex-last"
LOCK = STATE / ".wake-reflex.lock"


def product_id(w: float) -> str:
    """0.3 -> 'product-w03'. Matches the sample filenames already committed."""
    return f"product-w{int(round(w * 10)):02d}"


def pool() -> list:
    return SINGLE + [product_id(w) for w in PRODUCT_WEIGHTS]


def resolve(name: str) -> pathlib.Path:
    p = pathlib.Path(name)
    return p if p.exists() else ROOT / "wake" / name


def pick(exclude_last: bool = True) -> str:
    """Uniform over the pool, minus whatever answered last time.

    Without the exclusion two consecutive asks land on the same voice about one
    time in nine, which reads as the reflex being broken rather than as chance.
    The state file is advisory: if it is unreadable we just do not exclude.
    """
    choices = pool()
    if exclude_last:
        try:
            last = LAST.read_text().strip()
            if last in choices and len(choices) > 1:
                choices = [c for c in choices if c != last]
        except OSError:
            pass
    return random.choice(choices)


def remember(mid: str) -> None:
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        LAST.write_text(mid + "\n")
    except OSError:
        pass          # advisory only — never fail a reply over the repeat-guard


class GpuLock:
    """Serialise generations so two fast asks do not stack on the 3060.

    Held for the load+generate of one reply, which measures ~2s single / ~6s product.
    Nonblocking retry rather than a blocking flock so a stuck holder surfaces as a
    message to the operator instead of a Telegram reply that never comes.
    """

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout
        self.fh = None

    def __enter__(self):
        STATE.mkdir(parents=True, exist_ok=True)
        self.fh = open(LOCK, "w")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() >= deadline:
                    self.fh.close()
                    raise TimeoutError(
                        f"another generation has held the GPU for over {self.timeout:.0f}s"
                    )
                time.sleep(1.5)

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


def generate(mid: str, prompt: str, n_chars: int, temp: float, top_p: float):
    """Return (continuation, slid). Imports torch lazily so --test and --list stay fast."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if mid.startswith("product-"):
        from product_sample import generate_product, load_pair

        weight = int(mid.split("w")[-1]) / 10.0
        (ma, tok, _), (mb, _, _), window, _ = load_pair(PRODUCT_A, PRODUCT_B, dev)
        return generate_product(ma, mb, tok, prompt, n_chars, weight=weight,
                                temp=temp, top_p=top_p, window=window, device=dev)

    from gen import generate_chars

    path = resolve(mid)
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path)).to(dev).eval()
    return generate_chars(model, tok, prompt, n_chars, temp=temp, top_p=top_p,
                          device=dev, return_slid=True)


def smoke() -> int:
    """Every id in the pool must resolve to weights on disk. No GPU, no model load."""
    fail = 0
    for mid in pool():
        needed = [resolve(PRODUCT_A), resolve(PRODUCT_B)] if mid.startswith("product-") \
            else [resolve(mid)]
        missing = [str(p) for p in needed
                   if not (p / "config.json").exists() or not (p / "tokenizer.json").exists()]
        if missing:
            print(f"  FAIL: {mid} -> missing {', '.join(missing)}")
            fail = 1
        else:
            print(f"  ok: {mid}")
    # The product's shared-vocabulary check is the one that fails silently and
    # plausibly, so assert it here rather than discovering it mid-reply.
    import hashlib
    ta = resolve(PRODUCT_A) / "tokenizer.json"
    tb = resolve(PRODUCT_B) / "tokenizer.json"
    if ta.exists() and tb.exists():
        ha = hashlib.sha256(ta.read_bytes()).hexdigest()
        hb = hashlib.sha256(tb.read_bytes()).hexdigest()
        if ha == hb:
            print(f"  ok: product experts share a tokenizer (sha256 {ha[:16]})")
        else:
            print(f"  FAIL: product experts disagree ({ha[:16]} != {hb[:16]}) — "
                  f"the product would be arithmetic on nonsense")
            fail = 1
    print("smoke-test: " + ("FAIL" if fail else f"ok ({len(pool())} ids)"))
    return fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", help="the operator's message; used as the seed")
    ap.add_argument("--id", default=None, help="force a model id instead of drawing one")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repeat-ok", action="store_true", help="allow the same id twice running")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    if a.test:
        return smoke()
    if a.list:
        print("\n".join(pool()))
        return 0

    prompt = " ".join(a.prompt).strip()
    if not prompt:
        print("usage: /wake <message> — a model continues what you wrote")
        return 2

    mid = a.id or pick(exclude_last=not a.repeat_ok)
    if mid not in pool() and not a.id:
        print(f"internal: drew an id outside the pool ({mid!r})")
        return 1

    try:
        with GpuLock():
            text, slid = generate(mid, prompt, a.chars, a.temp, a.top_p)
    except TimeoutError as e:
        print(f"busy: {e}")
        return 3
    except SystemExit as e:
        # load_pair() exits on a vocabulary mismatch. Turn that into a reply rather
        # than a silent nonzero, so the operator learns why nothing came back.
        print(f"refused: {e}")
        return 1
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        return 1

    if not a.repeat_ok and not a.id:
        remember(mid)

    text = text.strip()
    if not text:
        print(f"{mid} produced nothing from that seed")
        return 1
    # `slid` means the context was rolled: past the window the model is continuing
    # from a tail of its own output with the seed gone. That is a different artifact
    # from a single-pass completion and the reader is told which one they hold.
    mark = " (window rolled)" if slid else ""
    print(f"🎲 {mid}{mark} ↩︎ {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
