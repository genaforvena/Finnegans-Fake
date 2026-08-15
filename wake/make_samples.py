#!/usr/bin/env python3
"""Write samples/ — long-form output from every configuration, on prompts the
book cannot answer, each file carrying the numbers that qualify it.

Why this exists. The repo does not distribute the Wake, and it does not
distribute weights either, so a reader has nothing to run. Every claim about
what these models sound like has so far been a sentence in a README with a
four-line excerpt under it, chosen by the person making the claim. Short
excerpts also flatter a character model badly: 200 characters is inside the
context window, so it is the one regime where the thing is at its most fluent.

So: ~4000 characters, which is well past the 512-token window and therefore
shows what the model does when it is continuing from its own tail; prompts that
are NOT 'riverrun', because prompting the Wake model with the Wake's first line
measures recitation and calls it style; and a header on each file with the
verbatim-overlap fraction, so 'is it just quoting Joyce?' has a number instead of
a shrug. The overlap comes from measure.py's own describe(), not a copy.

Every file is scored against BOTH corpora. A sample that is 'novel' against the
Wake may be ordinary English, and vice versa — one column cannot tell those
apart, and the product of experts is the exact case where the difference is the
whole question.

  python wake/make_samples.py                 # all configs, skips finished files
  python wake/make_samples.py --only product  # one family
"""
import argparse, datetime, json, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from measure import describe, load_reference
from gen import generate_chars
from product_sample import generate_product, load_pair, resolve

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

# Non-Wake prompts. The point is registers the book has no answer for: a model
# that only sounds Joycean when handed Joyce is a quoting machine, and the
# 'riverrun' control is kept precisely so that comparison is available rather
# than assumed.
PROMPTS = {
    "riverrun-control": "riverrun, past Eve and Adam's,",
    "recipe": "Preheat the oven to 180 degrees and butter a shallow dish.",
    "weather": "Rain will spread from the west overnight, with gusts of forty miles an hour along the coast.",
    "legal": "The parties hereto agree that, in the event of any dispute arising under this agreement,",
    "abstract": "We report the observation of a periodic signal in the residuals, significant at the three-sigma level.",
    "shell": "To list every file changed since the last commit, run git diff --name-only HEAD and pipe it to",
    "nursery": "Hey diddle diddle, the cat and the fiddle, the cow jumped over the moon.",
    "domestic": "I got home late and the front door was already open.",
    "kjv": "In the beginning God created the heaven and the earth.",
    "german": "Es war einmal ein kleines Maedchen, das wohnte am Rande des grossen Waldes.",
}

CHARS = 4000
TEMP = 0.9
TOP_P = 0.95
SEED = 0


def banner(lines):
    return "\n".join("# " + l if l else "#" for l in lines)


def loss_line(run):
    log = ROOT / "wake" / run / "trainlog.json"
    if not log.exists():
        return "best_val   (no trainlog.json — this checkpoint is not a finished run)"
    j = json.loads(log.read_text())
    b, f = j.get("best_val"), j.get("final_val")
    if f is None:
        h = j.get("history") or []
        f = h[-1].get("val") if h else None
    if b is None or f is None:
        return f"best_val   {b}   final_val {f}"
    verdict = ("final > best: this run OVERFIT, which is expected on a one-book "
               "corpus and is stated, not hidden") if f > b else "final <= best: no overfit"
    return f"best_val   {b:.4f}   final_val {f:.4f}   ({verdict})"


def score(text, refs):
    out = {}
    for name, ref in refs.items():
        d = describe(text, ref)
        out[name] = d
    return out


def score_lines(sc):
    lines = []
    for i, (name, d) in enumerate(sc.items()):
        if d is None:
            lines.append(f"{'overlap   ' if i == 0 else '          '} vs {name}: too short to score")
            continue
        lines.append(f"{'overlap   ' if i == 0 else '          '} vs {name:<14s} "
                     f"verb3 {d['verb3']:.3f}  verb5 {d['verb5']:.3f}  "
                     f"novel_words {d['novel_words']:.3f}")
    d = next(iter(sc.values()))
    if d:
        lines.append(f"           self_repeat {d['self_repeat']:.3f}  ttr {d['ttr']:.3f}  "
                     f"words {d['words']}")
    lines.append("           verb3/verb5 = output word 3-/5-grams found verbatim in that corpus;")
    lines.append("           novel_words = output word types absent from it. wake/measure.py.")
    return lines


def write(slug, config, header, prompt, text):
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{config}__{slug}.txt"
    p.write_text(banner(header) + "\n#\n" + "-" * 72 + "\n" + prompt + text + "\n",
                 encoding="utf-8")
    print(f"  wrote {p.relative_to(ROOT)} ({len(text)} chars)", flush=True)


def table():
    """Averages read back OUT of the committed files, not carried in memory from
    the run that wrote them. If a file were hand-edited or half-written the table
    would move with it, which is the only way the table stays a description of
    what is in samples/ rather than of what the generator believed it wrote."""
    import re, statistics
    rows = {}
    for f in sorted(OUT.glob("*.txt")):
        config = f.name.split("__")[0]
        h = f.read_text(encoding="utf-8").split("-" * 72)[0]
        d = {}
        for corpus in ("the Wake", "plain English"):
            m = re.search(rf"vs {re.escape(corpus)}\s+verb3 ([\d.]+)\s+verb5 ([\d.]+)\s+"
                          rf"novel_words ([\d.]+)", h)
            if m:
                key = "wake" if corpus == "the Wake" else "eng"
                d[f"verb3_{key}"], d[f"verb5_{key}"], d[f"novel_{key}"] = map(float, m.groups())
        m = re.search(r"self_repeat ([\d.]+)\s+ttr ([\d.]+)", h)
        if m:
            d["self_repeat"], d["ttr"] = map(float, m.groups())
        if d:
            rows.setdefault(config, []).append(d)

    cols = ["verb3_wake", "verb5_wake", "novel_wake", "verb3_eng", "verb5_eng",
            "novel_eng", "self_repeat", "ttr"]
    print(f"| config | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for config in sorted(rows):
        v = rows[config]
        cells = [f"{statistics.mean(x[c] for x in v if c in x):.3f}"
                 if any(c in x for x in v) else "" for c in cols]
        print(f"| {config} | {len(v)} | " + " | ".join(cells) + " |")
    print("\nverb3/verb5 = fraction of output word 3-/5-grams found verbatim in that")
    print("corpus; novel_* = fraction of output word TYPES absent from it. Averaged")
    print(f"over {len(next(iter(rows.values())))} prompts each. Properties, not a ranking.")


def coinages():
    """The sharpest form of the question the product was built to answer.

    novel_words in the header is measured against ONE corpus at a time, and a
    word can be novel to the Wake purely by being ordinary English. A coinage in
    the sense this project means it is a type present in NEITHER lexicon — the
    model made it up. So: rate, and the words themselves, because a rate with no
    examples is a number nobody can check.

    Read out of samples/, which is the artifact; nothing here re-runs a model, so
    the figures describe the committed text and not a fresh sample that happened
    to agree.
    """
    import re, statistics
    wake = set(re.findall(r"[a-z']+",
               (ROOT / "data" / "wake_clean.txt").read_text(encoding="utf-8").lower()))
    eng = set(re.findall(r"[a-z']+",
              (ROOT / "data" / "english_clean.txt").read_text(encoding="utf-8").lower()))
    known = wake | eng

    rows, examples = {}, {}
    for f in sorted(OUT.glob("*.txt")):
        config = f.name.split("__")[0]
        body = f.read_text(encoding="utf-8").split("-" * 72, 1)[1]
        # The file stores prompt+continuation, and the prompt is not the model's
        # invention. Left in, 'sigma' (from the abstract prompt) and the whole
        # German prompt scored as coinages for every config at once — a constant
        # added to every row, which is the kind of error that survives because it
        # does not change the ranking.
        slug = f.stem.split("__", 1)[1]
        body = body.lstrip("\n")
        if body.startswith(PROMPTS[slug]):
            body = body[len(PROMPTS[slug]):]
        w = re.findall(r"[a-z']+", body.lower())
        types = set(w)
        new = sorted(t for t in types if t not in known and len(t) > 3)
        rows.setdefault(config, []).append(len(new) / max(len(types), 1))
        examples.setdefault(config, []).extend(new)

    print("word types absent from BOTH data/wake_clean.txt and data/english_clean.txt")
    print("(len>3, averaged over the 10 prompts; these are the model's own inventions)\n")
    print(f"| config | coinage rate | examples |")
    print("|---|---|---|")
    for config in sorted(rows, key=lambda c: -statistics.mean(rows[c])):
        ex = sorted(set(examples[config]))
        pick = ex[::max(len(ex) // 12, 1)][:12]
        print(f"| {config} | {statistics.mean(rows[config]):.3f} | {' '.join(pick)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="wake-char257 | eng-char257 | bpe4096 | lora | product")
    ap.add_argument("--chars", type=int, default=CHARS)
    ap.add_argument("--force", action="store_true", help="rewrite files that already exist")
    ap.add_argument("--table", action="store_true",
                    help="re-read the written samples and print the summary table")
    ap.add_argument("--coinages", action="store_true",
                    help="rate and examples of word types absent from BOTH corpora")
    a = ap.parse_args()

    if a.table:
        return table()
    if a.coinages:
        return coinages()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    refs = {
        "the Wake": load_reference(ROOT / "data" / "wake_clean.txt"),
        "plain English": load_reference(ROOT / "data" / "english_clean.txt"),
    }
    stamp = datetime.date.today().isoformat()
    OUT.mkdir(exist_ok=True)

    def todo(config):
        return [(s, p) for s, p in PROMPTS.items()
                if a.force or not (OUT / f"{config}__{s}.txt").exists()]

    # ---- single character/BPE models -------------------------------------
    singles = [
        ("wake-char257", "wake-char257/final", "data/wake_clean.txt",
         "has read only Finnegans Wake"),
        ("eng-char257", "eng-char257/final", "data/english_clean.txt",
         "has read only ordinary English prose (8 public-domain novels)"),
        ("wake-bpe4096", "wake-bpe4096/final", "data/wake_clean.txt",
         "has read only Finnegans Wake, BPE-4096 instead of character level"),
    ]
    for config, path, corpus, what in singles:
        if a.only and a.only not in (config, "singles"):
            continue
        jobs = todo(config)
        if not jobs:
            continue
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mp = resolve(path)
        tok = AutoTokenizer.from_pretrained(str(mp))
        model = AutoModelForCausalLM.from_pretrained(str(mp)).to(dev).eval()
        run = config
        print(f"[{config}] {len(jobs)} to generate", flush=True)
        for slug, prompt in jobs:
            torch.manual_seed(SEED)
            text, slid = generate_chars(model, tok, prompt, a.chars, TEMP, TOP_P, dev,
                                        return_slid=True)
            hdr = [
                f"{config} · prompt '{slug}'",
                "",
                f"model      wake/{path}  (final/overfit weights — see README: on a",
                "           one-book corpus best-val is an undertrained checkpoint that emits",
                "           noise, so the end-of-run model is the one that speaks)",
                f"           {what}",
                f"corpus     {corpus}",
                loss_line(run),
                f"sampling   temp {TEMP}  top_p {TOP_P}  seed {SEED}  budget {a.chars} chars",
                f"window     {model.config.n_positions} tokens; context "
                + ("SLID — past the window this continues from its own tail, prompt gone"
                   if slid else "never slid (single pass)"),
            ] + score_lines(score(text, refs)) + [
                "", f"generated  {stamp} by wake/make_samples.py",
            ]
            write(slug, config, hdr, prompt, text)
        del model
        torch.cuda.empty_cache()

    # ---- product of experts ----------------------------------------------
    if not a.only or a.only == "product":
        weights = [0.3, 0.5, 0.7, 0.9]
        configs = {w: f"product-w{w:g}".replace(".", "") for w in weights}
        if any(todo(c) for c in configs.values()):
            (ma, tok, pa), (mb, _, pb), window, h = load_pair(
                "wake-char257/final", "eng-char257/final", dev)
            for w in weights:
                config = configs[w]
                jobs = todo(config)
                if not jobs:
                    continue
                print(f"[{config}] {len(jobs)} to generate", flush=True)
                for slug, prompt in jobs:
                    torch.manual_seed(SEED)
                    text, slid = generate_product(ma, mb, tok, prompt, a.chars, w,
                                                  TEMP, TOP_P, window, dev)
                    hdr = [
                        f"product of experts, w={w} · prompt '{slug}'",
                        "",
                        "method     combined = w*log_softmax(A) + (1-w)*log_softmax(B),",
                        "           renormalised, then temperature, then top-p. A GEOMETRIC",
                        "           mean: a character survives only if BOTH experts find it",
                        "           likely. The arithmetic mean would alternate between the",
                        "           two voices instead of conjoining them.",
                        f"A (w={w})   wake/wake-char257/final — has read only Finnegans Wake",
                        f"B (w={1-w:g})   wake/eng-char257/final — has read only ordinary English",
                        f"shared     tokenizer.json sha256 {h[:16]} — IDENTICAL in both, checked,",
                        f"           vocab {ma.config.vocab_size}. Equal sizes would not be enough: the same",
                        "           index can be a different character and the product would still",
                        "           run and still print plausible text.",
                        "best_val   A: " + loss_line("wake-char257").split("best_val   ")[-1],
                        "           B: " + loss_line("eng-char257").split("best_val   ")[-1],
                        f"sampling   temp {TEMP}  top_p {TOP_P}  seed {SEED}  budget {a.chars} chars",
                        f"window     {window} tokens (min of the two); context "
                        + ("SLID — past the window this continues from its own tail"
                           if slid else "never slid (single pass)"),
                    ] + score_lines(score(text, refs)) + [
                        "", f"generated  {stamp} by wake/make_samples.py",
                    ]
                    write(slug, config, hdr, prompt, text)
            del ma, mb
            torch.cuda.empty_cache()

    # ---- LoRA over a pretrained base --------------------------------------
    if not a.only or a.only == "lora":
        config = "wake-lora-base"
        jobs = todo(config)
        if jobs:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            adapter = ROOT / "wake" / "wake-lora-base"
            cfg = json.loads((adapter / "adapter_config.json").read_text())
            base_id = cfg["base_model_name_or_path"]
            log = json.loads((adapter / "trainlog.json").read_text())
            SYSTEM = log["system_prompt"]
            ltok = AutoTokenizer.from_pretrained(str(adapter))
            base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(dev)
            model = PeftModel.from_pretrained(base, str(adapter)).eval()
            print(f"[{config}] {len(jobs)} to generate", flush=True)
            for slug, prompt in jobs:
                torch.manual_seed(SEED)
                # Same prompt construction the adapter was trained under; a bare
                # continuation would be measuring the base model, not the LoRA.
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]
                try:
                    chat = ltok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    chat = f"{SYSTEM}\n\n{prompt}\n"
                ids = ltok(chat, return_tensors="pt").to(dev)
                text, rounds, stopped = "", 0, False
                while len(text) < a.chars and rounds < 6:
                    out = model.generate(**ids, max_new_tokens=700, do_sample=True,
                                         temperature=TEMP, top_p=TOP_P,
                                         pad_token_id=ltok.pad_token_id or ltok.eos_token_id)
                    new = ltok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                    text += new
                    rounds += 1
                    if not new.strip():
                        stopped = True
                        break
                    ids = ltok(chat + text, return_tensors="pt").to(dev)
                text = text[:a.chars]
                hdr = [
                    f"{config} · prompt '{slug}'",
                    "",
                    f"model      LoRA r={cfg['r']} alpha={cfg['lora_alpha']} over {base_id}",
                    "           — the only configuration here with an English prior it did not",
                    "           get from a corpus in this repo",
                    "corpus     data/finnegans_wake_dataset.jsonl (instruction pairs)",
                    loss_line(config),
                    f"prompting  the adapter's own chat template + its trained system prompt",
                    f"           ({SYSTEM!r})",
                    f"sampling   temp {TEMP}  top_p {TOP_P}  seed {SEED}  budget {a.chars} chars",
                    f"window     large (Qwen); no sliding. {rounds} generate round(s)"
                    + ("; model stopped early and had nothing further to add" if stopped else ""),
                ] + score_lines(score(text, refs)) + [
                    "", f"generated  {stamp} by wake/make_samples.py",
                ]
                write(slug, config, hdr, prompt + "\n\n", text)
            del model, base
            torch.cuda.empty_cache()

    print(f"\n{len(list(OUT.glob('*.txt')))} files in {OUT}")


if __name__ == "__main__":
    main()
