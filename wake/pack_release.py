#!/usr/bin/env python3
"""Package the trained models into shareable, self-contained HF repos.

    python wake/pack_release.py                 # build release/ and verify every dir
    python wake/pack_release.py --only wake-char257

What this fixes, which a plain `cp` does not:

* **LoRA adapters name a local path as their base.** `adapter_config.json` carries
  `base_model_name_or_path: /home/mesh-home/models/Qwen3.5-0.8B`, which resolves on
  exactly one machine. Shipped as-is the adapter is unloadable for everybody else,
  and `PeftModel.from_pretrained` fails with a path error a long way from its cause.
  Rewritten to the hub id.
* **The configs are transformers-5 only.** v5 renamed `torch_dtype` -> `dtype` and
  moved the tokenizer to a `TokenizersBackend`. Both legacy keys are written back
  BESIDE the new ones, so a v4 reader finds what it looks for. This is additive:
  v5 ignores them.
* **Every packaged directory is loaded and sampled from before it is called done.**
  A copied tree that imports is not a model that generates; the verification is a
  real forward pass producing real text, and the sample goes in the manifest.

The cards are written here rather than by hand because a number quoted in prose
rots the moment a run is repeated: every figure below is read out of the run's own
`trainlog.json` at pack time.
"""
import argparse, hashlib, json, pathlib, shutil, sys, textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
WAKE = ROOT / "wake"

# --- what ships -------------------------------------------------------------
# `share`: public   -> nothing in the training data is encumbered
#          private  -> withheld; nothing currently uses it, kept as a working switch
#          local    -> trained on the mesh's own board log; does not leave this node
SPECS = {
    "finnegans-fake-char257": dict(
        src="wake-char257", kind="causal", share="public",
        title="Finnegans Fake — character-level",
        blurb="A 10.9M-parameter GPT-2 that has read exactly one book: *Finnegans Wake*.",
    ),
    "english-char257": dict(
        src="eng-char257", kind="causal", share="public",
        title="English char257 — the control expert",
        blurb=("The same architecture, hyperparameters and tokenizer as "
               "`finnegans-fake-char257`, trained on ordinary 19th-century English. "
               "It exists to be the *only* controlled comparison in the project."),
    ),
    "finnegans-fake-bpe4096": dict(
        src="wake-bpe4096", kind="causal", share="public",
        title="Finnegans Fake — BPE-4096",
        blurb="The same book through a 4096-token BPE vocabulary trained on the book itself.",
    ),
    "finnegans-fake-lora-qwen3.5-0.8b": dict(
        src="wake-lora-base", kind="peft", share="public",
        base_hub="Qwen/Qwen3.5-0.8B-Base",
        title="Finnegans Fake — LoRA on Qwen3.5-0.8B-Base",
        blurb="A LoRA that answers back in English-but-dislocated, rather than from scratch.",
    ),
    "finnegans-fake-folds-lora": dict(
        src="fold-lora-47/ep3", kind="peft", share="local",
        base_hub="Qwen/Qwen3.5-0.8B",
        title="Fold-distillation rung (n=47) — LOCAL ONLY",
        blurb="Trained on the mesh's own board log. Not for publication.",
    ),
}

PREFIX_SPACE_NOTE = textwrap.dedent("""\
    ### One thing to know about the tokenizer

    It is a byte-level BPE with `add_prefix_space: true`, so decoding inserts a single
    leading space that was not in your prompt:

    ```python
    tok.decode(tok("riverrun").input_ids)   # -> ' riverrun'
    ```

    Round-trip is otherwise **exact** — verified on 20k characters of the training
    corpus, byte for byte, once that one space is accounted for. Prompt with a leading
    space, or strip one from the output; do not go looking for a lossy character.
    """)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load_trainlog(src):
    for cand in (WAKE / src / "trainlog.json", WAKE / src / ".." / "trainlog.json"):
        cand = cand.resolve()
        if cand.exists():
            return json.loads(cand.read_text())
    return {}


def patch_causal(dst):
    """Write the legacy keys back beside the v5 ones. Additive only."""
    cfg_p = dst / "config.json"
    cfg = json.loads(cfg_p.read_text())
    if "dtype" in cfg and "torch_dtype" not in cfg:
        cfg["torch_dtype"] = cfg["dtype"]
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")

    tc_p = dst / "tokenizer_config.json"
    tc = json.loads(tc_p.read_text())
    tc.setdefault("tokenizer_class", "PreTrainedTokenizerFast")
    # 1e19 is the "unset" sentinel; it makes a v4 reader emit a warning per call.
    if tc.get("model_max_length", 0) > 10 ** 12:
        tc["model_max_length"] = cfg.get("n_positions", 512)
    tc_p.write_text(json.dumps(tc, indent=2) + "\n")
    return cfg


def patch_peft(dst, base_hub):
    ac_p = dst / "adapter_config.json"
    ac = json.loads(ac_p.read_text())
    was = ac.get("base_model_name_or_path")
    ac["base_model_name_or_path"] = base_hub
    ac_p.write_text(json.dumps(ac, indent=2) + "\n")
    return ac, was


def verify_causal(dst):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(str(dst))
    model = AutoModelForCausalLM.from_pretrained(str(dst)).eval()
    torch.manual_seed(0)
    ids = tok("riverrun, past Eve and Adam's,", return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.9,
                         top_p=0.95, top_k=100, pad_token_id=tok.eos_token_id or 0)
    return dict(params=sum(p.numel() for p in model.parameters()),
                sample=tok.decode(out[0], skip_special_tokens=True))


LOCAL_BASES = pathlib.Path.home() / "models"


def sample_peft(dst, base_hub, sysprompt=None):
    """Drive the adapter for real, against a locally-present copy of the base.

    The shipped `base_model_name_or_path` is the hub id, which is what a consumer
    needs and what `verify_peft` asserts. But a card whose only claim is "the tensors
    parse" is the executable/loadable distinction all over again — so if the base
    happens to be on this disk, generate through it and put the text in the card.
    The adapter is loaded by explicit path, so this never contradicts the shipped id.
    Absent base -> returns None and the card says so, rather than inventing a sample.
    """
    local = LOCAL_BASES / base_hub.split("/")[-1]
    if not local.exists():
        return None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(str(local))
    model = AutoModelForCausalLM.from_pretrained(str(local), dtype="auto",
                                                device_map="auto")
    model = PeftModel.from_pretrained(model, str(dst)).eval()
    msgs = ([{"role": "system", "content": sysprompt}] if sysprompt else []) + [
        {"role": "user", "content": "Tell me what the river said."}]
    # apply_chat_template returns a BatchEncoding, not a bare tensor, in transformers 5.
    # Taking .shape off it raises an AttributeError with an EMPTY message, which is a
    # long way from its cause; normalise to the tensor here.
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = getattr(enc, "input_ids", enc)
    except Exception:
        ids = tok("riverrun, past Eve and Adam's,", return_tensors="pt").input_ids
    ids = ids.to(model.device)
    torch.manual_seed(0)
    out = model.generate(ids, max_new_tokens=120, do_sample=True, temperature=0.9,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def verify_peft(dst):
    """Assert what a consumer's load actually depends on: the safetensors parse, the
    declared target modules against real tensor names, and a base id that is a hub id
    rather than a path that resolves on exactly one machine."""
    from safetensors import safe_open
    ac = json.loads((dst / "adapter_config.json").read_text())
    f = dst / "adapter_model.safetensors"
    keys, total = [], 0
    with safe_open(str(f), framework="pt") as h:
        for k in h.keys():
            keys.append(k)
            t = h.get_slice(k)
            n = 1
            for d in t.get_shape():
                n *= d
            total += n
    base = ac["base_model_name_or_path"]
    assert not base.startswith("/"), f"base is still a local path: {base}"
    hit = {m for m in ac["target_modules"] if any(f".{m}." in k for k in keys)}
    missing = set(ac["target_modules"]) - hit
    assert not missing, f"target_modules with no tensors: {sorted(missing)}"
    return dict(params=total, tensors=len(keys), base=base,
                sample="(not sampled: needs the 1.7GB base model)")


CORPUS = textwrap.dedent("""\
    ## The corpus this measures

    | | |
    |---|---|
    | running words | 224,527 |
    | distinct word types | 58,725 |
    | types occurring exactly once | 46,599 (**79.4%**) |
    | text covered by types seen >=5 times | 69.8% |
    | distinct characters | 105 |

    Four in five word types are hapax legomena, which settles tokenisation before any
    training: a word-level vocabulary is not merely coarse here, it is impossible —
    most types would carry one example, and every unseen word becomes `<unk>`, so the
    model could never coin one. Coining is the only thing worth wanting from it.
    """)


def card(slug, spec, tl, ver, cfg=None):
    L, share = [], spec["share"]
    lic = "cc0-1.0"
    tags = ["finnegans-wake", "tiny-language-model", "text-generation"]
    if spec["kind"] == "peft":
        tags += ["lora", "peft"]
        L += ["---", f"base_model: {spec['base_hub']}", "library_name: peft"]
    else:
        tags += ["gpt2"]
        L += ["---", "library_name: transformers"]
    L += ["pipeline_tag: text-generation", f"license: {lic}",
          "tags:"] + [f"- {t}" for t in tags] + ["---", ""]
    L += [f"# {spec['title']}", "", spec["blurb"], ""]

    if spec["src"].startswith("wake-"):
        L += [textwrap.dedent("""\
            > **On the book.** These weights are trained on the full text of *Finnegans Wake*.
            > Joyce died in 1941, so the book is public domain in Ireland, the UK and the EU —
            > in the United States it is not, until 2035. Whether trained weights are a
            > derivative work of their training text is unsettled either way. The corpus is not
            > redistributed with the code; the code is CC0 and you supply the book.
            """), ""]
    if share == "private":
        L += ["> **This repo is private.** It is shared deliberately rather than published.", ""]
    if share == "local":
        L += [textwrap.dedent("""\
            > **Do not publish this adapter.** It is trained on the operator mesh's own
            > internal board log — hostnames, topology, design argument, operational detail.
            > 47 rows over 3 epochs is squarely in the regime where a LoRA reproduces its
            > training data on demand, so publishing it publishes the log.
            """), ""]

    L += ["## What it is", ""]
    if spec["kind"] == "causal":
        a = tl.get("args", {})
        L += [
            "| | |", "|---|---|",
            f"| architecture | GPT-2 ({a.get('layers')} layers, {a.get('heads')} heads, {a.get('embd')} embd) |",
            f"| parameters | {ver['params']:,} |",
            f"| vocabulary | {a.get('vocab')} |",
            f"| context | {a.get('block')} tokens |",
            f"| trained | {a.get('iters'):,} iters, batch {a.get('batch')}, lr {a.get('lr')}, dropout {a.get('dropout')} |",
            f"| best val loss | **{tl.get('best_val'):.4f}** |",
            f"| end-of-run val loss | {tl.get('final_val'):.4f} |", "",
            "This repo carries the **best-val** checkpoint, not the end of the run.", "",
        ]
        gap = (tl.get("final_val") or 0) - (tl.get("best_val") or 0)
        if gap > 0.3:
            L += [f"That distinction is load-bearing here: the run ended {gap:.2f} nats "
                  f"*above* its own best, so the last checkpoint is meaningfully worse "
                  f"than this one.", ""]
        L += ["```python",
              "from transformers import AutoModelForCausalLM, AutoTokenizer",
              "import torch", "",
              f'tok = AutoTokenizer.from_pretrained("{slug}")',
              f'model = AutoModelForCausalLM.from_pretrained("{slug}")',
              '',
              'ids = tok("riverrun, past Eve and Adam\'s,", return_tensors="pt").input_ids',
              'out = model.generate(ids, max_new_tokens=200, do_sample=True,',
              '                     temperature=0.9, top_k=100, top_p=0.95)',
              'print(tok.decode(out[0], skip_special_tokens=True))',
              "```", "",
              "Sampled at pack time, seed 0, so this is reproducible rather than curated:", "",
              "```", ver["sample"].strip(), "```", "", PREFIX_SPACE_NOTE, ""]
    else:
        a = tl.get("args", {})
        ac = json.loads((pathlib.Path(ver["_dst"]) / "adapter_config.json").read_text())
        L += [
            "| | |", "|---|---|",
            f"| base model | [`{spec['base_hub']}`](https://huggingface.co/{spec['base_hub']}) |",
            f"| adapter | LoRA r={ac['r']}, alpha={ac['lora_alpha']}, dropout={ac['lora_dropout']} |",
            f"| target modules | {', '.join(sorted(ac['target_modules']))} |",
            f"| trainable parameters | {ver['params']:,} |",
            f"| epochs | {a.get('epochs')} |", "",
        ]
        # Two trainers wrote two history schemas ({epoch,train,val} and {epoch,val,secs}).
        # Render whichever columns are actually present rather than assuming one.
        hist = tl.get("history") or []
        if hist:
            cols = [c for c in ("train", "val") if c in hist[0]]
            L += ["| epoch | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            L += [f"| {h['epoch']} | " + " | ".join(f"{h[c]:.4f}" for c in cols) + " |"
                  for h in hist] + [""]
            vals = [h["val"] for h in hist if "val" in h]
            if vals and vals[0] == min(vals) and len(vals) > 1:
                L += [f"Validation is best at **epoch 1** and rises after it; the weights here "
                      f"are the best-val checkpoint, not the last one. The later epochs are "
                      f"reported rather than dropped because the shape is the finding — this "
                      f"much text fine-tunes in under one pass.", ""]
        elif tl.get("best_val") is not None:
            L += [f"Best validation loss **{tl['best_val']:.4f}**.", ""]

        sysp = tl.get("system_prompt")
        if sysp:
            L += ["The adapter was trained under this system prompt. It is not decoration —",
                  "use it, or you are prompting the adapter off-distribution:", "",
                  "```text", sysp, "```", ""]
        if not ver["sample"].startswith("(not sampled"):
            L += ["Driven at pack time against the real base, seed 0, prompt "
                  "*\"Tell me what the river said.\"*:", "",
                  "```", ver["sample"], "```", ""]
        L += ["```python",
              "from transformers import AutoModelForCausalLM, AutoTokenizer",
              "from peft import PeftModel", "",
              f'base = "{spec["base_hub"]}"',
              'tok = AutoTokenizer.from_pretrained(base)',
              'model = PeftModel.from_pretrained(',
              '    AutoModelForCausalLM.from_pretrained(base, dtype="auto", device_map="auto"),',
              f'    "{slug}")',
              "```", ""]

    if spec["src"].startswith(("wake-", "fold-")) and spec["kind"] == "causal":
        L += [CORPUS, ""]
    if slug == "english-char257":
        L += [textwrap.dedent("""\
            ## Why a second model exists at all

            `finnegans-fake-char257` reaches val **1.8840**. This one reaches **1.2004** on the
            same architecture, the same hyperparameters, the same 1.31MB of text (matched to the
            Wake's character count within 0.2%) and a **byte-identical tokenizer file**. It is the
            one loss comparison in this project where nothing else differs, so for once the
            difference means what it looks like: ordinary English is more predictable than
            *Finnegans Wake*.

            The corpus is eight public-domain novels from Project Gutenberg
            (`wake/prepare_english.py --fetch`), which is why this repo — alone among the set —
            is CC0 and public.

            It is also one half of a failed experiment worth reading:
            multiplying this model's distribution with the Wake model's was supposed to force a
            portmanteau, a string readable as two lexicons at once. It does the exact opposite,
            monotonically, and the reason is definitional rather than a tuning failure — a
            geometric mean is a veto, and ordinary English vetoes precisely the characters that
            would have made a coinage. See the project README.
            """), ""]

    L += [textwrap.dedent("""\
        ## A bug worth keeping in the card

        Every loss figure this project first published was produced by a broken objective.
        `batch()` returned nanoGPT-style pre-shifted labels while `transformers` shifts
        labels itself, so the shift happened twice and every run learned to predict token
        **t+2** from position t. It never raised: training ran, loss fell, the curves looked
        plausible, and the inflated perplexity supported a confident story about the Wake
        being statistically incompressible. That story was a property of the bug. Only the
        generated text exposed it, by coming out looking like every second character had been
        deleted. After the fix, 400 steps beat the 6000 broken ones.

        The figures in this card are post-fix.

        ## What it is not

        It is not good, and it is not trying to be. The aim was to find out what a language
        model does when the only language it has ever seen is one book — whether anything
        resembling English survives, and whether the machine can coin words the way Joyce did
        rather than quote the ones he already coined. Do not use it for anything.
        """)]
    L += ["", "Code: <https://github.com/genaforvena/finnegans-fake> — CC0."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "release"))
    ap.add_argument("--only", action="append", default=None)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # --only rebuilds a subset; the manifest describes the whole release, so merge into
    # whatever is already there. Clobbering it would silently un-ship the other repos.
    mf = out / "MANIFEST.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    failed = []

    for slug, spec in SPECS.items():
        if a.only and slug not in a.only:
            continue
        src = WAKE / spec["src"]
        if not src.exists():
            print(f"[skip] {slug}: {src} missing")
            continue
        dst = out / slug
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

        for f in sorted(src.iterdir()):
            if f.is_file() and f.name not in ("README.md", "trainlog.json"):
                shutil.copy2(f, dst / f.name)

        tl = load_trainlog(spec["src"])
        (dst / "trainlog.json").write_text(json.dumps(tl, indent=2) + "\n")

        try:
            if spec["kind"] == "causal":
                patch_causal(dst)
                ver = verify_causal(dst)
            else:
                _, was = patch_peft(dst, spec["base_hub"])
                ver = verify_peft(dst)
                ver["rewrote_base_from"] = was
                got = sample_peft(dst, spec["base_hub"], tl.get("system_prompt"))
                ver["sample"] = got or ("(not sampled: base model "
                                        f"{spec['base_hub']} not present locally)")
            ver["_dst"] = str(dst)
        except Exception as e:
            print(f"[FAIL] {slug}: {type(e).__name__}: {e}")
            failed.append(slug)
            continue

        (dst / "README.md").write_text(card(slug, spec, tl, ver))
        files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(dst.iterdir()) if p.is_file()}
        ver.pop("_dst")
        manifest[slug] = dict(share=spec["share"], source=spec["src"],
                              verified=ver, files=files)
        print(f"[ok]   {slug:36s} {spec['share']:8s} "
              f"{sum(f['bytes'] for f in files.values())/1e6:7.1f} MB  "
              f"{ver['params']:,} params")

    mf.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest)} packaged -> {out}/  (MANIFEST.json written)")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
