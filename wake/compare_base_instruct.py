#!/usr/bin/env python3
"""Controlled Base-vs-Instruct style comparison.

This answers the open post-mortem question without pretending that losses from
different training objectives are comparable. Both raw checkpoints receive the
same plain-text prompts, seed, sampling settings, and character budget. The
result records the model/tokenizer hashes and current git state beside every
metric so the prose can be regenerated from this file.
"""
import argparse, datetime, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = {
    "recipe": "Preheat the oven to 180 degrees and butter a shallow dish.",
    "weather": "Rain will spread from the west overnight, with gusts of forty miles an hour along the coast.",
    "legal": "The parties hereto agree that, in the event of any dispute arising under this agreement,",
    "abstract": "We report the observation of a periodic signal in the residuals, significant at the three-sigma level.",
    "domestic": "I got home late and the front door was already open.",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/mesh-home/models/Qwen3.5-0.8B-Base")
    ap.add_argument("--instruct", default="/home/mesh-home/models/Qwen3.5-0.8B")
    ap.add_argument("--out", default=str(ROOT / "wake" / "base-instruct-results.json"))
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / "wake"))
    from measure import describe, load_reference

    refs = {"wake": load_reference(ROOT / "data" / "wake_clean.txt"),
            "english": load_reference(ROOT / "data" / "english_clean.txt")}
    result = {
        "method": {"same_plain_text_prompts": True, "same_seed": a.seed,
                    "same_character_budget": a.chars, "temperature": a.temperature,
                    "top_p": a.top_p, "interpretation": "properties, not a loss ranking"},
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": git_state(), "corpora": {
            "wake": {"path": "data/wake_clean.txt", "sha256": sha(ROOT / "data/wake_clean.txt")},
            "english": {"path": "data/english_clean.txt", "sha256": sha(ROOT / "data/english_clean.txt")},
        },
        "models": {},
    }

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for label, model_path in (("base", pathlib.Path(a.base)), ("instruct", pathlib.Path(a.instruct))):
        tok = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16 if dev == "cuda" else torch.float32
        ).to(dev).eval()
        model_result = {"path": str(model_path),
                        "config_sha256": sha(model_path / "config.json"),
                        "tokenizer_sha256": sha(model_path / "tokenizer.json"),
                        "samples": {}}
        for name, prompt in PROMPTS.items():
            torch.manual_seed(a.seed)
            ids = tok(prompt, return_tensors="pt").to(dev)
            target = max(1, a.chars // 4)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=target, do_sample=True,
                                     temperature=a.temperature, top_p=a.top_p,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            metrics = {k: describe(text, ref) for k, ref in refs.items()}
            model_result["samples"][name] = {"prompt": prompt, "text": text,
                                              "chars": len(text), "metrics": metrics}
        result["models"][label] = model_result
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    pathlib.Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}")
    for label, model in result["models"].items():
        vals = [x["metrics"]["wake"]["novel_words"] for x in model["samples"].values()]
        reps = [x["metrics"]["wake"]["self_repeat"] for x in model["samples"].values()]
        print(f"{label}: novel_words_vs_wake={sum(vals)/len(vals):.3f} "
              f"self_repeat={sum(reps)/len(reps):.3f} n={len(vals)}")


if __name__ == "__main__":
    main()
