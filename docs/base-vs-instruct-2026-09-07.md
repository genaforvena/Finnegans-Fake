# Base versus Instruct: controlled style read

This closes the last research question in the post-mortem: whether the Base checkpoint bends
further into Wakese than the instruction-tuned checkpoint.

## Design

`wake/compare_base_instruct.py` drove the two raw local checkpoints with the same five plain-text
prompts, seed 0, temperature 0.9, top-p 0.95, and a 600-character target. It intentionally uses
plain text rather than each model's chat template: the input string is then identical, and the
comparison asks how the two pretrained distributions continue the same prompt. This is a sample
property comparison, not a loss comparison and not a comparison of two Wake LoRA adapters.

The complete artifact is [`wake/base-instruct-results.json`](../wake/base-instruct-results.json).
It records the git commit/dirty state, corpus hashes, checkpoint config/tokenizer hashes, every
prompt, every continuation, and every metric. The run used the local Qwen3.5-0.8B-Base and
Qwen3.5-0.8B checkpoints; their files remain outside this repository.

## Result

| raw checkpoint | mean novel word types vs Wake | mean self-repeat | mean type/token |
|---|---:|---:|---:|
| Base | 0.108 | 0.164 | 0.526 |
| Instruct | 0.102 | 0.391 | 0.380 |

The Base run has lower self-repetition and higher lexical diversity under this probe. The Instruct
run repeats more and reuses a smaller vocabulary, while its Wake-novel rate is effectively the same
at this sample size. The bounded conclusion is therefore: **under identical plain-text prompts,
the Base checkpoint bends more toward varied Wake-like continuation; the Instruct checkpoint is
more repetitive, but this five-prompt probe does not establish a difference in novelty rate.**

This does not prove that a Base-trained Wake LoRA beats an Instruct-trained Wake LoRA. The repo has
only the Base LoRA (`wake/wake-lora-base`), and a matched Instruct LoRA training run was attempted
on 2026-09-07 but stopped at its first validation pass because the shared GPU was already occupied
and returned CUDA OOM. That absence is recorded rather than silently treated as a quiet run.

Reproduce:

```bash
/home/mesh-home/.venv-ai/bin/python wake/compare_base_instruct.py --chars 600
```

