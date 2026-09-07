# Publishable Finnegans Fake Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the measured Finnegans Fake prototype into a reproducible, useful, publishable result package whose claims, models, release metadata, and operator-facing reflex all agree.

**Architecture:** Keep the existing separation between source-text experiments, the English control, the pretrained LoRA, and the local-only board-fold adapter. Add one provenance/evaluation layer that derives public prose from run artifacts, and one release gate that loads and samples every artifact before it can be listed as publishable. The research conclusion remains property-based (copying, novelty, tokenisation, overfit, and control differences), not an aesthetic ranking of samples.

**Tech Stack:** Python 3, PyTorch, Transformers, PEFT, safetensors, Hugging Face repo layout, JSON/Markdown, existing shell harnesses.

## Global Constraints

- Preserve the operator’s instruction that project repositories are CC0, while never publishing `finnegans-fake-folds-lora` because it is trained on the internal board log.
- Never claim a score without its producing run, corpus/tokenizer, seed/replicate count, and artifact path.
- A pack/release check must exercise loading and generation, not only file existence or tensor parsing.
- Base-vs-Instruct comparison is a sample/property comparison, not a cross-tokenizer loss ranking.
- Do not overwrite unrelated dirty work; audit the existing `wake/pack_release.py` modification and retain its intent only when the generated cards and uploader enforce the same policy.

## Evidence captured from the operator brief

- Telegram record: `/home/mesh-home/.mesh/tg-sent.log`, 2026-08-15 17:44–19:46 UTC: resurrect the repo, run both bets, retain the failed-result post-mortem, and measure properties instead of calling samples “better” or “worse”.
- Telegram record: 2026-08-16 04:41 UTC: nearly finished; remaining research question is whether a Base model bends further into Wakese than an instruction-tuned model, judged by samples/shared-scale properties rather than raw loss.
- Repository evidence: `POSTMORTEM.md` Open section; current release manifest; `wake/pack_release.py`; `wake/push_release.py`; `wake/reflex.py`.

### Task 1: Freeze the requirements and current evidence

**Files:**
- Create: `docs/operator-brief-2026-09-07.md`
- Modify: `README.md` only if a stale requirement is contradicted by the brief
- Test: `python -m json.tool` on all JSON evidence files referenced by the brief

- [x] Extract the operator requirements above into a concise, citable repository note, including the publish/private boundary and the exact Telegram source locations.
- [x] Record the current dirty-tree state and the current release-manifest state in the note; do not silently treat the existing packer edit as verified.
- [x] Add a checklist mapping each requirement to an artifact and verification command.
- [x] Run `git diff --check` and JSON syntax checks.

### Task 2: Make release policy mechanically consistent

**Files:**
- Modify: `wake/pack_release.py`
- Modify: `wake/push_release.py`
- Modify: `README.md`
- Test: `wake/pack_release.py --only ...`, `wake/push_release.py --owner <owner> --dry-run`

- [x] Define one explicit policy table: code and source-independent control are CC0/public; Wake-derived artifacts follow the operator’s CC0/public direction; the board-fold adapter remains `local` and is refused unconditionally.
- [x] Ensure generated cards, `MANIFEST.json`, and uploader visibility are derived from that same table.
- [x] Remove contradictory prose such as “private” or “force-public” where it no longer describes the selected policy; keep a factual copyright-risk note without making the manifest lie.
- [x] Make the uploader refuse a local artifact before any network operation and verify manifest hashes before upload.
- [x] Fresh dry-run verified the local refusal and four public upload entries; no upload was performed.

### Task 3: Add provenance to every score-bearing result

**Files:**
- Modify: `wake/measure.py`, `wake/compare.py`, `wake/condent.py`, `wake/paired.py` as applicable
- Create: `wake/provenance.py`
- Create: `wake/evaluate_release.py`
- Modify: `README.md`, `POSTMORTEM.md`
- Test: fixture-driven provenance and stale-run tests

- [x] Attest each historical result with score-file hash, producing commit, producer script hash, model/run path, rung seed/training metadata, and budget in `wake/score-provenance.json`.
- [x] The attestation command exits nonzero for an untraceable producer rather than guessing; current 10/10 scored sets are attested.
- [x] Existing `paired.py --test` recomputes the replicated 47-vs-144 gate and refuses under-replicated comparisons; documentation points to rung/run IDs.
- [ ] Runtime stamping for future score producers remains a follow-up; historical artifacts are explicitly labeled by attestation rather than retroactively called runtime-stamped.

### Task 4: Close the Base-vs-Instruct research gap

**Files:**
- Modify: `wake/train_lora.py` only for reproducibility/provenance if needed
- Create: `wake/compare_lora_variants.py`
- Create: `wake/lora-variant-results.json`
- Create: `docs/base-vs-instruct-2026-09-07.md`
- Modify: `POSTMORTEM.md`, `README.md`
- Test: deterministic smoke run on existing adapters or a small fixture

- [x] Identify the available Base and Instruct checkpoints and compare them with identical plain-text prompts, seed, temperature/top-p, and character target.
- [x] Measure copy proxies, novel-word rate, type/token ratio, self-repeat, and save every sample in `wake/base-instruct-results.json`.
- [x] Publish a bounded conclusion in `docs/base-vs-instruct-2026-09-07.md`; no raw-loss ranking is used.
- [x] Verify the result file contains git/corpus/checkpoint hashes and every prose number is read back from it.
- [ ] A matched Instruct Wake-LoRA remains unresolved: the attempted run stopped at first validation with CUDA OOM from existing GPU occupants.

### Task 5: Rebuild and verify the release artifacts

**Files:**
- Modify: `wake/pack_release.py`
- Create: `wake/release_smoke.py`
- Modify: `release/MANIFEST.json` and generated release cards
- Modify: `README.md`
- Test: full pack, load/sample each public artifact, local-fold refusal, hash verification

- [x] Repack every source model from current trainlogs; do not hand-edit release cards or manifest values.
- [x] Load and sample each public causal model and adapter against its declared base; record non-empty output and parameter/tensor counts in `release/MANIFEST.json`.
- [x] Verify the public artifacts’ licenses/cards match the selected policy and the fold adapter has no upload path.
- [x] Run the release smoke path in `/home/mesh-home/.venv-ai`; the packer loaded/sampled all five artifacts.
- [x] Run `wake/push_release.py --dry-run` and inspect the exact upload plan; no external upload was performed.

### Task 6: Finish the user-facing reflex and documentation

**Files:**
- Modify: `wake/reflex.py`
- Modify: `README.md`, `samples/README.md`
- Create: `docs/release-evidence-2026-09-07.md`
- Test: `/wake`-equivalent local invocation with each eligible model and absent-model failure

- [x] Reflex smoke test enumerates nine present model/product IDs and checks the shared tokenizer.
- [x] Actual `wake-char257` invocation returned a named generated continuation; missing-model behavior remains covered by the existing `--test` inventory.
- [x] README and POSTMORTEM now link the Base-vs-Instruct artifact and state the exact unresolved matched-LoRA limitation.
- [ ] Write the final evidence index and handoff after the remaining source review/commit decision.

## Completion gate

- [x] `git diff --check` passes.
- [x] All targeted tests and release smoke checks pass with fresh output.
- [x] Every quoted historical score has provenance in the attestation index and is referenced by rung/run ID.
- [x] `POSTMORTEM.md` states the bounded Base-vs-Instruct result and does not claim the unrun matched Instruct LoRA.
- [x] `release/MANIFEST.json` agrees with generated cards and uploader policy.
- [x] The local-only board adapter is still refused by the uploader.
- [ ] Final handoff must name the unresolved matched-LoRA run and the operator-owned Hugging Face push as next actions.
