# Operator brief: what Finnegans Fake must deliver

This is the durable requirements record for the 2026-09-07 completion pass. It is derived from
the operator Telegram record and checked against the repository rather than treated as memory.

## Source record

- `/home/mesh-home/.mesh/tg-sent.log:3411-3422` (2026-08-15 17:44–18:09 UTC): the operator
  accepted the small Finnegans Wake experiment, asked for both a from-scratch wordplay bet and a
  conversational pretrained-model bet, and set the repository’s CC0 direction.
- `/home/mesh-home/.mesh/tg-sent.log:3430-3447` (2026-08-15 18:25–19:46 UTC): the first result
  was a failed objective, then a post-mortem and measured sample-property results replaced
  aesthetic claims such as “worse” or “nonsense”.
- `/home/mesh-home/.mesh/tg-sent.log:3564` and `/home/mesh-home/.mesh/tg-sent.log:3644`
  (2026-08-16): the wake reflex was made interactive and the lane was asked to continue from
  measured artifacts.
- `/home/mesh-home/.mesh/tg-sent.log:3774-3776` (2026-08-18): the wake lane was restored after
  disappearance and must remain a real, tended project lane.

The Telegram log is a transport record and truncates long messages. The repository artifacts are
the authoritative technical evidence: `README.md`, `POSTMORTEM.md`, `wake/*.py`, `wake/recs-*.json`,
`wake/*/trainlog.json`, and `release/MANIFEST.json`.

## Required outcome

Finnegans Fake is publishable/useful when a reader can:

1. understand what was measured and what failed;
2. reproduce the reported comparisons from checked-in code and declared input artifacts;
3. load and sample every artifact advertised as shippable;
4. distinguish Wake-derived, ordinary-English control, pretrained conversational, and
   internal-board-fold artifacts;
5. see the model/run identity beside every score; and
6. use the reflex without a hidden model fallback or an implied claim that a sample is a result.

## Policy boundary

The operator’s project-level direction is CC0 for repositories. The internal-board fold adapter is
the explicit exception to publication: it remains local-only because it can reproduce operational
mesh text. The code and release tooling must enforce this boundary mechanically; a README sentence
or a caller remembering a flag is not enough.

## Current gaps at the start of this pass

- `POSTMORTEM.md` still lists the Base-vs-Instruct comparison as open.
- The wake score files are useful artifacts but do not uniformly carry producer git/script/corpus
  provenance, so a bare score is not yet safe to quote.
- The existing dirty edit to `wake/pack_release.py` selects public/CC0 cards for Wake-derived
  artifacts, but that policy must be checked against `wake/push_release.py`, generated cards, and
  the local-only fold refusal.
- A release manifest exists, but the completion claim needs a fresh pack/load/sample run after the
  current source tree is settled.

## Completion evidence checklist

| Requirement | Artifact that proves it | Fresh check |
|---|---|---|
| failed objective remains visible | `POSTMORTEM.md` | inspect cited train logs and samples |
| measured properties, not taste ranking | `wake/measure.py`, `samples/`, result docs | rerun the narrow measurement |
| score identity | result JSON + provenance block | provenance validation |
| Base-vs-Instruct answer | `docs/base-vs-instruct-*.md`, JSON result | equal-character evaluation |
| shippable files load | `release/*`, `release/MANIFEST.json` | pack/release smoke gate |
| internal adapter cannot leave | `share=local` + uploader refusal | dry-run refusal test |
| reflex is useful | `wake/reflex.py` output | model-eligible invocation and absent-model failure |

