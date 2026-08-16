# H(source | summary) on our own summaries — first numbers

Task `summarizer-conditional-entropy` (owner: wake, priority:incident), 2026-08-16.
Harness: `wake/condent.py`, pair sets: `wake/pairs.py`.
Model: `Qwen3.5-0.8B-Base` (the same base the Wake LoRA sits on), bf16, one RTX 3060.

The question was whether the summaries we already write carry information about
their source, measured before anyone trains new ones. The measurement is the NLL
of the source's tokens with and without the summary in context; the delta in nats
per character is the answer. No reference summaries — the source is its own.

> **Update, same day.** Two later runs change the reading of everything above.
> A hand-written fold over the same board *does* carry content this metric sees —
> **+0.0294 nats/char** on tokens it never quotes, 8/8 windows — so the zero below
> is a fact about **what we write**, not a limit of the meter. And training the
> local 0.8B on 47 such folds roughly **doubles** its own copy-free signal
> (+0.0076 → +0.0153, 3.3 sem over 40 held-out windows), reaching about half the
> teacher. See [Constructed pairs](#constructed-pairs-the-ladder) and
> [Closing the gap](#closing-the-gap-distilling-the-fold-into-the-08b).
>
> The ladder section's own numbers were measured **before** a window-drift bug was
> found and are superseded by the post-freeze table in the distillation section;
> the drift and its fix are documented there.

## The headline

**They do not, beyond the words they literally copy.** Across all three pair sets,
once the summary is compared against a *foreign* summary of the same genre and the
same token budget, the gain on source tokens the summary does not quote is zero.

| pair set | n | H(source) alone | own − none | own − foreign | on tokens it quotes | on tokens it does not |
|---|---|---|---|---|---|---|
| handoff | 6 | 1.391 nats/char | +0.0438 | +0.0142 ± 0.0191 | **+0.1344** | **+0.0037 ± 0.0161** (2/6) |
| session | 13 | 1.240 | +0.0273 | +0.0170 ± 0.0106 | **+0.1354** | **+0.0001 ± 0.0045** (7/13) |
| ledger | 8 | 1.204 | +0.0051 | +0.0018 ± 0.0024 | **+0.0360** | **−0.0001 ± 0.0010** (4/8) |

Last column is the sign test: how many pairs came out positive. 7/13 and 4/8 are
exactly chance.

The cleanest single line in the whole run is the session set's novel subset:

```
delta vs no summary   own +0.0086   foreign +0.0085   shuffled +0.0009
```

Having *some* well-formed board prose in context is worth +0.0085 nats/char.
Having the *right* one is worth a further +0.0001. Word-salad made from your own
summary's tokens is worth +0.0009 — so the gain is not lexical priming either, it
is the genre. Control 1 was the control that could void the exercise, and on the
novel subset it voids it.

## What the instrument can see

The gate (`condent.py --test`) is not a smoke test; it fixes the scale of every
number above.

| gate | fixture | result |
|---|---|---|
| D structural | — | overlap/novel split disjoint, exhaustive, stable across conditions |
| A topical | 4 passages, 1-line topic summaries sharing almost no wording | signal/novel **+0.0777** |
| B uninformative | one constant summary shared by every pair | signal **+0.0000** |
| C verbatim | summaries that are literal fact lists | overlap **+2.4710** vs novel +0.0037 |

So the instrument resolves ~+0.078 nats/char of genuine content on the novel
subset, and returns exactly zero when there is nothing to find. Our summaries
return 0.000 ± 0.005 against a positive control of 0.078. The effect, if any, is
under 1/15 of what the same harness sees on ordinary topical prose, and the
measurement is powered to ~0.003 (sem over 13 pairs).

Gate B matters more than gate A. A metric that scores everything positive would
pass A and still be worthless — POSTMORTEM entry 9, a baseline that scored itself
zero by construction. B is the gate that proves this one can return zero, and it
returns it to four decimal places.

## The four controls

1. **Foreign summary** — another window's or day's summary, same genre, same
   budget. This is what collapses the result, and it is the whole reason the
   number means anything. `own − none` looks like a real effect (+0.027 on
   session, 2.2% of H); nearly all of it survives substituting someone else's
   summary.
2. **Length** — every condition puts exactly the same number of tokens in
   context. Budget is the minimum over the pair set, or explicit with `--budget`
   (which drops pairs that cannot fill it rather than padding them). `none` is
   the one condition that is *not* length-matched, by construction, which is why
   it is reported but never used as the answer.
3. **Lexical overlap** — source tokens are split by whether their id occurs in
   the own summary, with the partition fixed by the *own* summary in every
   condition so all conditions score the same subsets. The gap between the two
   columns is the finding: the entire measurable signal of our summaries sits on
   the words they copy. Gate C shows the split working at a ratio of ~670.
4. **Ledger as calibration** — `mesh-promises --balance`, a fold with *known*
   losses, replayed per board day via `MESH_CHAT_LOG` on a day slice. Each day
   yields two pairs sharing one ledger and one donor group: `#commit` (the axis
   the ledger keeps) and `#content` (what it discards by declaration).

## What the ledger control actually returned

The prediction was two-sided: recovery of the commitment axis should improve a
lot, recovery of content almost not at all. **Half of it reproduced.**

```
                signal.all       signal.overlap    signal.novel
#commit         +0.0018±0.0022   +0.0343±0.0411    -0.0000±0.0004
#content        +0.0018±0.0030   +0.0378±0.0327    -0.0002±0.0014
```

Content is flat, as predicted — so the meter is **not** manufacturing information
out of lexical overlap, which was the failure the control was built to catch. But
the commitment axis is flat too, and indistinguishable from content.

The reading is not that the meter is broken. It is a scope limit, and a real one:
the ledger's content is *relational* — this slug is open, that one is settled —
and a status fact does not shorten the description of the prose that carries it.
Everything the ledger contributes is its slugs, and slugs are overlap. Conditional
NLL of a source measures what lowers the code length of the source's **surface**.
A projection onto an axis that is true but not lexically recoverable is invisible
to it. That is worth knowing before the metric is used to grade a summarizer that
is supposed to preserve exactly such axes — the coverage/attribution metric named
in the design is not optional, and this is the measurement that shows why.

Caveat on this control's power: ledger source coverage was 7–91% (median 32%),
4 donor groups, 2–3 foreign donors each. It is powered to see ~0.002; a commit/
content difference smaller than that would not have shown. "Not detected", not
"proved identical".

## What the pair inventory turned out to be

Two corrections to the filed inventory, both measured:

- **The handoff pair decays in 900 seconds.** `mesh-handoff` line 47:
  `KEEP_MANUAL_SECS=900`. A manual handoff is protected for 15 minutes, after
  which the 5-minute `--snapshot` cron replaces it with a verbatim scrape of the
  window's terminal pane. This is deliberate — the crash-safety tier has to stay
  fresh — but it means the `.md` side of the pair has a 15-minute lifetime as
  full state. At read time: 22 of 23 files were `# source: auto-snapshot`, median
  age gap to their board line 8.1h, max 35.4h. `adint.md` was 4.0K/manual when
  the inventory was taken and 1146B/auto-snapshot forty minutes later.
- **The board `[handoff]` line is a truncated prefix, not a fold.** It is capped
  at ~215–295 characters with the `(full state → …)` pointer appended, and for
  `adint` it is character-for-character the opening of the `.md` body. This is
  why the handoff set puts +0.1344 on overlap and +0.0037 on novel: control 3 is
  reading a quotation, exactly what it was built to catch.

So the `session` set — a window's `[handoff]` line against that window's own board
lines since its previous handoff — is the pair that carries the result. Both sides
are chat.log, aligned in time by construction, nothing to decay. It agrees with
the handoff set and has twice the pairs.

`mesh-promises` does fold the board and the ledger control uses it. The gap that
remains is the one the task named: there is no **prose** fold over the board, and
the two prose summaries we do write score zero on content against their own genre.

## Constructed pairs: the ladder

The operator's call was to construct pairs and measure them. A single constructed
fold scoring zero would not have said whether the fold was poor or the meter was
at its limit here, so the folds were built as a **ladder over the same eight board
windows** (`wake/make_folds.py`, folds in `wake/constructed-folds.json`), each rung
answering a different question. Windows are deterministic cuts of chat.log,
1001–2262 tokens, scored at 100% coverage.

Only `abstractive` is authored. The other three are derived mechanically so they
cannot be hand-tuned toward a result.

At budget 219 tokens — the largest all three prose rungs can fill, so no rung is
padded and none is truncated more than another:

| rung | overlap | own − none | own − foreign | on quoted tokens | **on tokens it never quotes** | sign |
|---|---|---|---|---|---|---|
| **abstractive** (hand-written) | 21% | +0.0574 | +0.0491 | +0.1375 | **+0.0225 ± 0.0070** | **7/8** |
| model (local 0.8B instruct) | 32% | +0.0705 | +0.0561 | +0.2000 | −0.0012 ± 0.0036 | 4/8 |
| extractive (verbatim sentences) | 50% | +0.1329 | +0.0893 | +0.2672 | −0.0371 ± 0.0121 | 1/8 |

and with `entities` included, at the budget it can fill (126 tokens):

| rung | overlap | own − foreign | on quoted | **on unquoted** | sign |
|---|---|---|---|---|---|
| **abstractive** | 17% | +0.0160 | +0.0702 | **+0.0057 ± 0.0040** | 5/8 |
| entities (identifiers only) | 21% | +0.0304 | +0.2264 | −0.0045 ± 0.0022 | 1/8 |
| model | 23% | +0.0211 | +0.1140 | −0.0032 ± 0.0020 | 2/8 |
| extractive | 38% | +0.0320 | +0.1438 | −0.0197 ± 0.0076 | 1/8 |

### What this settles

**The meter is not at its limit on this corpus.** A real prose fold shows
+0.0225 nats/char on tokens it never quotes, 7 of 8 windows positive, 3.2 standard
errors from zero. Against the harness's own clean-English ceiling (+0.0777, gate A)
that is 29% — on operations-board prose in two languages, folded ~3x. The earlier
zero was a measurement of what we write, not of what can be measured.

**And it is dose-dependent.** The same folds over the same windows give +0.0057 at
budget 126 and +0.0225 at budget 219. More of the fold in context, more of the
source recovered. Noise does not do that.

**The local model produces fluent nullity, and now that has a number.** Its folds
read like summaries, quote *more* of the source than the hand-written ones (32%
overlap against 21%), score higher than them on `own − none` (+0.0705 vs +0.0574) —
and carry nothing: −0.0012 ± 0.0036, 4/8, indistinguishable from a stranger's fold
of a different window. This is exactly the failure the design named in advance —
*a fluent nothing reads well and barely moves recovery NLL* — measured on our own
material. It also means `own − none` is not merely confounded but actively
misleading: it ranks the model's folds **above** the ones that carry content.

**The teacher–student gap is now a measurement**, not a suspicion: +0.0225 against
−0.0012 on identical windows at an identical budget with identical controls.

**Copying scores negative.** Extractive puts +0.2672 on quoted tokens and
−0.0371 on the rest — worse than a foreign extract, 1/8 positive. Quoting source
sentences buys a large apparent gain that is entirely quotation, and on the residue
it leaves it does actual harm relative to generic board prose. Our real board
`[handoff]` lines are truncated prefixes, i.e. this rung.

**The information is not just the nouns.** `entities` and `abstractive` are matched
at 21% and 17% overlap, and separate cleanly: −0.0045 against +0.0057 at the same
budget. Identifiers alone reproduce the ledger control's whole behaviour — a large
overlap gain (+0.2264, the biggest of any rung) and nothing beyond it. That is the
ledger finding independently confirmed on constructed material: slugs are copying.

### Reading the cross-rung comparison honestly

The overlap fraction differs by rung (21% / 32% / 50%), so "tokens it never quotes"
is a *different subset* in each row, and the residue an extractive summary leaves is
selected to be the part it could not cover. The clean comparisons are therefore the
ones inside a rung (own vs foreign, same subset, same budget) and the two rungs that
happen to be overlap-matched (abstractive 17% vs entities 21%). Every claim above is
one of those two kinds. The rung ordering by `signal.novel` is consistent at both
budgets, which is what it would not be if the subset selection were driving it.

Other limits: 8 windows, so 7 foreign donors per pair. The folds were written by a
large model that had read the window — which is the point, since the question was
whether a *good* fold is visible at all, and it establishes the teacher ceiling the
0.8B student is being measured against. Whether the signal keeps rising past budget
219 was not tested; doing so drops the shorter windows and changes the pair set.

## Closing the gap: distilling the fold into the 0.8B

The operator's next call was to train the local model on the hand-written folds.
55 more board windows were folded by hand (`wake/teacher-folds.json`, ~3x), and a
LoRA (rank 16) was trained on Qwen3.5-0.8B-Instruct with the loss on the fold
tokens only and the prompt imported verbatim from `make_folds.py`, so training is
not confounded with a prompt change. Every epoch was kept: val loss is the wrong
selector here — the whole finding is that fluent-and-empty scores well on
likelihood — so the adapters are judged by `condent.py` afterwards.

### First, the run that had to be thrown away

The evaluation windows were being cut live from chat.log on every invocation.
chat.log is a fixed 3000-line ring: as the mesh posts, old lines evict and the
tail slides. Measured over about an hour, **all 8 windows had moved** — different
start times, different lengths, 0/8 identical. Nothing errored. A fold written at
05:50 was scored against a different window at 07:00, which reads as a foreign
summary; worse, a rung generated moments ago was matched while an older rung was
not, so the comparison silently favoured whatever was generated last — which was
always the student.

It surfaced because the teacher's +0.0225 did not reproduce on a re-run (+0.0035),
and a 7-token budget change cannot do that. Windows are now frozen to
`wake/test-windows.json`; 8 of 55 training rows had likewise been paired with a
drifted source and were repaired; both adapters were retrained. Every number
below is from after the freeze. This is the same failure the harness's own
controls are built around — a comparison that looks like content and is really
an artifact of how the two sides were produced.

### On the 8 test windows, at an equal 183-token budget

| rung | SIGNAL/novel | sem | sign |
|---|---|---|---|
| **abstractive** (teacher, hand-written) | **+0.0294** | 0.0045 | 8/8 |
| **ft47-e3** (student: 47 folds, 3 epochs) | **+0.0147** | 0.0032 | 8/8 |
| model (untrained baseline) | +0.0060 | 0.0031 | 6/8 |
| ft12-e2 (12 folds) | +0.0051 | 0.0067 | 3/8 |
| ft12-e1 (12 folds) | +0.0007 | 0.0071 | 4/8 |
| extractive (copying) | −0.0408 | 0.0133 | 1/8 |

Paired over the same windows: teacher − baseline **+0.0234 ± 0.0053** (4.4 sem,
8/8); student − baseline **+0.0088 ± 0.0051** (1.7 sem); teacher − student
**+0.0147 ± 0.0047** (3.1 sem, 7/8).

### Then on 40 held-out windows, because 8 was underpowered

The student-versus-baseline contrast needs no teacher folds, so it can be run on
far more windows. 40 fresh windows were cut from the union corpus, disjoint from
every training window and from the 8 test windows, and frozen
(`wake/eval-windows.json`). At a 157-token budget:

| rung | SIGNAL/novel | sem | sign |
|---|---|---|---|
| ft47-e3 (student) | **+0.0153** | 0.0022 | 36/40 |
| model (untrained) | +0.0076 | 0.0015 | 32/40 |

**Paired: +0.0077 ± 0.0023, 3.3 sem, 27/40 windows.** The 8-window estimate
(+0.0088) was right in magnitude and merely underpowered; at n=40 it resolves.

### What this settles, and what it does not

**Training worked, and the effect is real.** 47 hand-written folds roughly double
the copy-free content signal of the local model — +0.0076 to +0.0153 nats/char —
established at 3.3 sem on 40 held-out windows. This is the first rung of the
ladder that moved because of training rather than because of who wrote the text.

**The gap is narrowed, not closed.** On the shared 8-window measurement the
teacher sits at +0.0294 against the student's +0.0147: training recovered about a
third of the teacher−baseline gap and the teacher is still ahead by 3.1 sem. The
honest headline is *half the teacher*, not *matched*.

**Data, not epochs, is the axis that moved it.** 12 folds produced nothing
(+0.0007 and +0.0051, indistinguishable from baseline; paired −0.0053 ± 0.0083).
And the epoch curve is flat — measured at a 111-token budget on the test windows,
epochs 1/2/3 give +0.0081, +0.0039, +0.0048 over baseline, all between 0.8 and
1.8 sem with no trend. So the gain came from having enough examples, not from
training longer on few, and there is no evidence for picking a particular epoch.

**Val loss would have picked wrong-ish, and could not have seen this at all.**
Training loss fell monotonically (3.08 → 2.45 → 2.31 → 2.28) while the copy-free
content signal did not track it. The selection had to be made by this metric.

Limits: one seed, one rank, one base model; the teacher rung exists only on the 8
test windows, so *half the teacher* rests on 8 paired windows rather than 40; and
the student folds are shorter than the teacher's, which is why the budgets differ
between the two tables (157 and 183 are the largest each set could fill without
dropping pairs).

## Tripling the teacher data — a negative result, and a selection error of my own

The operator asked for another hundred folded windows and a retrain. 97 more were
folded by hand (152 pairs total, 144 train / 8 val) and the LoRA retrained.
**It did not close the gap. It widened it.**

On the same 40 held-out windows, same 157-token budget, same untrained baseline:

| student | SIGNAL/novel | paired vs baseline | vs the 47-fold student |
|---|---|---|---|
| **47 folds**, epoch 3 | **+0.0153 ± 0.0022** (36/40) | **+0.0077 ± 0.0023** (3.3 sem) | — |
| 144 folds, epoch 3 | +0.0124 ± 0.0016 (37/38) | +0.0050 ± 0.0022 (2.3 sem) | −0.0029 ± 0.0022 (−1.3 sem) |
| 144 folds, epoch 2 | +0.0101 ± 0.0019 (29/40) | +0.0025 ± 0.0022 (1.1 sem) | −0.0052 ± 0.0019 (−2.7 sem) |
| untrained baseline | +0.0076 ± 0.0015 (32/40) | — | — |

### The selection error, which is mine

The 97 new windows were cut by removing every board line already claimed by the
training, eval or test sets and then cutting from the contiguous runs of what was
left. That guarantees zero overlap — and it silently guarantees something else,
because the leftover runs are the *gaps between* already-claimed windows:

| window set | lines (median) | distinct authors (median) |
|---|---|---|
| train batch 1 (55) | 14 | 8.5 |
| **train batch 2 (97)** | **6** | **5.0** |
| eval (40) | 13 | 8.0 |
| test (8) | 14 | 8.5 |

Two thirds of the training set is now drawn from windows half as wide and half as
many-voiced as anything it is scored on. My folds for them are correspondingly
terser (1310 vs 1475 median chars, 3.19x vs 2.95x compression). So the honest
first reading of the negative result is not "hand-written folds stop helping past
47" but "I fed it 97 examples of a different task". Re-cutting the extra windows
to match the evaluation distribution — cutting them *first* and letting the eval
set take what is left — is the experiment that would actually answer the
operator's question, and it is the obvious next run.

### What survives the confound

The distribution artifact explains why 144 lost to 47. It does **not** explain
this, and this is the sharper finding:

**Val loss ranked the worse model higher, on both axes at once.**

- Across dataset size: val loss is markedly *better* at 144 than at 47 (2.1147 vs
  2.2838) while the content metric is *worse*.
- Across epochs within 144: val loss picks epoch 2 (2.1147 vs 2.1312) while the
  content metric prefers epoch 3 by +0.0023.

And the val rows are all batch-1 windows — the *wide* distribution — so this is
not the artifact talking. The model got measurably better at reproducing my folds
token by token while producing folds that carry measurably less recoverable
content. That is the same failure as `own − none` ranking empty folds above full
ones, promoted from the metric level to the model-selection level: on this task,
the likelihood a summarizer assigns to a reference summary is not merely a weak
proxy for whether its own summaries carry information — it points the wrong way.
Anyone selecting a checkpoint or a dataset size by val loss here selects the
worse model twice over.

Caveats: one seed; the 47-fold adapter is from the earlier training run, so the
two dataset sizes are not nested draws; and the 144-epoch-3 rung scored 38 of 40
windows because two of its folds fell below the shared budget, which mildly
favours it.

## Reproducing

```bash
.venv-ai/bin/python wake/condent.py --test                     # 4 gates, ~1 min
.venv-ai/bin/python wake/condent.py --pairs session --n-chunks 4 --n-foreign 6
.venv-ai/bin/python wake/condent.py --pairs ledger --budget 90 --n-chunks 4 --by axis
.venv-ai/bin/python wake/condent.py --pairs handoff

.venv-ai/bin/python wake/make_folds.py                         # rebuild the ladder
for v in abstractive extractive model; do
  .venv-ai/bin/python wake/condent.py --pairs $v --budget 219 --max-src-tok 2400
done
.venv-ai/bin/python wake/condent.py --pairs entities --budget 126 --max-src-tok 2400

# distillation: train, generate the student rung, score it
.venv-ai/bin/python wake/train_fold.py --n 47 --epochs 3 --out fold-lora-47
.venv-ai/bin/python wake/make_folds.py --rung --adapter fold-lora-47/ep3 --variant ft47-e3
.venv-ai/bin/python wake/condent.py --pairs ft47-e3 --budget 183 --max-src-tok 2400

# the 40-window held-out set swaps in by environment, no code change
export CONDENT_WINDOWS=wake/eval-windows.json CONDENT_FOLDS=wake/eval-folds.json
.venv-ai/bin/python wake/condent.py --pairs ft47-e3 --budget 157 --n-foreign 6
```

Windows are frozen on disk (`test-windows.json`, `eval-windows.json`). Deleting
either re-cuts from the live board and makes every fold beside it stale in the
same motion — see the drift note above.

`--dry-run` inventories a pair set without loading the model. The harness is
pair-agnostic: it takes `(source_text, summary_text)` and knows nothing else, so
the next pair set is a loader in `pairs.py`, not a change here.

## Limits worth stating

- Source coverage: handoff 100%, session 27–100% (median 71%), ledger 7–91%
  (median 32%). Long sources are scored as up to 4 evenly spaced chunks, each with
  the full context, pooled — better than reading only the head, still not all of it.
- The instrument's sensitivity is the 0.8B base model's ability to exploit a
  context. Gate A demonstrates that ability on clean topical English. Board prose
  is harder, and a larger model might resolve something this one cannot.
- Budgets here are 76–90 tokens. A longer budget was not tested on the prose sets;
  the ledger run at budget 56 vs 90 moved nothing on the novel subset.
