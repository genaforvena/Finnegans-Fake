# The run-to-run spread the ladder was read through

Task `seed-spread` (owner: wake), 2026-08-18. Driver: `wake/run-seed-spread.sh`.
Four independent trainings of the **same 144 rows**, scored on the **same 40 frozen
eval windows** at the same 157-token budget with the same 6 foreign donors as
`db416ad`. Model: `Qwen3.5-0.8B-Base` scored, `Qwen3.5-0.8B-Instruct` + LoRA r16
trained, one RTX 3060.

The question: every comparison this lane has published is a difference of ~0.002
nats/char between rungs trained **once each**. How much of that is the training
draw?

## First, the reason it could not be asked before

`train_fold.py` seeded `random.Random(a.seed)` — the row draw — and **never seeded
torch**. Every other script in `wake/` seeds it, including `train_scratch.py` next
door. So the LoRA init, the dropout masks and the DataLoader's shuffle came from an
unseeded global generator while `trainlog.json` recorded `"seed": 0` for all four
rungs on the ladder.

Measured, not inferred. Two runs, same `--seed 0`, same `--n 12`, identical
`[data]` line and identical step-0 val 3.0860 — `lora_B` is zero-init, so step-0
**is** the base model and cannot discriminate — diverged by epoch 1:

| | epoch-1 train | epoch-1 val |
|---|---|---|
| unseeded, run A | 3.1562 | 2.7637 |
| unseeded, run B | 3.1642 | 2.7705 |
| **seeded**, run C | 3.1621 | 2.7772 |
| **seeded**, run D | 3.1622 | 2.7784 |
| seeded, `--seed 7` | 2.9179 | 2.7668 |

Seeding removes the draw variance (80x on train loss) and leaves CUDA reduction
nondeterminism, which is a floor, not a knob. A different seed moves the run far
further than either residual. Fixed in `faaea27`.

`n=144` is what makes the experiment clean: `pool[:144]` takes **all** 144 training
rows whatever the shuffle, so the data is identical across seeds by construction
and the only thing varying is the training draw.

## The spread

Aggregate SIGNAL/novel on the 38 windows common to all six scored sets:

| rung | training | SIGNAL/novel | positive |
|---|---|---|---|
| untrained base | — | +0.0078 ± 0.0015 | 31/38 |
| 47 folds, ep3 | one draw | +0.0155 ± 0.0023 | 34/38 |
| 144 folds, ep3 | seed 0 (unseeded code) | +0.0132 ± 0.0015 | 37/38 |
| 144 folds, ep3 | seed 1 | +0.0136 ± 0.0016 | 36/38 |
| 144 folds, ep3 | seed 2 | +0.0135 ± 0.0015 | 37/38 |
| 144 folds, ep3 | seed 3 | **+0.0107** ± 0.0013 | 33/38 |

**sd 0.00140 across the four, range 0.00295.** Note the shape of it: three of the
four cluster inside 0.0004 and the fourth sits 0.0028 below them. Read after three
runs this looked like a spread far smaller than the effect under test — that
reading was published here in an interim board line and the fourth run falsified
it. Three agreeing runs are not a spread measurement; they are three draws that
happened to agree.

## What that does to the published verdict

The gate `db416ad` turned on was: 144 must beat 47, paired, same windows, same
budget. Re-run once per training:

| training | (144 rung) − (47 rung), paired | |
|---|---|---|
| seed 0 | −0.0023 ± 0.0022 | −1.0 sem, 19/38 positive |
| seed 1 | −0.0019 ± 0.0023 | −0.8 sem, 18/38 |
| seed 2 | −0.0021 ± 0.0021 | −1.0 sem, 17/38 |
| seed 3 | **−0.0049** ± 0.0022 | **−2.2 sem**, 15/38 |

**The sign survives replication — all four are negative.** The strength does not:
the verdict's own spread is sd 0.00140 and its range 0.00295 **exceeds** the
±0.0022 error bar that was published beside it. One of the four trainings would
have licensed "144 significantly loses to 47" at 2.2 sem; the other three would
have called it inside the bar. Same code, same data, same eval set, same budget.

The same swing hits the claim that training works at all:

| training | (144 rung) − (untrained base), paired |
|---|---|
| 47 folds | +0.0077 ± 0.0024 (+3.2 sem) |
| seed 0 | +0.0054 ± 0.0017 (+3.2 sem) |
| seed 1 | +0.0058 ± 0.0021 (+2.7 sem) |
| seed 2 | +0.0056 ± 0.0016 (+3.4 sem) |
| seed 3 | +0.0028 ± 0.0018 (+1.6 sem) |

Every rung beats the untrained base, so *that* finding holds; but seed 3 alone
would have been written up as "barely moved" while its three siblings read 2.7–3.4
sem.

## A single window's reading is dominated by which training you ran

Per-window sd across the four trainings: **median 0.0059, max 0.0139.** For scale,
the per-window 144-vs-47 differences have sd 0.0136. So the noise the aggregate
averages away is, window by window, about as large as the difference being
measured. Anything read off one window under one training is not a measurement.

## Val loss ranks the loser again — now between replicates

The four differ only in the training draw, so this is the cleanest instance yet:

| training | val ep1 / ep2 / ep3 | content, ep3 |
|---|---|---|
| seed 0 | 2.2557 / 2.0917 / 2.0853 | +0.0132 |
| seed 1 | 2.2345 / 2.0900 / 2.0719 | +0.0136 |
| seed 2 | 2.2298 / 2.0818 / **2.1011** | +0.0135 |
| seed 3 | 2.2403 / 2.0946 / 2.0864 | **+0.0107** |

By val loss at ep3 the order is s1 < seed0 < s3 < s2; by the content metric it is
s1 > s2 > seed0 > s3. Val puts **s2 last and s3 third**, while the content metric
puts s2 second and s3 last — and that pair carries the largest content gap of the
four (0.0135 vs 0.0107). Val's own best epoch also moves with the draw: three runs
are monotone to ep3, s2 turns back up at ep3. Selecting an epoch or a run by val
loss remains the wrong instrument, and its wrongness is not a property of a
dataset size — it shows up between runs that differ in nothing else.

## What this settles, and what it does not

- **Settled: the training-noise term is missing from every published error bar
  here, and it is not small.** A single-rung aggregate carries sd ≈ 0.0014 of it; a
  difference between two independently trained rungs carries ≈ 0.0020, which is the
  size of the effects being compared. Every future comparison at this scale needs
  replication, and its bar must include this term.
- **Settled: the direction of the 144-vs-47 result.** Four independent trainings,
  four negative paired differences. Doubling the hand-written teacher data does not
  improve this rung, and the earlier negative result is not an artifact of one
  unlucky draw.
- **NOT settled: how far 144 sits below 47.** −0.0019 and −0.0049 are both in the
  measured range, so "how much worse" has no answer yet at n=4.
- **NOT settled: the 47 rung's own spread.** It is one draw, and the gate is a
  difference of two rungs, so half of the gate's training noise is still unmeasured.
  Replicating it needs the row draw held fixed while the training draw varies, and
  `--seed` alone cannot do it: at `--n 47` the pool is 144 rows, so seeds 0 and 7
  share only **15 of their 47 rows** — a second training would also be a different
  training set. (At n=144 the two decouple by construction: the drawn set is
  identical across seeds, which is what made this experiment clean.) That is what
  `--data-seed` is for (this commit).
- Limits unchanged from `db416ad`: one rank, one base model, one epoch policy, and
  the teacher rung exists only on the 8 test windows.

## Reproducing

```bash
SEEDS="1 2 3" ./wake/run-seed-spread.sh        # ~25 min per seed on a 3060
# per-rung and paired, on the windows both sides scored. NOTE: this 1-vs-1 form is
# now REFUSED by paired.py's replication gate — see the section at the end; it is
# the exact comparison that was withdrawn. Pass all replicates with --a/--b.
.venv-ai/bin/python wake/paired.py wake/recs-ft47-e3.json wake/recs-ft144s3-e3.json
```

The driver resumes: a seed whose `ep3` adapter already exists is not retrained.
Note that python's stdout is block-buffered through the driver's `tee`, so a
stage's lines land in a burst when it ends — a frozen log mtime is not a stall.

---

# Replicating the OTHER side: the negative result does not survive it

Same day, second batch. The section above replicated the 144 rung four times and
concluded that the sign of the 144-vs-47 gap "survives replication, four for four".
That conclusion was drawn with **one side of the comparison unreplicated** — which
is the same error the section is about, committed one level up. The 47 rung was one
draw, and a difference of two rungs carries the training noise of both.

`PREFIX=t N=47` reproduces its training set exactly rather than approximately:
batch 1 is the 55 `t` windows, the frozen split holds 8 of them as val and 47 as
train (checked — all 8 val rows are `t` rows, and `t`-train is exactly 47), so the
drawn set is the whole pool whatever the shuffle. Three trainings, same rows, same
eval, same budget.

## On the 34 windows every scored set shares

| rung | | |
|---|---|---|
| untrained base | +0.0075 ± 0.0015 | 28/34 |
| **ft47-e3 (published)** | **+0.0165** ± 0.0024 | 31/34 |
| 47 rows, seed 1 | +0.0124 ± 0.0019 | 31/34 |
| 47 rows, seed 2 | +0.0113 ± 0.0017 | 30/34 |
| 47 rows, seed 3 | +0.0150 ± 0.0024 | 30/34 |
| 144 rows, seed 0 | +0.0130 ± 0.0017 | 33/34 |
| 144 rows, seed 1 | +0.0142 ± 0.0018 | 32/34 |
| 144 rows, seed 2 | +0.0133 ± 0.0016 | 33/34 |
| 144 rows, seed 3 | +0.0104 ± 0.0014 | 29/34 |

**The published 47 rung is a high draw.** Its three replicates mean +0.0129 with
sd 0.00178; the published one sits at +0.0165, **+0.0036 above them, 1.9 replicate
sd.** Nothing was wrong with it — it is one sample from a distribution whose width
nobody had measured.

## The gate, with both sides replicated

| (144 rung) − (47 rung) | result |
|---|---|
| vs the **published** 47 rung, 4 pairings | −0.0035, −0.0023, −0.0031, −0.0061 — mean **−0.0038**, up to −2.7 sem |
| vs the 47 **replicates**, 12 pairings | mean **−0.0002**, sd 0.0022, range [−0.0046, +0.0029], **negative in 6 of 12** |

Family means: 144 gives +0.0127, 47 gives +0.0129, difference **−0.0002 ± 0.0014**.

**So "144 folds lose to 47" was four draws of one rung measured against a single
high draw of the other.** With both sides replicated the difference is zero and its
sign is a coin flip — 6 of 12 pairings negative is exactly chance. The claim in
`4a07481` and `db416ad`, and the "the sign survives replication" line above, are all
withdrawn.

What survives, and is now on seven independent trainings: **the folding behaviour
distils.** Every rung beats the untrained base — +0.0029 to +0.0090, 1.5 to 3.5 sem
— and the two corpus sizes are indistinguishable from each other.

## The error bar this lane should have been using

Training noise is sd 0.00140 for a 144 rung and sd 0.00178 for a 47 rung, so a
difference between two singly-trained rungs carries **±0.00226 from the training
draw alone**. Folded with the window-sampling sem of about 0.0022, the honest bar on
a one-run-vs-one-run comparison is **±0.0032**, not ±0.0022. `db416ad`'s −0.0018 is
0.6 sem against that — it was never a result. A comparison at this scale needs both
sides replicated; a single pair of runs cannot resolve 0.002 no matter how many
windows it is averaged over.

## What to do about it

The effect sizes this lane cares about are ~0.002–0.008 and the per-comparison floor
is ~0.0032. Two ways out, and they compose: **replicate both sides** (3 runs a side
divides the training term by √3, to ±0.0013) and **widen the eval set** (the window
term is sem over 40 windows). Neither is a code change; both are runs. What must not
happen again is a rung comparison reported from one training per side.

## The rule above is now a gate, not this paragraph

Written down, a rule like that is honoured until someone is in a hurry — and the
comparison it forbids is exactly the cheap one (two files already on disk). So
`paired.py` **refuses to print a between-rung delta unless both sides carry ≥3
distinct trainings**, and prints why instead. The per-rung aggregates still print:
a single-rung reading is a measurement, it is their *difference* that is not.

Three details it would be vacuous without:

- It counts **distinct trainings, not files**. Two epochs of one run are one draw,
  and `recs-ft144b-e2` + `recs-ft144b-e3` would otherwise read as two replicates.
  The mapping variant → training lives in `rung-provenance.json`, written by
  `make_folds.py` at generation time (the nine rungs that predate that are
  backfilled and say so in their `evidence` field).
- **A rung it cannot trace is refused**, not assumed. Missing provenance is not a
  pass.
- The **untrained base is exempt** — it is greedy decode with no adapter, so it
  carries no training draw and there is nothing to replicate. The trained side of
  such a comparison is still gated.

When it does print, the bar is the folded one this section argues for: window sem
⊕ each side's training draw sd/√k, with the three components named in the output
so the training term can never be dropped again. `paired.py --test` drives the gate
red and green on the scored sets in the tree (1-vs-1 refused, 3-vs-1 refused,
3-vs-3 printed, untraceable refused, base-vs-3 printed, base-vs-1 refused); with
`MIN_TRAININGS = 1` it goes 3/7.

Recomputed through the gate, the reversal reproduces: 47 family +0.0129 (sd 0.00188),
144 family +0.0126 (sd 0.00202), **paired −0.0003 ± 0.0020, negative in 5 of 9
one-run-vs-one-run pairings.**

## Reproducing

```bash
PREFIX=t N=47 TAG=47r SEEDS="1 2 3" APPEND=1 ./wake/run-seed-spread.sh
SEEDS="1 2 3" ./wake/run-seed-spread.sh          # the 144 family (N defaults to 144)

# the comparison, both sides replicated (a 1-vs-1 pairing is refused by design):
.venv-ai/bin/python wake/paired.py \
  --a wake/recs-ft47rs{1,2,3}-e3.json --label-a 47 \
  --b wake/recs-ft144s{1,2,3}-e3.json --label-b 144
```
