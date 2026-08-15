# Post-mortem

The repository describes itself as *a post-mortem on the failed llm-replica of Finnegans
Wake*. The irony is that it is being written while the patient is still training. So the
failures are the primary document, not an appendix to it, and each one is recorded the same
way: what was observed, what was believed, and what killed the belief.

Every entry below is an error in the work, not a property of the book.

---

## 1. The double label shift — the model was predicting two tokens ahead

**Observed.** The character-level model finished a full 6000-step run and produced:

```
cmcii,ai rpaece nclms nwbde,gil nlr nnaglns udtrmnddes u vnie ote,tmsoe
```

It reads as if every second letter had been deleted.

**Believed.** That the model was simply undertrained, and that *Finnegans Wake* was too
irregular to learn from — 2.54 nats/char after 56 epochs seemed to say the text had no
structure to find.

**What killed it.** The tokenizer was checked for exact round-trip first: encode a sentence,
decode it, compare character by character. It matched. That ruled out tokenisation and left
the objective. Reading the loss function's source rather than trusting the convention:

```python
# transformers.loss.loss_utils.ForCausalLMLoss
labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
shift_labels = labels[..., 1:].contiguous()
```

`transformers` shifts labels itself. The batcher was handing it labels already shifted by one
in the nanoGPT style, so the shift happened **twice** and every run had been learning to
predict token *t+2* from position *t*.

**Why it survived so long.** It never raised. No exception, no shape mismatch. Training ran,
loss fell, curves looked ordinary. A bug that produces a plausible number is far more
expensive than one that crashes.

**Cost.** Every run before the fix. After it, **400 steps beat the 6000 broken ones** — val
6.15 against 6.80.

---

## 2. Selecting the checkpoint by validation loss

**Observed.** The saved model emitted punctuation soup:

```
riverrun, past Eve and Adam's, B's'et ting M Mh, N, the ofor, E, that
the, Aes on S,ed, T theur aire of B, M, F the ofast of Bn.. E, tal,, ag,
```

**Believed.** That best-validation-loss is the checkpoint worth keeping. It is the default
habit, and it went unexamined.

**What killed it.** On a single-book corpus, validation loss measures generalisation to
*unseen Joyce* — a quantity nobody wants. Its optimum arrives while the model still knows
almost nothing, and the code was carefully saving exactly that. "Do not overfit" is a rule
for models that must work on new data; this one only ever has to work on a text it already
contains in full.

**Fix.** Both ends are kept: `wake/<run>/` is best-val, `wake/<run>/final/` is the end-of-run
model. The overfit one is the one that speaks.

---

## 3. Claiming the Wake is statistically incompressible

**Observed.** Validation perplexity around 900 against a 4096-token vocabulary — barely
better than drawing from a hat.

**Believed.** That this was a measurement *of the book*: that with 79.4% of word types
occurring exactly once, the text is close to incompressible and no small model could do
better. It was a satisfying claim and it was stated confidently.

**What killed it.** Entry 1. The perplexity was a property of the broken objective. With the
objective fixed, the same architecture reaches best-val 5.877 on BPE-4096, and the
character-level model reaches 1.88 nats/char and is *still falling* when the run ends.

**Standing correction.** The hapax rate is a measurement of the text and stands. Every
perplexity figure from before the fix is void. The interesting residue is a claim the numbers
now support rather than contradict: at the level of **letters** the book has plenty of
learnable structure, and at the level of **words** very little. That is a different and more
specific statement than the one first made.

---

## 4. Promising that a Wake-trained BPE would keep the portmanteaux whole

**Believed.** That training the tokenizer on the book itself, rather than borrowing an
English one, would make Joyce's coinages into single units.

**What killed it.** Printing them:

```
penisolate     -> pen / is / ol / ate
wielderfight   -> w / iel / der / f / ight
passencore     -> pass / enc / ore
```

The coinages are hapax legomena. BPE merges by frequency, and a word occurring once offers
nothing to count. It shreds them exactly as an ordinary English tokenizer would.

**What the failure turned out to be worth.** The shredding is the mechanism, not the defect.
The model inherits Wake-flavoured fragments — `iel`, `enc`, `onn`, `ght` — and coins by
recombining them. Whole learned words would have produced a quoting machine. Observed
coinages absent from the book: `wisheard`, `ephemerries`, `unsheeperted`, `gough`.

---

## 5. Reading a verdict out of one sample

**Observed.** BPE-8192 reaches train loss 0.640 against BPE-4096's 1.061 — a decisively
tighter fit. A sample from it ran:

```
Pussy is it. To speak cloth, Flaria's a romprince, a nuptias grunted and out
looped. Pussy is she has she has l full promise her quoth's twate. Pussy is
she can she can't she can't air.
```

**Believed.** That the model "locks into repetition", that it "speaks worse", and that this
followed from its tighter fit: having memorised the book most closely, it was walking
memorised paths.

**What killed it.** Measuring instead of asserting, at n=8 samples per model
(`wake/measure.py`).

Self-repetition — the fraction of 4-grams in the output that recur within the output — is
**~0.007 for every variant**, with no separation between them. The repetition was one sample
at one seed, promoted to a property of the model.

The causal half failed too. If a tight fit meant retracing the book, BPE-8192 should copy the
most. It has the **lowest** 5-gram verbatim rate of the three BPE models — 0.002 against
BPE-4096-small's 0.009. Low training loss did not turn into quotation.

**The deeper error.** "Worse" is not an observation. These are texts with different
properties, and a loss function ranks them on an objective nobody here is optimising for.
The fix is not a better ranking; it is refusing to rank and publishing the properties.

---

## What the outputs actually are

Eight samples per model, 900 characters each, temperature 0.85, with the book measured the
same way at the same length as a baseline:

| | self-repeat | verbatim 3-gram | verbatim 5-gram | novel words | type/token |
|---|---|---|---|---|---|
| **the book** | 0.000 | 0.995 | 0.994 | 0.252 | 0.702 |
| `wake-bpe4096` | 0.007 | 0.083 | 0.004 | **0.276** | 0.617 |
| `wake-bpe4096-small` | 0.009 | 0.105 | 0.009 | 0.240 | 0.594 |
| `wake-bpe8192` | 0.006 | 0.101 | 0.002 | 0.233 | 0.635 |
| `wake-char257` | 0.000 | 0.083 | 0.001 | 0.199 | 0.558 |

**Novel words** — the share of word types absent from the rest of the book — is the column
worth stopping at. Joyce coins at 0.252. A 12M-parameter model trained for twenty minutes on
one consumer GPU coins at 0.276. The *rate* of invention is reproduced; the **nature** of it
is not — a model's new word is frequently a broken one, where Joyce's is constructed. Rate
and kind are different claims and only the first is measured here.

**Type/token ratio** is where all four models differ from the book in the same direction:
0.56–0.64 against 0.702. They reuse vocabulary more than Joyce does. That this holds across
four variants that differ in everything else is what makes it look like a property rather
than an artefact of sampling.

Method notes that the numbers depend on, each of which was wrong first — see entries 7–9.

---

## 6. A silent stall read as a slow network

**Observed.** A model download sat at 150MB without moving. The node's uplink was the obvious
suspect, and switching to a phone tether was proposed.

**What killed it.** Measuring both ends instead of one. Same path, same minutes: 1.0 MB/s
sustained from Hetzner, 370 KB/s with mid-file stalls from HuggingFace's CDN. The wifi link
itself: -43 dBm, 78 Mbit/s, 0.7 ms to the router, no loss. The radio was never the problem.

The proposed replacement was measured too, rather than assumed: the phone's tether interface
was up and its gateway answered in 0.3 ms — and passed **zero bytes** outward. A live local
gateway with no egress looks exactly like a working uplink.

---

## 7. Comparing variants at equal token budgets

**Observed.** Side by side at 130 generated tokens each, the character-level model appeared to
produce a fragment while the BPE models produced paragraphs.

**Believed.** Briefly, that the character model had less to say.

**What killed it.** 130 tokens is roughly 390 characters at vocab 4096 and roughly 130 at
character level. The comparison had silently handed the coarse tokenizer three times the
text. The whole point of the table is to compare tokenisations, and the axis being compared
was the one left uncontrolled.

**Fix.** `compare.py` budgets in **characters** and converts per model, measuring
chars-per-token on the prompt itself.

---

## The four variants, after the fix

| run | vocab | params | train | best val | final val |
|---|---|---|---|---|---|
| `wake-bpe4096` | 4096 | 12.3M | 1.061 | 5.877 | 7.770 |
| `wake-bpe4096-small` | 4096 | 4.3M | 4.417 | 5.897 | 6.073 |
| `wake-bpe8192` | 8192 | 13.9M | 0.640 | 6.454 | 9.025 |
| `wake-char257` | 257 | 10.9M | 1.487 | **1.884** | 1.943 |

Two things in that table are worth more than the samples.

**The 4.3M model matches the 12.3M one on best validation loss** — 5.897 against 5.877 —
using a third of the parameters, and barely overfits (final val 6.07 against 7.77). Nearly
all the capacity in the larger model goes into memorising, not into learning.

**Only the character model never overfits.** Its validation loss was still falling when the
budget ran out, which is why it is the one now training on a longer run. The word-level
variants are done learning within about 1000 steps and spend the remaining 5000 memorising.

The character and BPE losses are in different units — nats per character against nats per
token — and are not comparable to each other. Only the columns within a tokenisation are.

---

## 8. A copy-detector set too strict to detect copying

**Observed.** Verbatim overlap with the book, measured on 8-grams, came out at exactly 0.000
for all four models.

**Believed.** Briefly, that none of them reproduce the book at all.

**What killed it.** A measure that returns the same value for every input is
indistinguishable from a broken one, so the threshold was swept rather than trusted. On 526
generated words:

| n-gram | share found in the book |
|---|---|
| 2 | 0.453 |
| 3 | 0.097 |
| 4 | 0.017 |
| 5 | 0.004 |
| 8 | 0.000 |

The copying is real; it lives in short spans. Nearly half of all adjacent word *pairs* come
from the book. Set the window at eight and every model looks equally original — a clean
number that means nothing.

---

## 9. A baseline that scored itself zero by construction

**Observed.** Measuring "share of word types not in the book" gave the book 0.004 — as a
baseline for the models' 0.20–0.28, this made them look wildly inventive.

**What killed it.** A slice of the book was being compared against the vocabulary of the
whole book, itself included. It cannot contain a word it does not contain. The comparison was
a tautology dressed as a control.

**Fix.** Each slice is scored against the book *minus that slice*, which is the same question
the models are asked. The book's real figure is **0.252**, not 0.004 — and that changes the
finding from "the models invent far more than Joyce" to "they invent at about the same rate",
which is the interesting version and would have been missed.

---

## Open

- The character-level run's validation loss was **still falling** at the end of its budget. It
  is the only variant that never overfits, and it has not been trained to convergence.
- LoRA over a pretrained model is unmeasured, blocked on the download above.
- Whether a Base model bends further into Wakese than an instruction-tuned one — the suspicion
  is yes, because instruction tuning is training in exactly the coherence we are trying to
  remove. Untested.
