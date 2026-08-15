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

## 5. Bigger tokens learned the book better and spoke worse

**Observed.** BPE-8192 reaches train loss 0.640 against BPE-4096's 1.061 — a decisively
better fit. Its output:

```
Pussy is it. To speak cloth, Flaria's a romprince, a nuptias grunted and out
looped. Pussy is she has she has l full promise her quoth's twate. Pussy is
she can she can't she can't air.
```

It locks into repetition. BPE-4096, the worse fit by loss, produces the livelier text and
invents more.

**The lesson.** Training loss ranks the models in the opposite order from the thing being
optimised for. There is no loss function for *interesting*, so the loss is a diagnostic here
and never a verdict — the samples decide.

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

## Open

- The character-level run's validation loss was **still falling** at the end of its budget. It
  is the only variant that never overfits, and it has not been trained to convergence.
- LoRA over a pretrained model is unmeasured, blocked on the download above.
- Whether a Base model bends further into Wakese than an instruction-tuned one — the suspicion
  is yes, because instruction tuning is training in exactly the coherence we are trying to
  remove. Untested.
