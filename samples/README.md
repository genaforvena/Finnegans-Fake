# samples/

Long-form output from every configuration, on prompts the book cannot answer.

The repo distributes neither the corpus nor the weights, so a reader has nothing to run and
every claim about what these models sound like has so far been a sentence in a README with a
four-line excerpt underneath, chosen by the person making the claim. These files are the
artifact instead: ~4000 characters each, ten prompts, eight configurations, eighty files,
regenerable with `python wake/make_samples.py`.

Three deliberate choices, each of which was a way of being wrong earlier:

**Length.** 4000 characters, not 200. The character models have a 512-token window, so a short
excerpt is entirely *inside* the window — the one regime where the model is at its most
fluent. Past it the model is continuing from a tail of its own output with the prompt gone,
which is a different thing and is invisible in the text. Every header says whether the window
slid.

**Prompts.** Nine of the ten are not from the book. Prompting the Wake model with the Wake's
own first line measures recitation and reports it as style. `riverrun-control` is kept so that
comparison is available rather than assumed.

**A number on every file.** `verb3`/`verb5` — the fraction of output word 3- and 5-grams found
verbatim in a corpus — against **both** corpora, because a sample that is novel against the
Wake may simply be ordinary English, and the product of experts is exactly the case where one
column cannot tell those apart.

## What the product of experts was for, and what happened

Two character models with an identical architecture and a **shared tokenizer** (`sha256`
checked, not assumed): one has read only *Finnegans Wake*, one has read only 1.31MB of
ordinary 19th-century English. At each step their next-character distributions are combined in
log space —

```
combined = w·log_softmax(A) + (1-w)·log_softmax(B)
```

— which is a geometric mean, a **product of experts**. The arithmetic mean would be a mixture:
mass wherever *either* expert puts mass, so the text alternates between the two voices. The
geometric mean is a conjunction: a character survives only if **both** find it likely. That
is a portmanteau's condition of existence, and the bet was that it would produce coinage by
construction rather than by luck.

It does not. It does the opposite, monotonically, and the mechanism is the same one that
motivated it.

Word types absent from **both** corpora — the model's own inventions, the thing we actually
wanted (`python wake/make_samples.py --coinages`):

| config | coinage rate | examples |
|---|---|---|
| wake-bpe4096 | 0.312 | a'lthe barbarren chapball divorval flitubbledarched hoky lemut muddled pearful roosolarb storkters tyrrushday |
| **wake-char257 (expert A alone)** | **0.238** | absoaked billetailing cherring deltogether farread hariodants lipsing misfour perition renduce showshers tansend |
| product w=0.9 | 0.195 | accupan behost carments criming doubtly giftness inclutable manundar orinal pullettee sentents suggestions |
| wake-lora-base | 0.134 | bawlstuffs blumblunderen blunterblun commitcode drabbledays frischer gitconfig goobies hollis morrowdayhad saftiger surnam |
| product w=0.7 | 0.115 | accomples blooder's carding crooks distrusting footspring indings miservant perition regretary seamed strengthed |
| product w=0.5 | 0.066 | accompanient benefaction clothest currents disputting flavourished hitchens maidst panished profuses shamely substantion |
| product w=0.3 | 0.044 | accusion brickscruped confanted counterful dishlike gaught innuent outlight premitting reviveries seale substant |
| **eng-char257 (expert B alone)** | **0.017** | 'kick's 'stop anchoragements becau butches childly destairs forefice hoarsehold mentioner promotions scalled |

**The product is strictly bracketed by its two experts at every weight tried, and monotone in
the weight.** 0.017 → 0.044 → 0.066 → 0.115 → 0.195 → 0.238. It never once exceeds expert A
alone. Whatever the product is doing, it is not manufacturing portmanteaus; it is *removing*
them, in proportion to how much ordinary English is mixed in.

The reason is not a bug and not a tuning failure. It is the definition. **A portmanteau is
precisely a string that ordinary English assigns near-zero probability to.** A geometric mean
is a veto — one near-zero factor drives the product to zero — so the English expert vetoes
exactly the characters that would have made a coinage. The conjunction that was supposed to
force two meanings at once is the same conjunction that forbids the only strings that could
carry them. "Probable under both lexicons" turns out to select *the intersection*, and the
intersection of Joyce's lexicon with ordinary English is ordinary English.

This was the expected failure mode and it is reported as the result, not tuned past.

### The one thing that did change, offered as an impression and not a measure

The coinages the product does make look different in kind. Expert A's are morpheme salad from
one lexicon — `showshers`, `hariodants`, `billetailing`, `deltogether`. The product's are more
often two *ordinary English* words fused: `accompanient`, `substantion`, `flavourished`,
`carments`, `pullettee`, `crowdlings`, `tradespadent`, `disodation`. Arguably that is closer to
a portmanteau in Carroll's sense than what expert A does.

Arguably. Twelve hand-read examples per row is not a measurement, nobody has defined "fused"
operationally, and the rate — the thing that *is* measured — went down. Recorded because it is
the interesting lead here, flagged because the temptation is to promote it to the finding.

## The other measurements

`python wake/make_samples.py --table`, read back out of the committed files rather than
carried in memory from the run that wrote them:

| config | n | verb3_wake | verb5_wake | novel_wake | verb3_eng | verb5_eng | novel_eng | self_repeat | ttr |
|---|---|---|---|---|---|---|---|---|---|
| eng-char257 | 10 | 0.130 | 0.000 | 0.076 | 0.291 | 0.004 | 0.019 | 0.000 | 0.410 |
| product-w03 | 10 | 0.117 | 0.000 | 0.098 | 0.206 | 0.002 | 0.052 | 0.000 | 0.435 |
| product-w05 | 10 | 0.121 | 0.000 | 0.102 | 0.189 | 0.000 | 0.085 | 0.000 | 0.423 |
| product-w07 | 10 | 0.121 | 0.000 | 0.133 | 0.149 | 0.001 | 0.156 | 0.001 | 0.435 |
| product-w09 | 10 | 0.107 | 0.000 | 0.202 | 0.116 | 0.001 | 0.261 | 0.001 | 0.442 |
| wake-bpe4096 | 10 | 0.092 | 0.003 | 0.324 | 0.058 | 0.000 | 0.463 | 0.006 | 0.543 |
| wake-char257 | 10 | 0.094 | 0.000 | 0.247 | 0.096 | 0.000 | 0.333 | 0.000 | 0.453 |
| wake-lora-base | 10 | 0.145 | 0.000 | 0.149 | 0.176 | 0.002 | 0.200 | 0.103 | 0.510 |

Two things worth reading off it that are not about the product:

- **`verb5` is 0.000 nearly everywhere.** No configuration reproduces a five-word run from
  either corpus. The copying that exists is real but it lives in three-word spans.
- **The product borrows *less* than either expert.** `verb3_eng` at w=0.5 is 0.189 against
  expert B's own 0.291, while `verb3_wake` stays around 0.12. The most fluent-sounding
  configuration here is also the least plagiarising one, which is a genuine effect of the
  conjunction even though it is not the effect that was wanted.
- **`wake-lora-base` has `self_repeat` 0.103**, an order of magnitude above everything else,
  and it is the only configuration that ever answers the prompt: given `recipe` it writes a
  recipe. It has an English prior it did not get from any corpus in this repo, and it shows.

## Files

`<config>__<prompt>.txt`. Header is `#`-commented; the text begins after the rule. Configs:
`wake-char257`, `eng-char257`, `wake-bpe4096`, `wake-lora-base`, `product-w03/w05/w07/w09`.

Every header states `best_val` and `final_val`. **Every from-scratch run in this repo overfit**
— `final_val > best_val` in all of them — and the final/overfit weights are the ones used here
anyway, for the reason `train_scratch.py` gives: on a one-book corpus, val loss measures
generalisation to unseen Joyce, which is not what anyone wants from this, and the best-val
checkpoint is an undertrained one that emits noise.

| run | best_val | final_val | |
|---|---|---|---|
| wake-char257 | 1.8840 | 1.9426 | overfit |
| eng-char257 | 1.2004 | 1.2065 | overfit |
| wake-bpe4096 | 5.8768 | 7.7704 | overfit |
| wake-lora-base | 3.5746 | 3.6558 | overfit |

`eng-char257` sits well below `wake-char257` on the same architecture, tokenizer, corpus size
and hyperparameters — 1.20 against 1.88. Ordinary English is simply more predictable than
*Finnegans Wake*, which is the one comparison in this repo where a loss difference across two
runs is meaningful, because for once nothing else differs.
