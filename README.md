# Finnegans Fake

> Bite my laughters, drink my tears. Pour into me. Volumes. Spell me and. Stark and spill me
> swooning. I just don't care what my daughters think.

Small language models that have read exactly one book.

The aim is not a good model. It is to find out what a language model does when the only
language it has ever seen is *Finnegans Wake* — whether anything that resembles English
survives, and whether the machine can coin words the way Joyce did rather than merely quote
the ones he already coined.

## What the corpus turns out to be

Measured on the cleaned text (`wake/prepare_corpus.py`), 224,527 running words:

| | |
|---|---|
| distinct word types | 58,725 |
| types occurring exactly once | 46,599 (**79.4%**) |
| running text covered by types seen ≥5 times | 69.8% |
| distinct characters | 105 |

Four out of five word types are hapax legomena. This settles the tokenisation question
before any training happens: a **word-level vocabulary is not merely coarse here, it is
impossible**. Most types would carry a single training example, and any unseen word becomes
`<unk>` — so the model could never invent one, which is the only thing we want from it.

## Tokenisation

BPE trained **on the Wake itself**, not borrowed from an English corpus. Three sizes, and one
degenerate case that is really character level:

| vocab | tokens | chars/token |
|---|---|---|
| 8192 | 403,331 | 3.25 |
| 4096 | 440,084 | 2.98 |
| 257 (byte/char) | 1,313,049 | 1.00 |

Training the tokenizer on the book does **not** make the coinages into single units, and it
is worth being clear about that because it is the intuitive expectation:

```
penisolate     -> pen / is / ol / ate
wielderfight   -> w / iel / der / f / ight
passencore     -> pass / enc / ore
```

They are hapax; BPE has nothing to count. It shreds them. That shredding is the useful
outcome rather than the failure — the model inherits Wake-flavoured *morphemes* (`iel`,
`enc`, `onn`, `ght`) and generates by recombining them. Learned whole words would give a
quoting machine; fragments give a coining machine.

### The scan-split trap

The source text breaks Joyce's words across lines: `passen-` newline `core`. Fed to a
tokenizer unchanged, the coinage `passencore` becomes two ordinary English fragments.
Rejoining them is the first line of `prepare_corpus.py` and it preserves a large share of the
wordplay the project exists to study.

## A bug worth keeping in the README

Every loss figure this file first carried was produced by a broken objective, and the way it
was caught is the useful part.

`batch()` returned nanoGPT-style pre-shifted labels (`x = d[j:j+B]`, `y = d[j+1:j+1+B]`).
But `transformers` shifts labels *itself* — `ForCausalLMLoss` pads and slices
`labels[..., 1:]`. So the shift happened twice and every run learned to predict token **t+2**
from position t.

It never raised. Training ran, loss fell, the curves looked plausible, and the inflated
perplexity supported a confident story about *Finnegans Wake* being statistically
incompressible. That story was a property of the bug.

Only the generated text exposed it, by coming out looking like every second character had
been deleted:

```
cmcii,ai rpaece nclms nwbde,gil nlr nnaglns udtrmnddes u vnie ote,tmsoe
```

The tokenizer was checked for exact round-trip first, which ruled out tokenisation and left
the objective. After the fix, **400 steps beat the 6000 broken ones** (val 6.15 vs 6.80):

```
riverrun, past Eve and Adam's, when you know it! But you're I'll don't
I can me, you! With me, I take the Pa's a way for you will be will my
gough to have being. I am your hear so, I'll you she and I'll not, you be be
```

English, dislocated on schedule, and `gough` is the model's own coinage — it is not in the
book. The corpus statistics above are measurements of the text and stand. All perplexity and
overfitting figures are being remeasured; this section will carry the real ones.

Both ends of each run are saved — `wake/<run>/` is best-val, `wake/<run>/final/` is the
end-of-run model.

## Saying several things at once — and why it fails

The Wake model coins words, but by luck: it is sampling from one lexicon and sometimes lands
between two of Joyce's morphemes. A portmanteau is something stricter — a string readable as
**two lexicons at once** — and a conjunction of two distributions is their *product*, not their
average. So: two character models, identical architecture, identical hyperparameters,
**identical tokenizer**, one having read only the Wake and one having read only 1.31MB of
ordinary 19th-century English (`data/english_clean.txt`, eight public-domain novels matched to
the Wake's own character count to within 0.2%). At each step,

```
combined = w·log_softmax(A) + (1-w)·log_softmax(B)   # renormalise, then temp, then top-p
```

The arithmetic mean would be a *mixture* — mass wherever either expert puts mass — so the text
would alternate between the two voices. The geometric mean is a *conjunction*: a character
survives only if both experts find it likely. That is a portmanteau's condition of existence.

**It does the exact opposite of what it was built for, monotonically.** Word types absent from
*both* corpora — the model's own inventions:

| expert B alone | w=0.3 | w=0.5 | w=0.7 | w=0.9 | expert A alone |
|---|---|---|---|---|---|
| 0.017 | 0.044 | 0.066 | 0.115 | 0.195 | 0.238 |

The product is strictly bracketed by its two experts at every weight and never once exceeds
expert A alone. And the reason is not a bug or a tuning failure, it is the definition: **a
portmanteau is precisely a string that ordinary English assigns near-zero probability to.** A
geometric mean is a veto — one near-zero factor kills the product — so the English expert
vetoes exactly the characters that would have made a coinage. "Probable under both lexicons"
selects the *intersection*, and the intersection of Joyce's lexicon with ordinary English is
ordinary English. The conjunction that was supposed to force two meanings at once is the same
conjunction that forbids the only strings that could carry them.

Full numbers, the qualitative lead that did survive (the product's rarer coinages look more
like two ordinary English words fused — `accompanient`, `substantion`, `flavourished` — offered
as an impression, not a measure), and eighty generated files: **[`samples/`](samples/)**.

One thing the second expert settles cleanly. `eng-char257` reaches val 1.20 where
`wake-char257` reaches 1.88, on the same architecture, tokenizer, corpus size and
hyperparameters. It is the only loss comparison in this repo where nothing else differs, so for
once the difference means what it looks like: ordinary English is more predictable than
*Finnegans Wake*.

## Running it

The corpus is **not distributed here**. Joyce died in 1941, so the text is public domain in
Ireland, the UK and the EU, but not in the US until 2035. Put a plain-text *Finnegans Wake*
at `data/wake_reconstructed_raw.txt` yourself.

```bash
python wake/prepare_corpus.py                       # clean + rejoin split words
python wake/train_tokenizer.py 4096                 # BPE on the Wake itself
python wake/train_scratch.py --vocab 4096 --iters 6000
python wake/sample.py wake-bpe4096/final --chat     # play
```

The plain-English expert and the product of experts. The English corpus *is* redistributable
(Project Gutenberg, public domain), it is just a 6MB download rather than a repo artifact:

```bash
python wake/prepare_english.py --fetch              # 8 novels -> data/english_clean.txt
python wake/train_scratch.py --data data/english_clean.txt --vocab 257 \
    --block 512 --batch 24 --iters 6000 --out eng-char257      # wake-char257's args verbatim
python wake/product_sample.py wake-char257/final eng-char257/final \
    --compare --weights 0.3,0.5,0.7
python wake/make_samples.py                         # regenerate samples/
python wake/make_samples.py --coinages              # the table that settles it
```

`--vocab 257` is load-bearing, not a default: both models must share
`wake/tokenizer-257.json` or the two logit vectors index different symbols and the sum above is
arithmetic on nonsense. `product_sample.py` refuses to run unless the two `tokenizer.json`
files hash equal — equal vocab *sizes* are not a shared vocabulary, and the mismatched version
would still run and still print plausible text.

LoRA over a small pretrained model, for something that answers back in English-but-dislocated:

```bash
python wake/train_lora.py --base Qwen/Qwen3.5-0.8B-Base --epochs 3
```

Trained weights are also kept out of the repo: they are a derivative of the source text, so
this project's CC0 cannot reach them.

## Licence

CC0. The code is public domain. The book is Joyce's problem, and yours.
