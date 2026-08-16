#!/usr/bin/env python3
"""H(source | summary) — how much of a source a summary actually carries.

A summary is good if the source is more predictable given it. So: score the NLL
of the source's tokens under a base LM twice, once with the summary in context
and once without, and read the delta in nats per character. No reference
summaries needed — the source is its own reference.

The number is meaningless without controls, so this harness never reports it
alone. Every pair is scored under four context conditions at an EQUAL TOKEN
BUDGET:

  none      empty summary body (the zero-context reference; NOT length-matched,
            so deltas against it carry a context-length confound by construction)
  own       the pair's own summary, truncated to the budget
  foreign   other pairs' summaries — same genre, same length, wrong content.
            If own == foreign the metric is reading format and style, not
            content, and the whole exercise is zero. This is the control that
            can void the result on its own.
  shuffled  own summary's tokens in random order — same lexicon, same length,
            no structure. Separates "these words are around" from "this says
            something".

and every condition is reported three ways: over all source tokens, over source
tokens whose id also occurs in the own summary (OVERLAP — where a verbatim
quote drops NLL by copying, not by compression), and over the rest (NOVEL —
where real folding has to show up). The overlap partition is defined by the OWN
summary in every condition, so all conditions score the same token subsets.

The headline number is own minus foreign on the NOVEL subset: length-matched,
copy-free, content-specific.

Pair-agnostic on purpose. It takes (source_text, summary_text) and knows nothing
else; wake/pairs.py supplies the mesh's pairs.

  python condent.py --test                     # red/green gate, no pairs needed
  python condent.py --pairs handoff            # board [handoff] line vs handoff/<w>.md
  python condent.py --pairs handoff --dry-run  # inventory only, no model load
"""
import argparse, json, math, pathlib, random, statistics, sys, time
from dataclasses import dataclass, field

MODEL = "/home/mesh-home/models/Qwen3.5-0.8B-Base"

# The scaffold is identical in every condition, including `none`, so the first
# source token is always predicted from the same preceding token and the scored
# positions never shift between conditions.
SCAFFOLD_PRE = "SUMMARY:\n"
SCAFFOLD_POST = "\n\nSOURCE:\n"


@dataclass
class Pair:
    """One (source, summary) pair. Everything downstream sees only this."""
    name: str
    source: str
    summary: str
    meta: dict = field(default_factory=dict)
    # Pairs sharing a group never donate foreign summaries to each other. Needed
    # when one summary covers several sources (the ledger control scores the same
    # ledger against two different axes of the same board window) — without this
    # the "foreign" context would be the pair's own summary and the control would
    # silently read zero.
    group: str = None

    def gid(self):
        return self.group or self.name


class Scorer:
    """Per-token NLL of a source under an arbitrary token-id context."""

    def __init__(self, model_path=MODEL, device=None, max_tokens=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.max_tokens = max_tokens
        self.lsm_step = 256
        self.pre = self.tok(SCAFFOLD_PRE, add_special_tokens=False)["input_ids"]
        self.post = self.tok(SCAFFOLD_POST, add_special_tokens=False)["input_ids"]

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tok.decode(ids)

    def nll(self, body_ids, src_ids):
        """Returns a per-token NLL list (nats), len == len(src_ids).

        Context is assembled at the id level so src_ids is bit-identical across
        conditions — no tokenizer boundary drift between runs.
        """
        torch = self.torch
        ids = self.pre + list(body_ids) + self.post + list(src_ids)
        if len(ids) > self.max_tokens:
            raise ValueError(f"sequence {len(ids)} > max_tokens {self.max_tokens}")
        start = len(ids) - len(src_ids)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device))
            logits = out.logits[0, start - 1:-1]
            tgt = torch.tensor(src_ids, device=self.device)
            # chunked: a 248k vocab in float32 over a long source is gigabytes.
            # The GPU is shared with other windows, so back off rather than die.
            step = self.lsm_step
            while True:
                try:
                    nll = []
                    for i in range(0, logits.shape[0], step):
                        lp = torch.log_softmax(logits[i:i + step].float(), -1)
                        nll.append(-lp.gather(1, tgt[i:i + step, None])[:, 0])
                    nll = torch.cat(nll)
                    break
                except torch.OutOfMemoryError:
                    if step <= 16:
                        raise
                    step //= 4
                    self.lsm_step = step
                    torch.cuda.empty_cache()
                    print(f"[oom] log-softmax step -> {step}", file=sys.stderr)
            del out, logits
        return nll.cpu().tolist()


def partition(src_ids, ctx_id_set):
    """Split source positions into (overlap, novel) by token-id membership."""
    ov = [i for i, t in enumerate(src_ids) if t in ctx_id_set]
    nv = [i for i, t in enumerate(src_ids) if t not in ctx_id_set]
    return ov, nv


def subset_stats(nlls, src_ids, idxs, char_len):
    """nats total / per token / per char over a subset of source positions."""
    if not idxs:
        return {"n_tok": 0, "n_char": 0, "nats": 0.0, "npt": None, "npc": None}
    tot = sum(nlls[i] for i in idxs)
    chars = sum(char_len[i] for i in idxs)
    return {
        "n_tok": len(idxs),
        "n_char": chars,
        "nats": tot,
        "npt": tot / len(idxs),
        "npc": tot / chars if chars else None,
    }


def chunk_source(src_ids, chunk_tokens, n_chunks):
    """Evenly spaced windows across the source.

    A summary covers its whole source, so scoring only the head of a long one
    dilutes any signal by whatever fraction was left unread. Chunks are scored
    independently — each gets the full context — and pooled.
    """
    if len(src_ids) <= chunk_tokens:
        return [src_ids]
    n = min(n_chunks, max(1, len(src_ids) // chunk_tokens))
    if n == 1:
        return [src_ids[:chunk_tokens]]
    span = len(src_ids) - chunk_tokens
    return [src_ids[round(i * span / (n - 1)):round(i * span / (n - 1)) + chunk_tokens]
            for i in range(n)]


def score_pair(scorer, pair, foreign_summaries, budget, seed=0,
               chunk_tokens=1536, n_chunks=1):
    """All four conditions for one pair, each split all/overlap/novel."""
    chunks = chunk_source(scorer.encode(pair.source), chunk_tokens, n_chunks)
    src_ids = [t for c in chunks for t in c]
    own_ids = scorer.encode(pair.summary)[:budget]
    char_len = [len(scorer.decode([t])) for t in src_ids]

    ov, nv = partition(src_ids, set(own_ids))

    def run(body_ids):
        nlls = []
        for c in chunks:
            nlls.extend(scorer.nll(body_ids, c))
        return {
            "all": subset_stats(nlls, src_ids, list(range(len(src_ids))), char_len),
            "overlap": subset_stats(nlls, src_ids, ov, char_len),
            "novel": subset_stats(nlls, src_ids, nv, char_len),
        }

    rng = random.Random(seed)
    shuf = own_ids[:]
    rng.shuffle(shuf)

    conds = {"none": run([]), "own": run(own_ids), "shuffled": run(shuf)}

    # foreign: same genre, same budget, wrong content — averaged over donors
    fruns = []
    for fs in foreign_summaries:
        fids = scorer.encode(fs)[:budget]
        if len(fids) < budget:
            continue
        fruns.append(run(fids))
    if fruns:
        conds["foreign"] = {
            part: {
                k: (statistics.mean([f[part][k] for f in fruns])
                    if fruns[0][part][k] is not None else None)
                for k in fruns[0][part]
            }
            for part in ("all", "overlap", "novel")
        }
    conds["_n_foreign"] = len(fruns)

    return {
        "name": pair.name,
        "meta": pair.meta,
        "n_chunks": len(chunks),
        "src_coverage": len(src_ids) / max(1, len(scorer.encode(pair.source))),
        "n_src_tok": len(src_ids),
        "n_src_char": sum(char_len),
        "n_sum_tok_full": len(scorer.encode(pair.summary)),
        "budget": budget,
        "overlap_frac_tok": len(ov) / len(src_ids) if src_ids else 0.0,
        "cond": conds,
    }


def deltas(rec):
    """delta = nll(none) - nll(cond); positive means the context helped."""
    out = {}
    for cond in ("own", "foreign", "shuffled"):
        if cond not in rec["cond"]:
            continue
        for part in ("all", "overlap", "novel"):
            base = rec["cond"]["none"][part]["npc"]
            got = rec["cond"][cond][part]["npc"]
            out[f"{cond}.{part}"] = None if (base is None or got is None) else base - got
    for part in ("all", "overlap", "novel"):
        a, b = out.get(f"own.{part}"), out.get(f"foreign.{part}")
        out[f"signal.{part}"] = None if (a is None or b is None) else a - b
    return out


def report(records, fh=sys.stdout, by=None):
    def p(*a):
        print(*a, file=fh)

    ds = [deltas(r) for r in records]
    p("")
    p("=" * 100)
    p("H(source | summary) — delta NLL in nats/char, source of each pair scored under 4 contexts")
    p("=" * 100)
    p("")
    p(f"{'pair':<22} {'src_tok':>7} {'sum_tok':>7} {'ov%':>5} "
      f"{'d_own':>8} {'d_frgn':>8} {'d_shuf':>8} {'SIGNAL':>8} {'sig_novel':>9}")
    p("-" * 100)
    for r, d in zip(records, ds):
        f = lambda v: "   —   " if v is None else f"{v:+8.4f}"
        p(f"{r['name'][:22]:<22} {r['n_src_tok']:>7} {r['n_sum_tok_full']:>7} "
          f"{r['overlap_frac_tok']*100:>4.0f}% "
          f"{f(d['own.all'])} {f(d['foreign.all'])} {f(d['shuffled.all'])} "
          f"{f(d['signal.all'])} {f(d['signal.novel'])}")
    p("-" * 100)

    def block(subset, title):
        def agg(key):
            v = [d[key] for d in subset if d.get(key) is not None]
            if not v:
                return None
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)

        p("")
        p(f"AGGREGATE {title} (mean ± sd over {len(subset)} pairs, nats/char)")
        p("")
        p(f"  {'':<26} {'all tokens':>20} {'overlap':>20} {'novel':>20}")
        for cond, label in (("own", "delta vs no-summary"),
                            ("foreign", "  foreign summary"),
                            ("shuffled", "  shuffled own"),
                            ("signal", "SIGNAL own-foreign")):
            row = []
            for part in ("all", "overlap", "novel"):
                a = agg(f"{cond}.{part}")
                row.append("        —        " if a is None else f"{a[0]:+.4f} ± {a[1]:.4f}")
            p(f"  {label:<26} {row[0]:>20} {row[1]:>20} {row[2]:>20}")
        sig = [d["signal.novel"] for d in subset if d.get("signal.novel") is not None]
        if sig:
            pos = sum(1 for v in sig if v > 0)
            p(f"  sign test on SIGNAL/novel: {pos}/{len(sig)} pairs positive")

    block(ds, "")
    if by:
        groups = {}
        for r, d in zip(records, ds):
            groups.setdefault(r["meta"].get(by, "?"), []).append(d)
        for k in sorted(groups):
            block(groups[k], f"— {by}={k}")
    p("")
    p("READ IT LIKE THIS:")
    p("  d_own vs no-summary is NOT the answer — it carries a context-length confound")
    p("  (the `none` condition has zero context tokens by construction).")
    p("  SIGNAL = own - foreign is length-matched: both contexts are the same number of")
    p("  tokens of the same genre, so what is left is content.")
    p("  SIGNAL/novel excludes source tokens the summary quotes verbatim, so it is the")
    p("  part that copying cannot buy. SIGNAL near zero = the summary carries nothing")
    p("  its own genre does not already carry.")
    p("")


# ---------------------------------------------------------------- self-test

# Four passages on unrelated topics with a one-line topic summary each. The
# summary's own words are few; almost all of each passage's topic vocabulary is
# NOT in its summary, so any gain there has to travel through the novel subset.
TOPICS = [
    ("A cold-water coral reef grows in permanent darkness on the continental slope.",
     "The polyps take no light and keep no algae inside their tissue, so the whole "
     "colony feeds by catching particles that drift down from the surface far above. "
     "Growth is slow, a few millimetres in a year, and the framework the dead skeletons "
     "leave behind outlasts the animals by millennia. Trawl gear dragged across such a "
     "mound flattens in one pass what took eight thousand years to build up."),
    ("A pipe organ's wind system must hold steady pressure while stops are drawn.",
     "The bellows feed a reservoir, and the reservoir feeds the chests under each rank. "
     "If the supply sags when a full chord is pulled, every pipe already speaking will "
     "drop in pitch, and the fault is heard as a shudder rather than as a wrong note. "
     "Tuners chase this by weighting the reservoir and by widening the trunk, never by "
     "adjusting the pipes themselves, which were true all along."),
    ("Rye bread is leavened by a sour culture rather than by baker's yeast.",
     "The acid the culture produces does more than flavour the crumb: it holds the "
     "starches in check long enough for the loaf to set, because the grain carries an "
     "enzyme that would otherwise break the interior down into paste while it bakes. "
     "This is why a loaf made from that grain with commercial yeast alone collapses in "
     "the middle, and why the old method was never an affectation."),
    ("A marine diesel engine burns residual fuel that must be heated before injection.",
     "Cold, the oil is nearly solid; the plant carries it in steam-traced tanks and "
     "raises it through a series of heaters until the viscosity at the pump matches what "
     "the injectors were designed around. Get it wrong in the cold direction and the "
     "spray pattern collapses into a jet that washes the liner; get it wrong hot and the "
     "fuel cracks in the line. The control loop watches viscosity itself, not temperature."),
]


def _topical():
    """(source, summary) pairs whose shared content is topic, not wording."""
    return [Pair(f"topic{i}", body, summ) for i, (summ, body) in enumerate(TOPICS)]


def _synth(rng, n_facts=14):
    """A source with n_facts arbitrary facts buried in filler, and a summary
    that lists exactly those facts. Nothing but the summary can predict them."""
    words = ["ZORB", "KRAL", "VUNT", "MIXO", "PLEB", "TARN", "GRIV", "OSKA",
             "WEND", "FLUM", "DERZ", "QOPA", "HINT", "BRAX", "LUME", "SVAR"]
    rng.shuffle(words)
    facts = [(w, rng.randint(1000, 9999)) for w in words[:n_facts]]
    filler = ("The unit was inspected on the usual schedule and the readings were "
              "entered into the log without incident. ")
    src = []
    for w, n in facts:
        src.append(f"{filler}Channel {w} settled at {n} counts. ")
    summary = "; ".join(f"{w}={n}" for w, n in facts)
    return "".join(src), summary


def selftest(scorer):
    """Four gates. Each channel is tested with content that channel can carry,
    and the set is red in both directions — it can fail for finding too much as
    well as too little.

    D structural, no model: the overlap/novel split is disjoint, exhaustive, and
      identical across conditions, so the conditions are comparable at all.
    A topical: summaries that share TOPIC but almost no wording with their source
      -> own must beat foreign, including on the NOVEL subset. This is the only
      gate that proves the harness sees content travelling through tokens the
      summary never contained.
    B uninformative: one constant summary shared by every pair -> own must NOT
      beat foreign. The gate that proves the harness can return zero; without it
      a metric that scores everything positive passes A and is still worthless
      (POSTMORTEM entry 9, 'a baseline that scored itself zero by construction').
    C verbatim: summaries that are literal fact lists -> the gain must land on
      the OVERLAP subset, far above the novel one, i.e. copying is detected and
      quarantined rather than banked as compression.
    """
    ok = True

    # D — structural, cheap, and it gates the meaning of every other number
    src_ids = scorer.encode(TOPICS[0][1])
    ctx = set(scorer.encode(TOPICS[0][0]))
    ov, nv = partition(src_ids, ctx)
    d_ok = (sorted(ov + nv) == list(range(len(src_ids)))
            and not (set(ov) & set(nv))
            and all(src_ids[i] in ctx for i in ov)
            and all(src_ids[i] not in ctx for i in nv))
    print(f"  [D structural] overlap {len(ov)} + novel {len(nv)} = {len(src_ids)} source tokens")
    print(f"     {'PASS' if d_ok else 'FAIL'}  partition disjoint, exhaustive, id-correct")
    ok &= d_ok

    rng = random.Random(1234)
    pairs_a = _topical()
    facts = [_synth(rng) for _ in range(4)]
    pairs_c = [Pair(f"fact{i}", s, m) for i, (s, m) in enumerate(facts)]
    const = "; ".join(f"NOTE={1000+i}" for i in range(14))
    pairs_b = [Pair(f"flat{i}", s, const) for i, (s, _) in enumerate(facts)]

    gates = {
        "A topical": (pairs_a, [("signal.all  > 0.010", lambda s: s["sig"] > 0.010),
                                ("signal.novel> 0.005", lambda s: s["sig_nv"] > 0.005)]),
        "B uninformative": (pairs_b, [("|signal.all| < 0.020 (must score zero)",
                                       lambda s: abs(s["sig"]) < 0.020)]),
        "C verbatim": (pairs_c, [("own.overlap > 10x own.novel",
                                  lambda s: s["own_ov"] > 10 * abs(s["own_nv"]))]),
    }
    for label, (pairs, checks) in gates.items():
        budget = min(len(scorer.encode(p.summary)) for p in pairs)
        recs = [score_pair(scorer, p, [q.summary for q in pairs if q is not p], budget)
                for p in pairs]
        ds = [deltas(r) for r in recs]
        s = {"sig": statistics.mean(d["signal.all"] for d in ds),
             "sig_nv": statistics.mean(d["signal.novel"] for d in ds),
             "own_ov": statistics.mean(d["own.overlap"] for d in ds),
             "own_nv": statistics.mean(d["own.novel"] for d in ds)}
        print(f"  [{label}] budget={budget}tok signal.all={s['sig']:+.4f} "
              f"signal.novel={s['sig_nv']:+.4f} own.overlap={s['own_ov']:+.4f} "
              f"own.novel={s['own_nv']:+.4f}")
        for desc, fn in checks:
            got = fn(s)
            print(f"     {'PASS' if got else 'FAIL'}  {desc}")
            ok &= got
    print(f"\n  --test {'GREEN' if ok else 'RED'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None, help="pair set name (see wake/pairs.py)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--budget", type=int, default=None,
                    help="summary token budget; default = min over the pair set")
    ap.add_argument("--n-foreign", type=int, default=8)
    ap.add_argument("--max-src-tok", type=int, default=1536, help="tokens per source chunk")
    ap.add_argument("--n-chunks", type=int, default=1,
                    help="chunks spread across a long source (1 = head only)")
    ap.add_argument("--min-src-tok", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by", default=None, help="split the aggregate by this meta key")
    ap.add_argument("--out", default=None, help="write per-pair records as JSON")
    ap.add_argument("--dry-run", action="store_true", help="inventory only, no model")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    if a.test:
        sys.exit(0 if selftest(Scorer(a.model)) else 1)

    if not a.pairs:
        ap.error("need --pairs or --test")

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import pairs as pairmod
    pairs, notes = pairmod.load(a.pairs)
    for n in notes:
        print(f"[pairs] {n}")
    print(f"[pairs] {len(pairs)} pair(s) in set '{a.pairs}'")
    if a.dry_run:
        for p in pairs:
            print(f"  {p.name:<24} src {len(p.source):>6}c  sum {len(p.summary):>5}c  "
                  f"{len(p.source)/max(len(p.summary),1):>5.1f}x  {p.meta}")
        return

    t0 = time.time()
    sc = Scorer(a.model)
    print(f"[model] {a.model} on {sc.device} in {time.time()-t0:.1f}s")

    kept, skipped = [], []
    for p in pairs:
        n = len(sc.encode(p.source))
        if n < a.min_src_tok:
            skipped.append((p.name, f"source {n} tok < {a.min_src_tok}"))
            continue
        kept.append(p)

    # An explicit budget drops pairs that cannot fill it rather than padding
    # them — a short summary compared at a long budget is not length-matched.
    budget = a.budget or min(len(sc.encode(p.summary)) for p in kept)
    if a.budget:
        short = [p for p in kept if len(sc.encode(p.summary)) < budget]
        for p in short:
            skipped.append((p.name, f"summary {len(sc.encode(p.summary))} tok < budget {budget}"))
        kept = [p for p in kept if p not in short]
    for name, why in skipped:
        print(f"[skip] {name}: {why}")
    print(f"[budget] {budget} summary tokens (equal in every condition)")
    print(f"[source] up to {a.n_chunks} chunk(s) of {a.max_src_tok} tok, spread across the source")

    rng = random.Random(a.seed)
    records = []
    for i, p in enumerate(kept, 1):
        others = [q.summary for q in kept if q.gid() != p.gid()
                  and len(sc.encode(q.summary)) >= budget]
        if not others:
            print(f"[skip] {p.name}: no foreign donor outside its group")
            continue
        rng.shuffle(others)
        t = time.time()
        records.append(score_pair(sc, p, others[:a.n_foreign], budget, seed=a.seed,
                                  chunk_tokens=a.max_src_tok, n_chunks=a.n_chunks))
        print(f"[{i}/{len(kept)}] {p.name} ({time.time()-t:.1f}s, "
              f"{records[-1]['n_chunks']} chunk, {records[-1]['src_coverage']*100:.0f}% of source)")

    report(records, by=a.by)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"model": a.model, "pairs": a.pairs, "budget": budget,
             "n_foreign": a.n_foreign, "records": records}, ensure_ascii=False, indent=1))
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
