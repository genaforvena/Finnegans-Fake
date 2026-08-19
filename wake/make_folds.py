#!/usr/bin/env python3
"""Build wake/constructed-folds.json — the summary ladder for the constructed pairs.

The real summaries we already write scored zero off their quoted words. A single
constructed fold scoring zero would not say whether the fold is poor or the meter
is at its limit on this corpus, so the folds come as a ladder over the SAME
windows, each rung answering a different question:

  abstractive  written by hand against the window, in prose, carrying its content.
               The ceiling: if this scores zero, no prose fold of this corpus is
               visible to conditional NLL and the metric is done here.
  extractive   verbatim source sentences spread across the window at the same
               budget. What pure copying buys — and the mode our board lines are
               actually in (they are truncated prefixes).
  entities     the window's identifiers only — slugs, hosts, paths, numbers — no
               syntax. Tests the ledger finding's hypothesis directly: if the
               recoverable information is just the nouns, this rung matches the
               ones above it.
  model        what the local instruct model produces unaided. What we could
               actually serve, as opposed to what a careful writer can do.

Only `abstractive` is authored. The rest are derived here so they are auditable
and reproducible rather than hand-tuned.

  python make_folds.py            # all variants (loads the local model)
  python make_folds.py --no-model # skip the model rung
"""
import argparse, datetime, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pairs as pairmod

import os
OUT = pathlib.Path(os.environ.get("CONDENT_FOLDS",
                  pathlib.Path(__file__).resolve().parent / "constructed-folds.json"))
INSTRUCT = "/home/mesh-home/models/Qwen3.5-0.8B"

# ---------------------------------------------------------------------------
# Authored folds. Written against each window's text, aiming at the compression
# our handoffs claim (~5x) and at carrying content rather than flagging topic.
# ---------------------------------------------------------------------------
ABSTRACTIVE = {
"w00": """A night of connectivity thrash on the mesh's own senses, with two pieces of real
work inside it. The observer's reach broke and returned — the tunnel went down and came back,
while the legs to the model providers and to the phone stayed dark the whole window. Watchdog
lost one host to no-internet and recovered it, and lost this node twice, the second time worse:
off the tailnet with no local fallback at all. The other node's link flapped up and down three
times. Output stayed jammed throughout — autoland reported twice that the local branch had
diverged from the remote and its rebase conflicted, so nothing lands until a steward reconciles
by hand, and it carries incident priority. Housekeeping around the edges: two state-derived
tasks injected, two gigabytes of swap residue reclaimed, the living-room television to standby,
the media scene quiet. The work: the router's dead listener was traced to a firmware I/O bug and
fixed by switching the tunnel stack to a userspace one, with egress verified — but forwarded
local traffic is still uncaptured, and the service sits stopped and disabled with the operator
driving it manually. Separately the donor was swapped to a Montreal university address on its own
network to get out from under the geographic block, restarted and verified on the other node,
with the decision to re-enable left to the operator.""",

"w01": """Almost nothing happened, and the board says so at length. Eight identical injections
of a single state-derived task, four posture changes on the other node that amount to a flap
between nominal and watch with the household offline, and four self-readings reporting the same
stale reflex health — twice noting the board had gone silent for longer than the threshold, once
for nearly forty minutes, while the household was active. Two real items sit inside the
repetition. A steward-required fix waited through the entire window: one tool has a failing test,
it is not eligible for automatic landing, and it was reported unchanged at the start and again at
the end. And the node is under memory pressure — free swap fell to sixty-odd megabytes out of two
gigabytes with pressure saturated, the largest resident processes being the editor and the agent,
posted as a before-signal for exhaustion with nothing killed and the decision left to the kernel
or the operator. One security note in passing: a single address has been banned three times in six
hours on the ssh jail, which is scripted retry across the ban and expiry cycle rather than
background noise.""",

"w02": """One long report and its consequences. Genome landed an interpretant check into the
doctor tool and found the task had been half stale — the consumer-side sibling already existed,
and what was missing was exactly the load-bearing half the task named, plus a blindness nobody had
measured. Three findings. The slug pattern only ever matched one of two live naming conventions,
so both of the dead vehicles that a review had verified by hand in June were structurally
invisible to the very check built to catch them; emitters seen went from sixty-four to two hundred
and ten. Treating a write-then-read-back-next-run baseline as a closed loop — the same system at a
later time serving as its own reader — excluded forty-nine of sixty-two candidates, without which
the check is a backlog rather than a signal. And the trap that ate its own finding: the reader scan
searched the raw tree, so naming the two files in the tool's own explanatory comment made the tool
their reader and silenced both, in the same edit that documented them, caught only because a
comment-only change moved the count the wrong way. Thirteen dead vehicles stand after debouncing.
One correction to the earlier review — a tool called dead is not dead, it has five invokers and
consumes its own state as a rate baseline — and one confirmation. A side finding became its own
task: a tool declares a state artifact it never writes, invisible because the reach gate that would
surface it is permanently not-applicable while the phone is down.""",

"w03": """A window of degraded senses with one decision in it. The path outward flapped
repeatedly — restored, degraded past the jitter threshold with inference explicitly called at
risk, restored again — until the quality lane latched itself: three alerted episodes inside a day,
the last with forty percent loss, so it now reports once daily instead of on every bad-and-good
pair, and unlatches after a clean day. That latch is the only decision here. Around it: the
observer's own reach went down and came back once, the model provider stayed unreachable the entire
time and the phone stayed blind throughout, one peer fell from a direct link back to a relay, and
the self-reading kept returning stale reflex health with the board silent past its threshold.
Ambient sensing went from moderate and rising, on one television in neighbour range with a
permanent fixture excluded as carrying no clock information, to quiet and falling with no
appliances in twenty-five scans. Three more state-derived tasks were injected, three gigabytes of
swap residue were reclaimed, the media scene turned over to video, and the other node settled back
to nominal.""",

"w04": """The mind fleet was brought back from the dead twice over, on two nodes, and each time
the cause was something other than what it looked like. Here the login failure was neither
authentication nor the network: the authorization URL handed to the operator had been concatenated
with itself, twice the length it should have been, so its state parameter was garbage and the
endpoint refused it. The reason is a terminal detail — the URL is printed as a hyperlink
terminated by a bell character, and the extracting pattern ran straight through the bell and
swallowed the visible copy as well; the fix for any future scraper is to exclude that byte too.
The node itself was clean throughout, proxy chain and certificates checked both direct and
proxied. Underneath sat a second fault: the javascript runtime was on no path at all on this node,
so a session-end hook died on every single agent run. Both fixed and verified red to green with a
real round trip. Then all ten channel minds were stopped and relaunched — and parked on an
interactive theme picker, because the fresh login had reset the onboarding flag, and a mind that
relaunches into a picker is alive as a process and dead as a mind. Patched with every engine
stopped so none could clobber the write. What is offered as proof is an artifact rather than a
claim: one mind did a live round trip and posted its own heartbeat. The same shape appeared on the
other node, where a mind was running but logged out on expired credentials, showing a login prompt
while its process name still read normally. Fixed the same way, with its own round-trip artifact.
One gap was flagged and deliberately left open: that node declares a second mind that was merged
away three weeks ago and has never planted, and reviving it would spend a paid seat on a shared
login, so it is the operator's call. Meanwhile memory consumption was accelerating super-linearly
with a projected eighteen minutes to exhaustion, posted for visibility with nothing killed.""",

"w05": """Two long reports, and both of them correct the record. The first measures the block on
our own address and finds it address-level and carrier-specific: from that carrier's raw uplink
every port fails, while twenty-nine of thirty outside vantages reach the same address in about a
tenth of a second, including three inside the country. The consequence is stated flatly —
obfuscation cannot help, only a different address or a third-party path can. It then traces how
the household still reaches the internet at all, four hops deep, and corrects an earlier report
that had called one relay fully reverted: it is not, it carries the entire household, and this
node is an unmonitored single point of failure in the household's connectivity, which is the loop
that caused the blackout two days earlier. The real casualty is the friend lane — sixteen peers
configured and exactly one alive, the rest stale for weeks or never used since the day they were
built. One thing stays unmeasured and needs a single tap on a phone with its wifi off, because no
node can see the raw carrier any more now that the router tunnels everything. Four options went to
the operator and none were started. The second report applies the swarm literature's own critique
to the mesh's selection operator: a selector's preference has to be tested against a deliberately
uninformative objective, because you cannot read it off the code or off the outcome histogram.
Measured that way, every tier turns out position-bound — the winner is decided by slot — while the
identity histogram reads evenly spread, and the load-spreading path reaches only one class of
worker, leaving free-engine dispatch with no load spread at all and unmeasured until now. Seven
hermetic gates, a falsifier that flips the verdict, and no routing changed.""",

"w06": """The operator was on the line and three threads moved at once. On the job lane an inline
request went out asking him to open a mailbox and relay two verification codes; his answer, by
voice, was that he is in the middle of recovering access to that mailbox — so the silence was a
closed inbox rather than an unread message, the codes are blocked for reasons outside our control,
and the application window may run past its idle limit. On the advertising study the fourth
capture landed: seven hundred and fifty-two rows over thirty-five minutes across twenty-four
applications, and the result is a negative one — no exchanges at all, only attribution and
analytics endpoints, plus a single name lookup with no connection behind it. What makes that a
finding rather than an absence is the diagnostic: one application known to carry fourteen
mediation partners issued not a single lookup for any of them, so the software never asked, and
the resolver cannot be blamed because a blackhole would still have left the lookup in the log. The
protocol correction handed back to the operator is to hold each application until an advertisement
is actually visible and to mark where a banner appeared and where it did not, since otherwise an
empty row cannot be told apart from a component that never asked. On the sensing side the wireless
dongle is currently not flapping — nearly ten hours connected with no disconnect events in six,
and a lower rate of a known kernel event than in the July measurement — offered explicitly as one
window's observation and not as a claim that the standing fault is fixed. The attached phone is
holding a carrier data connection and is a candidate second uplink for the node, deliberately
untouched because the default route on a single path is load-bearing underneath the messaging
lane. Genome landed a literature lens on record dynamics, whose point is that an aging system
mints heavy tails with no critical point anywhere near it, and whose test needs no fitting and no
free exponent: conditional on the number of events, a constant rate puts arrivals uniform in
elapsed time and an aging rate puts them uniform in log-elapsed time, so map the same events both
ways and compare the two distances. Its coverage check killed three candidate concepts on
contact, and that check is presented as the work rather than as an obstacle to it.""",

"w07": """A closing and a cleanup, and in both the measuring instrument turned out to be the
thing at fault. Senses closed a family of five narrow-window accumulator defects — all fixed,
landed and deployed across four commits — and reported two side findings it does not own. The
first is that kernel worker time on this node is charged in bursts: two hundred and fifty-two
ticks inside one second, which any tool sampling processor use below a second reads as a figure
over a thousand percent, and over a third of a second as several thousand. It is genuinely in the
kernel's own accounting rather than a parsing error; the auditing tool now refuses impossible rows
in both windows and counts them, but the burst itself is unexplained and anything sampling that
fast on this node is reading it. The second is that a power tool's header quoted a firmware limit
from a file that does not exist on this processor family at all, so the capped reading it
cautioned about had never once been taken in the tool's whole life. The larger piece is the
temporary-file leak, and both halves were in code rather than in policy. The sense was wrong: the
disk check read a single filesystem, so its normal was an honest claim about a different
filesystem than the broken one; it now walks every real local mount, takes the worst of block and
inode use, and names which mount it means. The producers were more than the one the task named —
two more were measured, the worst stranding nearly four thousand files in two days, another
leaving almost two thousand fixture directories behind whenever an hourly timeout kills it. Each
now allocates inside one process-scoped directory under a trap, named after the tool so that a
survivor says who leaked it, and each went from one survivor to none in a sandboxed before-and-
after. The pushback is the interesting part: a trap does not cover an untrappable kill, so
trap-only would have read complete without being complete, and that is why the sweep is not
optional. The new guard probes each directory with a real write read back, on the grounds that a
percentage is not a permission — on a full filesystem the file is still created and only the write
fails, which is how tens of thousands of zero-byte files come to exist. It sweeps only the shapes
it can prove it owns and counts everything else without touching it, reporting what belongs to
another owner because a user-run guard cannot clean up after root. Every gate was seen red first.""",
}


def clean(s):
    return " ".join(s.split())


# ---------------------------------------------------------------------------
def extractive(text, budget_chars):
    """Verbatim sentences spread across the window — the copying ceiling.

    Coverage rather than lead: the window is cut into as many segments as will
    fit, and each contributes the opening of its longest line, so the result is
    a real extractive summary and not the truncated prefix our board lines are.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    seen, uniq = set(), []
    for l in lines:                      # this board repeats itself verbatim
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    k = max(1, min(len(uniq), 6))
    seg = max(1, len(uniq) // k)
    picks = []
    for i in range(0, len(uniq), seg):
        block = uniq[i:i + seg]
        if not block:
            continue
        best = max(block, key=len)
        body = best.split(": ", 1)[-1]
        m = re.split(r"(?<=[.!?]) ", body)
        picks.append(" ".join(m[:2])[:budget_chars // k])
    out = " ".join(picks)
    return clean(out)[:budget_chars]


IDENT = re.compile(
    r"(?:[a-z][a-z0-9]*(?:-[a-z0-9]+){1,}"      # slugged-names
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b"             # addresses
    r"|\b[0-9a-f]{7,40}\b"                      # hashes
    r"|\b[A-Z]{3,}\b"                           # SHOUTED states
    r"|\b\d+(?:\.\d+)?(?:ms|MB|GB|s|%|h)\b"     # measurements
    r"|\b\w+\.(?:py|sh|md|json|log|csv|state)\b"  # files
    r")")


def entities(text, budget_chars):
    """Identifiers only, in order of appearance, deduped. No syntax."""
    out, seen = [], set()
    for m in IDENT.findall(text):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return " ".join(out)[:budget_chars]


# The one prompt. Imported verbatim by train_fold.py so the trained student is
# asked exactly what the untrained baseline was asked — otherwise the training
# and a prompt change would be confounded.
PROMPT = ("Below are lines from an operations message board. Write a single dense "
          "paragraph summarising what happened: the concrete findings, decisions and "
          "faults, not the format. Do not list the lines; fold them.\n\n")


def model_folds(win_texts, budget_chars, adapter=None):
    """What the local instruct model produces — optionally with a LoRA on top."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(INSTRUCT)
    m = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16).to("cuda").eval()
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).eval()
        print(f"  [adapter] {adapter}")
    out = {}
    for wid, text in win_texts.items():
        msgs = [{"role": "user", "content": PROMPT + text[:6000]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False)
        ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            g = m.generate(ids, max_new_tokens=400, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)
        out[wid] = clean(txt)[:budget_chars]
        print(f"  {wid}: {len(out[wid])} chars")
    return out


PROVENANCE = pathlib.Path(__file__).resolve().parent / "rung-provenance.json"


def record_provenance(variant, adapter):
    """Write which TRAINING generated this rung, at generation time.

    paired.py refuses a between-rung delta unless each side carries >= 3 distinct
    trainings, and it counts DISTINCT trainings, not files — two epochs of one run
    are one draw. That count is only as good as this record, and a rung it cannot
    trace is refused rather than assumed. Reconstructing it later from directory
    names is a convention, not evidence; written here it is a fact recorded by the
    process that made the artifact.
    """
    entry = {"training": None, "adapter": None, "train": None,
             "evidence": f"written by make_folds.py at generation time "
                         f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"}
    if adapter:
        ap = pathlib.Path(adapter).resolve()
        entry["adapter"] = str(ap)
        entry["training"] = str(ap.parent)           # the run; ep1/ep2/ep3 share it
        tl = ap.parent / "trainlog.json"
        if tl.exists():
            log = json.loads(tl.read_text())
            args = log.get("args", {})
            entry["train"] = {k: args.get(k) for k in
                              ("seed", "data_seed", "n", "epochs", "data")}
            entry["train"]["n_train"] = log.get("n_train")
        else:
            entry["evidence"] += f" (no {tl} on disk: train args unknown)"
    else:
        entry["evidence"] += " (no --adapter: the UNTRAINED base rung, no training draw)"
    doc = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    doc.setdefault("_note", "variant -> the training that generated its folds. paired.py's "
                            "replication gate counts DISTINCT 'training' values per side; two "
                            "epochs of one run share a training and are one draw, not two.")
    doc.setdefault("variants", {})[variant] = entry
    PROVENANCE.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[provenance] {variant} <- {entry['training'] or 'untrained base'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--rung", action="store_true", help="generate only the model rung")
    ap.add_argument("--adapter", default=None, help="LoRA to generate the model rung with")
    ap.add_argument("--variant", default="model", help="key to store the generated rung under")
    a = ap.parse_args()

    wins = {w["id"]: w["text"] for w in pairmod.windows()}

    if a.adapter or a.rung:
        # only regenerate the named rung; everything else on disk is left alone
        old = json.loads(OUT.read_text()) if OUT.exists() else {}
        budget = max([len(v.get("abstractive", "")) for v in old.values()] or [0]) or 2000
        gen = model_folds(wins, budget, adapter=a.adapter)
        for wid, t in gen.items():
            old.setdefault(wid, {})[a.variant] = t
        OUT.write_text(json.dumps(old, ensure_ascii=False, indent=1))
        print(f"[out] {OUT} :: {a.variant}")
        record_provenance(a.variant, a.adapter)
        return
    absr = {k: clean(v) for k, v in ABSTRACTIVE.items()}
    # every rung gets the same character budget as the authored fold for its
    # window, so the ladder is not a length comparison in disguise
    budget = {k: len(v) for k, v in absr.items()}

    folds = {}
    for wid, text in wins.items():
        folds[wid] = {"abstractive": absr[wid],
                      "extractive": extractive(text, budget[wid]),
                      "entities": entities(text, budget[wid])}
    if not a.no_model:
        print("[model] generating with", INSTRUCT)
        for wid, t in model_folds(wins, max(budget.values())).items():
            folds[wid]["model"] = t
        record_provenance("model", None)

    if OUT.exists():                       # keep rungs we are not regenerating
        old = json.loads(OUT.read_text())
        for wid, v in old.items():
            folds.setdefault(wid, {})
            for k, t in v.items():
                folds[wid].setdefault(k, t)
    OUT.write_text(json.dumps(folds, ensure_ascii=False, indent=1))
    print(f"[out] {OUT}")
    for wid in sorted(folds):
        row = " ".join(f"{k}={len(v)}c" for k, v in sorted(folds[wid].items()))
        print(f"  {wid} src={len(wins[wid])}c  {row}")


if __name__ == "__main__":
    main()
