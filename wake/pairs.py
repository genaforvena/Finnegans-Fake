#!/usr/bin/env python3
"""Mesh-specific (source, summary) pair sets for condent.py.

condent.py knows nothing about the mesh; this is the only file that does. Each
loader returns (list[Pair], list[str] notes) — the notes carry whatever is true
about the corpus that the number would otherwise hide.

  handoff   board [handoff] line (summary) vs ~/.mesh/handoff/<window>.md (source)
            The pair as filed. CAVEAT, measured not assumed: mesh-handoff keeps a
            MANUAL handoff for KEEP_MANUAL_SECS=900 only, after which the
            5-minute --snapshot cron overwrites it with a verbatim pane scrape
            (`# source: auto-snapshot`). So the .md side has a ~15-minute
            lifetime as full state. Board [handoff] lines span days; the .md
            files on disk are minutes old. The loader reports the age gap per
            pair and tags the .md tier, because a scrape of a terminal pane is
            not the full state the board line was pointing at.

  session   board [handoff] line (summary) vs that window's OWN board lines
            since its previous handoff (source). Both sides on disk, aligned in
            time by construction, and a genuine fold rather than a selection.
            This is the pair set that survives the caveat above.
"""
import pathlib, re, sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from condent import Pair

MESH = pathlib.Path.home() / ".mesh"
CHATLOG = MESH / "chat.log"
HANDOFF_DIR = MESH / "handoff"

LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(\S+?)@(\S+)\s+::\s+(.*)$")
TAG = re.compile(r"^\[([a-z0-9-]+)\]\s*")


def _ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def read_board(path=CHATLOG):
    """[(ts, window, tag, text)] — continuation lines folded into their entry."""
    out = []
    for raw in path.read_text(errors="replace").splitlines():
        m = LINE.match(raw)
        if m:
            ts, win, _host, body = m.groups()
            tag = TAG.match(body)
            out.append([_ts(ts), win, tag.group(1) if tag else "", body])
        elif out:
            out[-1][3] += "\n" + raw
    return [tuple(r) for r in out]


def _strip_handoff_prefix(body, window):
    """`[handoff] wake: text` -> `text`"""
    b = TAG.sub("", body)
    if b.lower().startswith(window.lower() + ":"):
        b = b[len(window) + 1:]
    return b.strip()


def _md_body(path):
    """Drop the `# ...` header block; return the content the file actually carries."""
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip()


def _md_meta(path):
    head = path.read_text(errors="replace").splitlines()[:6]
    src = written = None
    for h in head:
        if h.startswith("# source:"):
            src = h.split(":", 1)[1].strip()
        if h.startswith("# written:"):
            written = h.split(":", 1)[1].strip()
    return src, written


def load_handoff():
    notes, pairs = [], []
    board = read_board()
    hs = [b for b in board if b[2] == "handoff"]
    notes.append(f"{len(hs)} [handoff] board lines in chat.log "
                 f"({hs[0][0].date()} .. {hs[-1][0].date()})" if hs else "no [handoff] lines")

    mds = sorted(HANDOFF_DIR.glob("*.md"))
    tiers = {}
    for f in mds:
        src, _ = _md_meta(f)
        tiers[src] = tiers.get(src, 0) + 1
    notes.append(f"{len(mds)} handoff/*.md on disk, tier: " +
                 ", ".join(f"{k}={v}" for k, v in sorted(tiers.items(), key=lambda x: -x[1])))
    notes.append("mesh-handoff KEEP_MANUAL_SECS=900 — a manual handoff survives 15 min, "
                 "then the 5-min --snapshot cron replaces it with a pane scrape")

    # latest board [handoff] per window
    latest = {}
    for ts, win, _tag, body in hs:
        latest[win] = (ts, body)

    for f in mds:
        win = f.stem
        if win not in latest:
            continue
        ts, body = latest[win]
        src_tier, written = _md_meta(f)
        gap = None
        if written:
            gap = (_ts(written) - ts).total_seconds()
        source = _md_body(f)
        summary = _strip_handoff_prefix(body, win)
        if not source or not summary:
            continue
        pairs.append(Pair(
            name=f"handoff/{win}",
            source=source,
            summary=summary,
            meta={"md_tier": src_tier,
                  "gap_h": None if gap is None else round(gap / 3600, 1),
                  "board_ts": ts.isoformat()},
        ))
    unmatched = sorted({f.stem for f in mds} - set(latest))
    if unmatched:
        notes.append(f"{len(unmatched)} .md with no board [handoff] line, dropped: "
                     + ", ".join(unmatched))
    gaps = [p.meta["gap_h"] for p in pairs if p.meta["gap_h"] is not None]
    if gaps:
        notes.append(f"age gap .md-minus-board: median {sorted(gaps)[len(gaps)//2]:.1f}h, "
                     f"min {min(gaps):.1f}h, max {max(gaps):.1f}h — a large gap means the .md "
                     f"on disk is NOT the full state that board line pointed at")
    return pairs, notes


def load_session(lookback_h=48, min_lines=3):
    """Source = a window's own board lines since its previous [handoff]."""
    notes, pairs = [], []
    board = read_board()
    by_win = {}
    for rec in board:
        by_win.setdefault(rec[1], []).append(rec)

    n_hand = 0
    for win, recs in by_win.items():
        recs.sort(key=lambda r: r[0])
        prev_end = None
        for i, (ts, _w, tag, body) in enumerate(recs):
            if tag != "handoff":
                continue
            n_hand += 1
            lo = prev_end or (ts.timestamp() - lookback_h * 3600)
            lo = lo if isinstance(lo, float) else lo.timestamp()
            prev_end = ts
            src_lines = [b for (t, _w2, tg, b) in recs[:i]
                         if t.timestamp() > lo and tg != "handoff"]
            if len(src_lines) < min_lines:
                continue
            pairs.append(Pair(
                name=f"session/{win}@{ts.strftime('%m%dT%H%M')}",
                source="\n".join(src_lines),
                summary=_strip_handoff_prefix(body, win),
                meta={"n_src_lines": len(src_lines), "board_ts": ts.isoformat()},
            ))
    notes.append(f"{n_hand} [handoff] lines seen; {len(pairs)} had >= {min_lines} own board "
                 f"lines since the window's previous handoff (lookback {lookback_h}h)")
    notes.append("source and summary are both chat.log, written by the same window, "
                 "aligned in time by construction — no external file to decay")
    return pairs, notes


# commitment-bearing board tags — the axis mesh-promises projects the board onto.
# Everything else is the content the ledger throws away by declaration.
COMMIT_TAGS = {"task", "done", "verify", "taking", "claim", "retract", "dispatch"}


def load_ledger(days=8, min_lines=6):
    """The fourth control: a fold with KNOWN losses, used to calibrate the meter.

    mesh-promises folds the board for real, but onto ONE axis — who owes what,
    settled or not — discarding everything else by declaration. That makes its
    losses checkable by construction, which is exactly what a prose summary's
    are not. So it yields a two-sided prediction the metric MUST reproduce:

      commitment axis  recovery improves a lot
      content          recovery barely improves

    A large gain on content is not a good summary, it is a broken meter — it
    would mean the metric is reading lexical overlap rather than information,
    and the thing to fix is the instrument, before any run is spent on it.

    Each board day yields two pairs sharing one summary (that day's ledger) and
    one donor group, so the two axes cannot serve as each other's foreign control.
    """
    import subprocess, tempfile
    notes, pairs = [], []
    board = read_board()
    by_day = {}
    for rec in board:
        by_day.setdefault(rec[0].date().isoformat(), []).append(rec)
    day_keys = sorted(by_day)[-days:]

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="condent-ledger-"))
    empty = 0
    for day in day_keys:
        recs = by_day[day]
        slice_path = tmp / f"board-{day}.log"
        slice_path.write_text("\n".join(
            f"{t.strftime('%Y-%m-%dT%H:%M:%SZ')}  {w}@mesh-home  ::  {b}"
            for t, w, _tg, b in recs) + "\n")
        env = {**__import__("os").environ, "MESH_CHAT_LOG": str(slice_path)}
        try:
            led = subprocess.run(["mesh-promises", "--balance"], env=env,
                                 capture_output=True, text=True, timeout=180).stdout.strip()
        except Exception as e:
            notes.append(f"{day}: mesh-promises failed: {e}")
            continue
        # a ledger with no standing obligations is a header and nothing else
        if len([l for l in led.splitlines() if l.strip().startswith(("1 ", "  1"))]) == 0 \
                and len(led) < 200:
            empty += 1
            continue
        commit, content = [], []
        for _t, _w, tg, b in recs:
            (commit if tg in COMMIT_TAGS else content).append(b)
        for axis, lines in (("commit", commit), ("content", content)):
            if len(lines) < min_lines:
                continue
            pairs.append(Pair(
                name=f"ledger/{day[5:]}#{axis}",
                source="\n".join(lines),
                summary=led,
                meta={"axis": axis, "n_src_lines": len(lines), "day": day},
                group=f"ledger/{day}",
            ))
    notes.append(f"{len(day_keys)} board days replayed through mesh-promises --balance "
                 f"(MESH_CHAT_LOG per-day slice); {empty} produced an empty ledger, dropped")
    notes.append("each day gives two pairs on ONE ledger: #commit (the axis the ledger keeps) "
                 "and #content (what it discards). Same donor group, so they cannot be each "
                 "other's foreign control")
    notes.append("PREDICTION the meter must reproduce: SIGNAL on #commit >> SIGNAL on #content. "
                 "A large SIGNAL/novel on #content means the meter is reading lexical overlap")
    notes.append("ledger roster is frozen (accounts.journal, 24 Jul): job/adint/wake route to "
                 "'unrouted' — quarantine, not absence of obligations "
                 "(task promises-roster-frozen-quarantines-live-windows, not ours)")
    notes.append(f"slices under {tmp}")
    return pairs, notes


SETS = {"handoff": load_handoff, "session": load_session, "ledger": load_ledger}


def load(name):
    if name not in SETS:
        raise SystemExit(f"unknown pair set '{name}'; have: {', '.join(SETS)}")
    return SETS[name]()


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SETS)):
        ps, ns = load(name)
        print(f"\n=== {name} ===")
        for n in ns:
            print(f"  note: {n}")
        for p in ps:
            print(f"  {p.name:<28} src {len(p.source):>6}c  sum {len(p.summary):>5}c  "
                  f"{len(p.source)/max(len(p.summary),1):>5.1f}x  {p.meta}")
