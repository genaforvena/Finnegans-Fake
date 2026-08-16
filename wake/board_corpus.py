#!/usr/bin/env python3
"""The full board history, and windows cut from it for distillation.

chat.log holds 3000 lines (~8 days) because it is evicted. The hourly copies in
~/.mesh/board-snapshots/ each hold their own 3000-line view, so their union
reaches back three weeks further. Deduped by (timestamp, window) that is 5298
entries over ~20 days instead of 3000 over 8 — the difference between a corpus
that can train something and one that cannot.

The eight windows already measured (pairs.windows(), cut from chat.log's tail)
are the TEST set and are excluded here by time span, so nothing the student is
trained on overlaps what it is scored on.
"""
import pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MESH = pathlib.Path.home() / ".mesh"
LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(\S+?)@(\S+)\s+::\s+(.*)$")


def union_board():
    """[(ts, window, body)] — every board entry we still have, deduped, in order."""
    seen = {}
    files = sorted((MESH / "board-snapshots").glob("*.log")) + [MESH / "chat.log"]
    for f in files:
        if not f.exists():
            continue
        cur = None
        for raw in f.read_text(errors="replace").splitlines():
            m = LINE.match(raw)
            if m:
                cur = (m.group(1), m.group(2))
                seen.setdefault(cur, [m.group(1), m.group(2), m.group(4)])
            elif cur and cur in seen:
                seen[cur][2] += "\n" + raw
    return [tuple(v) for v in sorted(seen.values(), key=lambda r: (r[0], r[1]))]


def test_spans():
    """(first_ts, last_ts) of each already-measured window, to keep training off it."""
    import pairs as pairmod
    board = pairmod.read_board()[-2500:]
    n, step = pairmod.WINDOWS_N, len(pairmod.read_board()[-2500:]) // pairmod.WINDOWS_N
    spans = []
    for k in range(n):
        i, chars, lines = k * step, 0, 0
        first = board[k * step][0].isoformat()
        last = first
        while i < len(board) and chars < pairmod.WINDOW_CHARS and lines < 22:
            ts, win, _tag, body = board[i]
            chars += len(f"{win}: {body}") + 1
            last = ts.isoformat()
            lines += 1
            i += 1
        spans.append((first[:19] + "Z", last[:19] + "Z"))
    return spans


def cut_windows(target_chars=4200, max_lines=22, exclude=True):
    """Contiguous, disjoint windows over the union board, skipping test spans."""
    rows = union_board()
    bad = test_spans() if exclude else []

    def blocked(ts):
        return any(lo <= ts <= hi for lo, hi in bad)

    out, i, k = [], 0, 0
    while i < len(rows):
        if blocked(rows[i][0]):
            i += 1
            continue
        buf, chars, j = [], 0, i
        while j < len(rows) and chars < target_chars and len(buf) < max_lines:
            if blocked(rows[j][0]):
                break
            ts, win, body = rows[j]
            line = f"{win}: {body}"
            buf.append(line)
            chars += len(line) + 1
            j += 1
        if len(buf) >= 4 and chars >= 1800:
            out.append({"id": f"t{k:03d}", "text": "\n".join(buf),
                        "n_lines": len(buf), "start": rows[i][0]})
            k += 1
        i = max(j, i + 1)
    return out


if __name__ == "__main__":
    ws = cut_windows()
    rows = union_board()
    print(f"union board: {len(rows)} entries, {rows[0][0]} .. {rows[-1][0]}")
    print(f"test spans held out: {len(test_spans())}")
    print(f"windows cut: {len(ws)}")
    tot = sum(len(w['text']) for w in ws)
    print(f"chars {tot/1e6:.2f}M, mean {tot//len(ws)} per window")
