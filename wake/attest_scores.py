#!/usr/bin/env python3
"""Create a provenance index for the historical condent score artifacts.

The scored JSON files predate runtime provenance stamping. This tool does not
rewrite them or pretend otherwise: it records the score-file hash, the commit
that introduced/last changed it, the rung training metadata, and whether the
producer script can be recovered from that commit. Consumers can quote only
entries whose status is ``attested``.
"""
import argparse, hashlib, json, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "score-provenance.json"))
    a = ap.parse_args()
    rung = json.loads((HERE / "rung-provenance.json").read_text())["variants"]
    rows = {}
    for score in sorted(HERE.glob("recs-*.json")):
        name = score.stem.removeprefix("recs-")
        data = json.loads(score.read_text())
        try:
            commit = git("log", "-1", "--format=%H", "--", str(score.relative_to(ROOT)))
            producer = subprocess.check_output(
                ["git", "show", f"{commit}:wake/condent.py"], cwd=ROOT
            )
            producer_hash = sha_bytes(producer)
            status = "attested"
        except (subprocess.CalledProcessError, OSError):
            commit, producer_hash, status = None, None, "unmeasured"
        rows[name] = {
            "score_file": str(score.relative_to(ROOT)),
            "score_sha256": sha_file(score),
            "score_commit": commit,
            "producer": "wake/condent.py",
            "producer_sha256_at_score_commit": producer_hash,
            "pairs": data.get("pairs"),
            "model": data.get("model"),
            "budget": data.get("budget"),
            "n_records": len(data.get("records", [])),
            "training": rung.get(name),
            "status": status,
        }
    out = {"schema": 1, "generated_from": "wake/attest_scores.py", "variants": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    bad = [k for k, v in rows.items() if v["status"] != "attested"]
    print(f"attested {len(rows) - len(bad)}/{len(rows)} scored sets -> {a.out}")
    if bad:
        print("unmeasured: " + ", ".join(bad))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
