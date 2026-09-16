#!/usr/bin/env python3
"""Validate and train the bounded multi-stream target.

The target currently contains deterministic proxy annotations.  A dry-run may
inspect those records, but a real run refuses them until an independent human
annotation pass has replaced the proxy labels.  This is intentionally a
separate entry point from ``train_fold.py`` so the stream prompt and its
provenance cannot be confused with the single-summary folds.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "multistream-target.jsonl"
DEFAULT_OUT = HERE / "multistream-adapter"
MIN_FREE_VRAM_MIB = 5600


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_target(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("target is empty")
    errors = []
    for record in records:
        atoms = record.get("atoms", [])
        streams = record.get("streams", [])
        if not record.get("id") or not record.get("source_sha256"):
            errors.append(f"{record.get('id', '<missing>')}: missing identity")
        if not atoms or len(streams) < 3:
            errors.append(f"{record.get('id', '<missing>')}: needs atoms and at least 3 streams")
        ids = {atom.get("id") for atom in atoms}
        if len(ids) != len(atoms) or None in ids:
            errors.append(f"{record.get('id', '<missing>')}: atom ids are not unique")
        for atom in atoms:
            if not atom.get("streams"):
                errors.append(f"{record['id']}/{atom.get('id')}: no stream membership")
        declared = {fact_id for stream in streams for fact_id in stream.get("fact_ids", [])}
        if not ids.issubset(declared):
            errors.append(f"{record['id']}: not every atom is declared by a stream")
    if errors:
        raise ValueError("invalid target:\n" + "\n".join(errors))
    return records


def gpu_prerequisite(min_free=MIN_FREE_VRAM_MIB):
    """Return a reproducible, non-mutating GPU gate observation."""
    smi = shutil.which("nvidia-smi")
    result = {"minimum_free_vram_mib": min_free, "observed": False, "status": "UNMEASURED"}
    if not smi:
        result["reason"] = "nvidia-smi unavailable"
        return result
    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=5,
        ).strip()
        rows = []
        for line in out.splitlines():
            free, util = (int(part.strip()) for part in line.split(",", 1))
            rows.append({"free_vram_mib": free, "utilization_percent": util})
        result.update({"observed": True, "readings": rows})
        if rows and max(row["free_vram_mib"] for row in rows) >= min_free:
            result["status"] = "PASS"
        else:
            result["status"] = "BLOCKED"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["reason"] = f"nvidia-smi probe failed: {exc}"
    return result


def annotation_status(records):
    independent = all(
        record.get("annotation", {}).get("independent_human_annotation") is True
        for record in records
    )
    return "PASS" if independent else "BLOCKED: independent human annotation required"


def dry_run(records, target, out, min_free):
    report = {
        "status": "DRY_RUN_PASS",
        "target": str(target),
        "target_sha256": sha256(target),
        "windows": len(records),
        "atoms": sum(len(record["atoms"]) for record in records),
        "streams": sum(len(record["streams"]) for record in records),
        "annotation_gate": annotation_status(records),
        "gpu_prerequisite": gpu_prerequisite(min_free),
        "training": "not launched",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=pathlib.Path, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT / "dry-run.json")
    parser.add_argument("--min-free-vram-mib", type=int, default=MIN_FREE_VRAM_MIB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base", help="reserved for the GPU training implementation")
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), help="reserved for training")
    args = parser.parse_args(argv)
    try:
        records = load_target(args.target)
        if args.dry_run:
            dry_run(records, args.target, args.out, args.min_free_vram_mib)
            return 0
        if annotation_status(records) != "PASS":
            raise RuntimeError("refusing training: independent human annotation is not present")
        raise RuntimeError("training implementation is intentionally gated pending annotated target")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"train_multistream: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
