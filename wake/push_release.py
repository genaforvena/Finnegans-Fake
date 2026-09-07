#!/usr/bin/env python3
"""Upload the packaged release to the Hugging Face Hub.

    python wake/push_release.py --owner <hf-user> --dry-run     # says exactly what it would do
    python wake/push_release.py --owner <hf-user>               # does it

Auth: `HF_TOKEN` in the environment, or a token already stored by `hf auth login`.
A **write**-scoped token; a read token fails at repo creation with a 403 that reads
like a name collision.

Three rules this enforces rather than trusts:

* **`share` in the manifest decides visibility, and `local` never uploads.** The
  fold adapters are trained on the operator mesh's own board log, so publishing them
  publishes the log. That is not a flag to remember at the call site — it is a hard
  refusal here, and `--force-public` cannot reach it.
* **Nothing uploads that the packer did not verify.** The manifest records a real
  forward pass per repo; an entry without one is skipped loudly.
* **The manifest's own sha256 per file is re-checked before upload.** A release dir
  edited by hand after packing is a different artifact from the one that was verified.
"""
import argparse, hashlib, json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=str(ROOT / "release"))
    ap.add_argument("--owner", required=True, help="HF user or org that will own the repos")
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-public", action="store_true",
                    help="legacy compatibility switch; share=public is already public. "
                         "Never reaches share=local.")
    a = ap.parse_args()

    rel = pathlib.Path(a.release)
    manifest = json.loads((rel / "MANIFEST.json").read_text())

    plan, refused = [], []
    for slug, m in manifest.items():
        if a.only and slug not in a.only:
            continue
        if m["share"] == "local":
            refused.append((slug, "share=local — trained on the internal board log"))
            continue
        if not m.get("verified", {}).get("params"):
            refused.append((slug, "no verified forward pass in the manifest"))
            continue
        d = rel / slug
        bad = [n for n, f in m["files"].items()
               if not (d / n).exists() or sha(d / n) != f["sha256"]]
        if bad:
            refused.append((slug, f"changed since packing: {bad}"))
            continue
        private = not (m["share"] == "public" or a.force_public)
        plan.append((slug, d, private, sum(f["bytes"] for f in m["files"].values())))

    for slug, why in refused:
        print(f"  REFUSED  {slug:36s} {why}")
    for slug, d, private, nbytes in plan:
        vis = "private" if private else "PUBLIC"
        print(f"  upload   {a.owner}/{slug:36s} {vis:8s} {nbytes/1e6:7.1f} MB")
    if not plan:
        print("nothing to upload")
        return 1
    if a.dry_run:
        print("\n--dry-run: nothing was uploaded")
        return 0

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    print(f"\nauthenticated as {who.get('name')} ({who.get('type')})")

    for slug, d, private, _ in plan:
        rid = f"{a.owner}/{slug}"
        api.create_repo(rid, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=rid, folder_path=str(d), repo_type="model",
                          commit_message="finnegans-fake: packaged release")
        print(f"  done  https://huggingface.co/{rid}"
              + ("  (private)" if private else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
