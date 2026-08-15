#!/usr/bin/env bash
# Pull a model into a plain directory with curl.
#
# huggingface_hub's downloader hangs on this node rather than failing: the process
# stays alive, the retry loop never fires because no exception is ever raised, and
# the .incomplete file simply stops growing. Two blobs sat untouched for 70 minutes
# while the job looked healthy. --speed-limit/--speed-time turns a stall into a
# non-zero exit, which is the thing the loop needs in order to retry at all.
set -u
REPO="${1:?usage: fetch_model.sh <org/model> [destdir]}"
DEST="${2:-$HOME/models/$(basename "$REPO")}"
mkdir -p "$DEST"

files=$(curl -sS --max-time 60 "https://huggingface.co/api/models/$REPO" \
  | ~/.venv-ai/bin/python -c 'import json,sys
for f in json.load(sys.stdin).get("siblings", []):
    n = f["rfilename"]
    if n.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")):
        print(n)')
[ -n "$files" ] || { echo "no file list for $REPO"; exit 1; }

for f in $files; do
  out="$DEST/$f"; mkdir -p "$(dirname "$out")"
  for try in $(seq 1 40); do
    # -C - resumes; abort if under 20 KB/s for 30s so a stall becomes an exit code
    if curl -sS -L -C - --speed-limit 20000 --speed-time 30 --max-time 1800 \
         -o "$out" "https://huggingface.co/$REPO/resolve/main/$f"; then
      echo "ok   $f ($(stat -c%s "$out" 2>/dev/null) bytes)"; break
    fi
    sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
    echo "stall $f try=$try at ${sz} bytes"; sleep 3
  done
done
echo "DONE $REPO -> $DEST"
