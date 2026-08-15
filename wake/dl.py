"""Download the base models. Retries: the node's egress stalls mid-blob rather
than erroring, so a one-shot snapshot_download can sit at 150MB forever."""
import time
from huggingface_hub import snapshot_download

MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-2B"]
for m in MODELS:
    for attempt in range(1, 21):
        try:
            p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                                  max_workers=2, etag_timeout=30)
            print("ok", m, p, flush=True)
            break
        except Exception as e:
            print(f"retry {attempt} {m}: {type(e).__name__} {e}"[:200], flush=True)
            time.sleep(5)
    else:
        print("FAIL", m, flush=True)
