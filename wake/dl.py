from huggingface_hub import snapshot_download
for m in ['Qwen/Qwen3.5-0.8B','Qwen/Qwen3.5-0.8B-Base','Qwen/Qwen3.5-2B']:
    try:
        p = snapshot_download(m, allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
        print('ok', m, p, flush=True)
    except Exception as e:
        print('FAIL', m, repr(e), flush=True)
