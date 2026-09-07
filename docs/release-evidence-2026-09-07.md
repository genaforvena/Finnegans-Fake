# Release evidence — 2026-09-07

Fresh verification from the current working tree:

```text
python3 wake/attest_scores.py
attested 10/10 scored sets

python3 wake/paired.py --test
[test] 7/7 passed

/home/mesh-home/.venv-ai/bin/python wake/pack_release.py
5 packaged -> release/  (MANIFEST.json written)
finnegans-fake-char257       public  10,942,848 params
english-char257              public  10,942,848 params
finnegans-fake-bpe4096       public  12,318,720 params
finnegans-fake-lora...       public  12,779,520 params
finnegans-fake-folds-lora    local   6,389,760 params

python3 wake/reflex.py --test
smoke-test: ok (9 ids)

/home/mesh-home/.venv-ai/bin/python wake/reflex.py --id wake-char257 --chars 80 ...
🎲 wake-char257 ↩︎ y. Maisty minute mutter ...

/home/mesh-home/.venv-ai/bin/python wake/push_release.py --owner genaforvena --dry-run
REFUSED finnegans-fake-folds-lora share=local
upload ... four public artifacts
--dry-run: nothing was uploaded
```

`git diff --check` and Python byte-compilation of the changed tools also passed. The repository
does not claim an external Hugging Face upload; that outward action remains operator-owned.

The Base-vs-Instruct raw-checkpoint result is independently recorded in
`wake/base-instruct-results.json` and its method/conclusion in `docs/base-vs-instruct-2026-09-07.md`.
The attempted matched Instruct Wake-LoRA run is not counted: it stopped at first validation with
CUDA OOM because the shared GPU was occupied by existing processes.

