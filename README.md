# CoconutBLT — a short intro

CoconutBLT is a small research prototype that mixes ideas from Coconut (continuous latent reasoning)
and BLT (byte-level patching). It's designed for experiments and learning — not a polished production project.

Who this is for
- Researchers and engineers exploring hybrid transformer ideas.
- Anyone who wants a compact, runnable starting point (toy training, generation, and tooling).

Quick start (run a tiny test)

1. Install the Python requirements:

```bash
pip install -r requirements.txt
```

2. Run a short toy training (CPU friendly):

```bash
PYTHONPATH=src python3 scripts/train_toy.py --steps 20 --device cpu
```

3. Sample from a saved checkpoint:

```bash
PYTHONPATH=src python3 scripts/generate.py --ckpt checkpoints/final.pt --prompt "Hello" --max_new_tokens 64
```

Where to look next
- `docs/USAGE.md` — more detailed examples and scaling notes
- `ARCHITECTURE.md` — design and parameter budget for a ~1B model
- `src/coconutblt` — the minimal PyTorch implementation and patcher code
- `scripts/` — training, generation, and export helpers

Licensing notes
- Some ideas are inspired by Coconut (MIT license).
- Parts inspired by BLT follow CC-BY-NC-4.0 (non-commercial) — check `blt/README.md` and the license files before reusing BLT-derived code.

That's it — the repo provides a simple playground for research. See `docs/USAGE.md` if you want to run larger experiments.