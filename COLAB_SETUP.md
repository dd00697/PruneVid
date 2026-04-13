# VidHalluc evaluation on Colab

End-to-end recipe for running the VidHalluc PLLaVA vs PruneVid comparison in a Colab notebook. Requires a T4 (minimum) or A100 GPU runtime.

> **Don't use `requirements.txt` / `requirements.torch.txt` / `requirements.no_torch.txt`.** They pin `torch==2.2.1+cu118`, `av==10.0.0`, `apex`, and a hardcoded Linux flash-attn wheel URL — all of which fail on modern Colab. Use the minimal install below instead; it only pulls the packages the VidHalluc eval actually imports.

## 1. Clone the fork

```python
!nvidia-smi
!git clone https://github.com/<your-user>/PruneVid.git /content/PruneVid
%cd /content/PruneVid
```

## 2. Install minimal dependencies (skip the pinned requirements files)

```python
# Colab already has a matching torch+CUDA. Do not reinstall torch.
# Install only what the PLLaVA + VidHalluc eval actually needs.
!pip install -q \
    "transformers==4.37.1" \
    "tokenizers==0.15.2" \
    "accelerate==0.26.1" \
    "peft==0.10.0" \
    "decord==0.6.0" \
    "einops" \
    "safetensors" \
    "sentencepiece" \
    "moviepy==1.0.3" \
    "imageio==2.27.0" "imageio-ffmpeg" \
    "tqdm" "pillow" \
    "huggingface_hub" \
    "av>=11,<14"                 # skip the broken av==10.0.0 pin

# mmcv — let openmim pick a wheel that matches Colab's torch+cuda
!pip install -q -U openmim
!mim install -q "mmcv==2.1.0" || !mim install -q mmcv-full
```

If you later hit `ModuleNotFoundError` for any package at runtime, just `!pip install <name>` and continue — don't try to install the full requirements file.

**`transformers==4.37.1` is load-bearing** — PruneVid's `models/pllava/` subclasses internals (`LlamaModel`, cache classes) from that exact minor version. Newer transformers versions will crash on import.

## 3. Download PLLaVA-7B

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ermu2001/pllava-7b",
    local_dir="/content/PruneVid/MODELS/pllava-7b",
    local_dir_use_symlinks=False,
)
```

If the model is gated, run `from huggingface_hub import login; login()` first and paste a token.

If your PLLaVA checkpoint is the base+LoRA split (not a merged one), you'll need to add `--use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b` to the invocation — edit `COMMON_ARGS` in `scripts/eval_vidhalluc.sh`.

## 4. Download VidHalluc and extract videos

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="chaoyuli/VidHalluc",
    repo_type="dataset",
    local_dir="/content/VidHalluc",
    local_dir_use_symlinks=False,
)

# Extract the video zips into the layout the eval script expects
import os, zipfile, glob
for z in glob.glob('/content/VidHalluc/*_videos.zip'):
    name = os.path.basename(z).replace('_videos.zip', '')   # ACH / STH / TSH
    dst = f'/content/VidHalluc/data/{name}'
    os.makedirs(dst, exist_ok=True)
    print('extracting', z, '->', dst)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dst)

# Sanity check
!ls /content/VidHalluc/ach_binaryqa.json /content/VidHalluc/ach_mcq.json /content/VidHalluc/sth.json /content/VidHalluc/tsh.json
!ls /content/VidHalluc/data/ACH | head -3
!ls /content/VidHalluc/data/STH | head -3
!ls /content/VidHalluc/data/TSH | head -3
```

The eval script's `--data-root` resolver auto-detects `/content/VidHalluc`, so you don't need to pass it.

If the TSH zip extracts into a nested layout (`/content/VidHalluc/data/TSH/VidHalluc/data/TSH/...`), the resolver also handles `data/VidHalluc/data/TSH/` as a fallback — don't worry about reshuffling.

## 5. Smoke test (20 ach_binaryqa examples, ~2 min on T4)

```python
%cd /content/PruneVid
%env MODEL=MODELS/pllava-7b
%env SKIP_FULL=1
!bash scripts/eval_vidhalluc.sh
```

This runs the baseline (`--disable-pruning`) and the pruned config on the same 20 examples, then prints a comparison table with baseline acc / pruned acc / delta / yes-rate delta. Outputs go to `outputs/vidhalluc/smoke_{baseline,pruned}/` with `predictions.jsonl`, `summary.json`, and `run_meta.json` in each. Combined comparison at `outputs/vidhalluc/smoke_comparison.md`.

## 6. Full sweep (all 4 subsets, ~13.8k examples)

```python
%env SKIP_FULL=0
!bash scripts/eval_vidhalluc.sh
```

Hours on T4, much less on A100. Outputs at `outputs/vidhalluc/full_{baseline,pruned}/` plus `full_comparison.md`.

## Manual invocation (bypass the shell script)

```python
!python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --subset ach_binaryqa --max_samples 20 --num_frames 16 \
    --save_path outputs/vidhalluc/smoke_baseline --disable-pruning

!python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    --pretrained_model_name_or_path MODELS/pllava-7b \
    --subset ach_binaryqa --max_samples 20 --num_frames 16 \
    --selected_layer 10 --alpha 0.4 --tau 0.8 \
    --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
    --save_path outputs/vidhalluc/smoke_pruned

!python -m tasks.eval.vidhalluc.compare \
    --baseline outputs/vidhalluc/smoke_baseline/summary.json \
    --pruned   outputs/vidhalluc/smoke_pruned/summary.json
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'X'`** — just `!pip install X` and rerun. The minimal install above covers the hot path; some code branches may pull in other deps we didn't preinstall.

**`transformers` version errors** (`cannot import name 'LlamaModel' from 'transformers...'` or similar) — you installed a newer transformers. Pin back: `!pip install transformers==4.37.1 tokenizers==0.15.2 --force-reinstall`.

**CUDA OOM on T4** — drop `--num_frames 16` to `--num_frames 8`. Must match between baseline and pruned runs (set `NUM_FRAMES=8` env var before running the shell script).

**PLLaVA checkpoint is base+LoRA** — if `load_pllava` errors on weight shape mismatch, your checkpoint is the LoRA-adapter style. Edit `scripts/eval_vidhalluc.sh` and add `--use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b` to `COMMON_ARGS`.

**Baseline accuracy ≈ 0.5 on smoke test** — that's chance for yes/no. Check `outputs/vidhalluc/smoke_baseline/predictions.jsonl` — look at `raw_output`. If outputs are empty or garbage, the model didn't load correctly. If outputs are coherent but always "Yes" or always "No", the prompt template isn't landing — try `--conv_mode eval_mvbench` instead of `plain`.

**Flash-attn complaints** — PLLaVA's config requests `_attn_implementation = "sdpa"` (in `modeling_pllava.py:586`), which does NOT need flash-attn. If transformers logs a flash-attn warning, ignore it.

## Notes on the implementation

- Decoding is hard-coded in `tasks/eval/vidhalluc/pllava_eval_vidhalluc.py` (`FIXED_DECODING` dict: greedy, `max_new_tokens=32`) so baseline vs pruned cannot drift on sampling params.
- `--disable-pruning` flips a real config flag (`prune_enabled=False`) that short-circuits `merge_frames_dynamic` in `modeling_pllava.py` and the `VTPWindowCache` call in `llama.py` — not a ratio=1 hack.
- `--data-root` auto-detect priority: `VIDHALLUC_DATA_ROOT` env var → `<PruneVid>/../data/VidHalluc` → `/content/VidHalluc` → `/content/drive/MyDrive/VidHalluc` → `<PruneVid>/VidHalluc` → legacy nested layout.
