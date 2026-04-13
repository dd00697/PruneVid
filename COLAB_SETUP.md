# VidHalluc evaluation on Colab

End-to-end recipe for running the VidHalluc PLLaVA vs PruneVid comparison in a Colab notebook. Requires a T4 (minimum) or A100 GPU runtime.

## 1. Clone the fork and install deps

```python
!nvidia-smi
!git clone https://github.com/<your-user>/PruneVid.git /content/PruneVid
%cd /content/PruneVid
```

PruneVid pins `torch==2.2.1+cu118` and a Linux flash-attn wheel. On Colab the `requirements.txt` path is the fast one; fall back to the split install only if the flash-attn wheel URL 404s.

```python
# Preferred: full install
!pip install -r requirements.txt

# If the flash-attn wheel line fails, install in two steps and let pip
# pick a wheel that matches the current Colab CUDA:
# !pip install -r requirements.torch.txt
# !pip install -r requirements.no_torch.txt
# !pip install flash-attn --no-build-isolation
```

If you hit an `mmcv-full` build error, try `pip install "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.2/index.html` first.

## 2. Download PLLaVA-7B

```python
!pip install -q "huggingface_hub[cli]"
# If the model is gated you'll need: from huggingface_hub import login; login()
!huggingface-cli download ermu2001/pllava-7b --local-dir MODELS/pllava-7b --local-dir-use-symlinks False
```

`scripts/eval.sh` references `--use_lora --lora_alpha 14 --weight_dir <lora>`. If your PLLaVA checkpoint is the base+LoRA split, add those flags to the script below. If it's a merged checkpoint, the defaults are fine.

## 3. Get VidHalluc data into /content/VidHalluc

Options (pick one):

**A. zip upload** — upload `VidHalluc.zip` via the Colab file browser, then:
```python
!mkdir -p /content/VidHalluc
!unzip -q /content/VidHalluc.zip -d /content/VidHalluc
```

**B. Google Drive mount** — if you already have the data on Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# The eval script will auto-detect /content/drive/MyDrive/VidHalluc if
# ach_binaryqa.json is at its top level. Otherwise pass --data-root.
```

**C. huggingface / gdown** — if there's a public download URL, use `gdown` or `huggingface-cli` accordingly.

The VidHalluc root must directly contain `ach_binaryqa.json`, `ach_mcq.json`, `sth.json`, `tsh.json`, and a `data/` folder with `ACH/`, `STH/`, `TSH/` subdirs full of `.mp4` files. The eval script accepts either `data/TSH/` or `data/VidHalluc/data/TSH/` for the TSH unzip-layout quirk.

## 4. Smoke test (20 ach_binaryqa examples, ~2 min on T4)

```python
%env MODEL=MODELS/pllava-7b
%env SKIP_FULL=1
!bash scripts/eval_vidhalluc.sh
```

This runs the baseline (`--disable-pruning`) and pruned configs on the same 20 examples, then prints a comparison. Outputs land in `outputs/vidhalluc/smoke_{baseline,pruned}/` with `predictions.jsonl`, `summary.json`, and `run_meta.json` each. The combined comparison is at `outputs/vidhalluc/smoke_comparison.md`.

## 5. Full sweep (all 4 subsets)

```python
%env SKIP_FULL=0
!bash scripts/eval_vidhalluc.sh
```

~13.8k examples across ach_binaryqa, ach_mcq, sth, tsh. On a T4 this takes hours; on an A100 much less. Outputs at `outputs/vidhalluc/full_{baseline,pruned}/` + `outputs/vidhalluc/full_comparison.md`.

## Manual invocation (if you want to skip the shell script)

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

## Notes

- Decoding is hard-coded in `pllava_eval_vidhalluc.py` (greedy, `max_new_tokens=32`) so baseline vs pruned can't drift on sampling params.
- `--disable-pruning` flips a real config flag (`prune_enabled=False`) that short-circuits `merge_frames_dynamic` and the `VTPWindowCache` call — not a ratio=1 hack.
- If the data-root auto-detect picks the wrong location, pass `--data-root /abs/path/to/VidHalluc` or set env var `VIDHALLUC_DATA_ROOT`.
