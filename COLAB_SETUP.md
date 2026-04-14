# VidHalluc eval on RunPod (recommended) or Colab

End-to-end recipe for running the VidHalluc PLLaVA vs PruneVid comparison. **RunPod with a 3090 is what's known to work** — I've gone through this end-to-end and the fork has all the transformers-version-compat fixes baked in (see "What's fixed in this fork" at the bottom). Colab works too but the session limit makes the ~16 hour full sweep painful.

**Verified on**: RunPod Pytorch 2.2.x template, Python 3.12, RTX 3090, April 2026. Smoke test produces 14/20 baseline and 14/20 pruned on ach_binaryqa — real, reproducible numbers.

---

## Part 1 — RunPod walkthrough

### 1. Sign up + credit

1. https://runpod.io → sign up.
2. Billing → add **$10** credit (whole project runs ~$4).

### 2. Deploy a pod

1. Pods → **Deploy**.
2. **Cloud type**: Community Cloud.
3. **GPU**: **RTX 3090** (~$0.22/hr). 24GB VRAM is enough; don't overpay for a 4090.
4. **Template**: search "pytorch" → **RunPod Pytorch 2.2.0** (or 2.1.x if 2.2.x isn't available). Python 3.10 or 3.12 both work.
5. **Customize**:
   - Container Disk: 20 GB (default)
   - **Volume Disk: 60 GB** (model ~14GB + VidHalluc ~25GB + results + buffer)
   - Volume Mount Path: `/workspace` (default)
6. **Deploy On-Demand** (not Spot — Spot can be preempted mid-run).

Pod spins up in ~30s. Click **Connect** → **Connect to Jupyter Lab [8888]**, then File → New → Terminal.

### 3. Clone the fork

```bash
cd /workspace
git clone https://github.com/dd00697/PruneVid.git
cd PruneVid
```

(Replace `dd00697` with your username if you're someone else.)

### 4. Install minimal deps (skip `requirements.txt`)

**Do NOT use `pip install -r requirements.txt`.** It pins `torch==2.2.1+cu118`, `av==10.0.0`, a hardcoded Linux flash-attn wheel URL, and `apex` — all of which break on modern Python/CUDA. Use this minimal install instead:

```bash
pip install \
    "transformers==4.40.2" \
    "tokenizers==0.19.1" \
    "accelerate==0.26.1" \
    "peft==0.10.0" \
    "decord==0.6.0" \
    "einops" "safetensors" "sentencepiece>=0.1.99" \
    "moviepy==1.0.3" \
    "imageio" "imageio-ffmpeg" \
    "tqdm" "pillow" "huggingface_hub" \
    "av>=11,<14" \
    "protobuf>=4"
```

`transformers==4.40.2` is the load-bearing version — older (< 4.38) lacks `StaticCache`, older than 4.40 lacks `LlamaConfig.mlp_bias`, and newer 4.43+ has broken internals that conflict with PruneVid's LlamaModel subclass.

Verify:
```bash
python -c "from tasks.eval.model_utils import load_pllava; print('imports ok')"
```

Should print `imports ok`.

### 5. Download PLLaVA-7B (~14 GB, 3-5 min)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ermu2001/pllava-7b',
                  local_dir='MODELS/pllava-7b')
print('done')
"
```

This is a full merged checkpoint (not a LoRA-only adapter), but the weights are stored in PEFT-wrapped key-naming (`language_model.base_model.model.model.*`) with separate `base_layer` / `lora_A` / `lora_B` tensors for q_proj and v_proj. The eval script already loads it via the LoRA pathway (`--use_lora --lora_alpha 14 --weight_dir`) which is wired into `scripts/eval_vidhalluc.sh`.

### 6. Download + extract VidHalluc (~25 GB, 5-10 min)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='chaoyuli/VidHalluc',
                  repo_type='dataset',
                  local_dir='/workspace/VidHalluc')
print('done')
"
```

The dataset ships three zips. Extract them with this script — it handles the TSH nested-layout quirk automatically:

```bash
python - <<'PY'
import os, shutil, zipfile, glob
for z in sorted(glob.glob('/workspace/VidHalluc/data/*_videos.zip')):
    name = os.path.basename(z).replace('_videos.zip', '')
    dst = f'/workspace/VidHalluc/data/{name}'
    os.makedirs(dst, exist_ok=True)
    print(f'extracting {name}...')
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dst)

# Flatten: some zips contain `<SUBSET>/*.mp4`, some contain a nested `VidHalluc/data/<SUBSET>/*.mp4`
for name in ('ACH', 'STH', 'TSH'):
    dst = f'/workspace/VidHalluc/data/{name}'
    mp4_dir = None
    for dirpath, _, filenames in os.walk(dst):
        if any(f.endswith('.mp4') for f in filenames):
            mp4_dir = dirpath
            break
    if mp4_dir and mp4_dir != dst:
        for f in os.listdir(mp4_dir):
            if f.endswith('.mp4'):
                shutil.move(os.path.join(mp4_dir, f), os.path.join(dst, f))
    # Clean up empty intermediate dirs
    for dirpath, _, _ in os.walk(dst, topdown=False):
        if dirpath != dst:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    count = sum(1 for f in os.listdir(dst) if f.endswith('.mp4'))
    print(f'{name}: {count} mp4 files')
PY
```

Expected counts: ACH=3933, STH=445, TSH=600.

Verify the dataset flattens cleanly:

```bash
export VIDHALLUC_DATA_ROOT=/workspace/VidHalluc
python -c "
from tasks.eval.vidhalluc import VidHallucDataset
ds = VidHallucDataset(data_root='/workspace/VidHalluc',
                      subsets=['ach_binaryqa','ach_mcq','sth','tsh'],
                      num_segments=16)
print(ds)
print('missing:', ds.missing_videos)
"
```

Expected: `13871 examples`, `missing: {}`. If anything is missing, the extraction didn't land.

Free the zip space:
```bash
rm /workspace/VidHalluc/data/*_videos.zip
df -h /workspace
```

### 7. Smoke test (~1 min)

```bash
MODEL=MODELS/pllava-7b SKIP_FULL=1 bash scripts/eval_vidhalluc.sh
```

Runs baseline (`--disable-pruning`) + pruned on 20 ach_binaryqa examples, prints a comparison table. Expected output:

```
=== VidHalluc results (baseline) ===
Overall: 14/20 = 0.7000        (your number may vary by 0-2)
  ach_binaryqa      14/  20 = 0.7000   yes_rate=0.500

=== VidHalluc results (pruned) ===
Overall: 14/20 = 0.7000
  ach_binaryqa      14/  20 = 0.7000   yes_rate=0.600

metric                  baseline      pruned       delta
--------------------------------------------------------
overall                   0.7000      0.7000     +0.0000
ach_binaryqa              0.7000      0.7000     +0.0000   yes_rate: 0.5000 -> 0.6000 (+0.1000)
```

**If you see `0/20` on either run**, something's wrong — check `outputs/vidhalluc/smoke_baseline/predictions.jsonl` for empty `raw_output` fields; every example failing silently means the model is erroring inside generation. Paste the first traceback.

### 8. Full sweep (~16 hours, all 4 subsets × 2 configs × ~13.8k examples)

Launch inside `tmux` so SSH/Jupyter disconnects don't kill it:

```bash
# Install tmux if not present
which tmux || apt-get update && apt-get install -y tmux

tmux new -s eval
```

Your prompt changes — you're now inside tmux. Then:

```bash
cd /workspace/PruneVid
export VIDHALLUC_DATA_ROOT=/workspace/VidHalluc
MODEL=MODELS/pllava-7b bash scripts/eval_vidhalluc.sh 2>&1 | tee /workspace/full_run.log
```

Detach with **Ctrl+B**, release, then **D**. You're back at the normal shell but the run keeps going on the pod.

Verify it's still running: `tmux ls` (should show `eval: 1 windows`).

**Now close the Jupyter tab / laptop / wifi — anything. The pod runs independently.** Only thing that stops it is clicking "Stop" on the pod, or credit running out.

### 9. Check progress anytime during the run

Reconnect to the pod (Jupyter → Terminal) and either:

```bash
# Non-invasive peek
wc -l /workspace/PruneVid/outputs/vidhalluc/full_baseline/predictions.jsonl 2>/dev/null
wc -l /workspace/PruneVid/outputs/vidhalluc/full_pruned/predictions.jsonl 2>/dev/null
tail -20 /workspace/full_run.log
```

or reattach to the tmux session for live output:

```bash
tmux attach -t eval
# Detach again with Ctrl+B D
```

Expected timings on 3090 (~2 s/example):

| Phase | Examples | Approx time |
|---|---|---|
| smoke (20) × 2 | 40 | ~2 min |
| full_baseline `ach_binaryqa` | 8550 | ~5 h |
| full_baseline `ach_mcq` | 4276 | ~2.5 h |
| full_baseline `sth` | 445 | ~15 min |
| full_baseline `tsh` | 600 | ~20 min |
| full_pruned (all same) | 13871 | ~8 h |
| **Total** | | **~16 h** |

### 10. Grab the results

When the tmux session finishes (final line: `Done. Outputs under: outputs/vidhalluc`):

```bash
cd /workspace/PruneVid
cat outputs/vidhalluc/full_comparison.md      # quick look
zip -r /workspace/vidhalluc_full_results.zip outputs/vidhalluc
```

In Jupyter Lab's file browser (left panel) → navigate to `/workspace/` → right-click `vidhalluc_full_results.zip` → **Download**. Saves locally.

### 11. Stop the pod

Once the zip is on your laptop AND you've verified it unzips cleanly:

- RunPod UI → Pods → your pod → **Stop** (not Delete)
- Stopped pod costs ~$0.01/hr for the disk only. You can restart any time to rerun.
- **Delete** the pod only once you're sure you're done — this nukes the volume.

### Costs

| Phase | Hours | Cost |
|---|---|---|
| Setup + smoke | ~1 | $0.22 |
| Full sweep | ~16 | $3.52 |
| Buffer (accidental idle) | ~2 | $0.44 |
| **Total** | **~19** | **~$4** |

---

## Part 2 — Troubleshooting

### `ModuleNotFoundError` for some package at runtime
`pip install <name>` and rerun. The minimal install covers the hot path; some PruneVid code branches may want extras.

### `CUDA out of memory` on the 3090
Shouldn't happen with num_frames=16 on a 24GB 3090, but if it does, drop to 8 frames and rerun both configs with `NUM_FRAMES=8`:
```bash
NUM_FRAMES=8 MODEL=MODELS/pllava-7b bash scripts/eval_vidhalluc.sh
```
Frame count must match between baseline and pruned for a fair comparison.

### Baseline accuracy close to chance (0.5 on binary, 0.25 on MCQ)
Open `outputs/vidhalluc/smoke_baseline/predictions.jsonl`. If `raw_output` is empty or garbage, the model is erroring silently — the try/except in `pllava_answer` catches it. Paste the first traceback.

### `StaticCache` import error, `mlp_bias` not found, `attention_mask` unexpected kwarg
These are transformers version mismatches that were encountered during bring-up. The patches are already baked into this fork. If you see any of these, something downgraded transformers — force it back:
```bash
pip install --upgrade "transformers==4.40.2" "tokenizers==0.19.1"
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
```

### `No module named 'mmcv.runner'`
The fork's `tasks/eval/model_utils.py` already wraps the import in `try/except` so mmcv is optional. If you see this error, a stale `.pyc` is being used — clear cache:
```bash
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
```

### tmux session "died" after laptop sleep
It didn't — the pod runs independently. Reconnect (Jupyter → Terminal), `tmux attach -t eval`. If `tmux ls` shows no sessions, either the job finished or the pod crashed. Check `/workspace/full_run.log` for the last lines.

### Progress bar stuck on a single example for many minutes
A corrupted VidHalluc video is hanging decord. The fork already handles this: `VidHallucDataset.__getitem__` decodes videos in a child process with a 45-second hard timeout, and `pllava_eval_vidhalluc.py` catches the timeout and writes a skipped-placeholder record so the run continues. A skipped example appears in `predictions.jsonl` with `"skipped": true` and `"pred": null`.

If the run still appears to hang longer than ~60 seconds on one example, check the log:
```bash
tail -30 /workspace/full_run.log | grep -v "mmco\|h264"
grep -c "Skipping idx=" /workspace/full_run.log
```

Known bad file encountered during bring-up: `eXMF6Skt2To_clip_3.mp4` in the ACH subset. If you want to speed things up, move it out before the run:
```bash
mkdir -p /workspace/VidHalluc/_bad_videos
mv /workspace/VidHalluc/data/ACH/eXMF6Skt2To_clip_3.mp4 /workspace/VidHalluc/_bad_videos/ 2>/dev/null
```

The subprocess-based decode adds ~0.3-0.5 s per example overhead (spawning a child + loading decord), which translates to ~1-2 extra hours over 13.8k examples. That's the cost of reliability; it's not negotiable given the hang issue.

---

## Part 3 — Colab (only if you can't use RunPod)

Colab works for the smoke test and small subsets (`sth`, `tsh`) but is painful for the full `ach_binaryqa` sweep because of the session time limit (~12h on Pro, less on free).

Same install commands as RunPod section 4, except prefix with `!`:

```python
!nvidia-smi
!git clone https://github.com/dd00697/PruneVid.git /content/PruneVid
%cd /content/PruneVid
!pip install "transformers==4.40.2" "tokenizers==0.19.1" "accelerate==0.26.1" "peft==0.10.0" "decord==0.6.0" "einops" "safetensors" "sentencepiece>=0.1.99" "moviepy==1.0.3" "imageio" "imageio-ffmpeg" "tqdm" "pillow" "huggingface_hub" "av>=11,<14" "protobuf>=4"
```

Then the same model + dataset download + smoke test cells. For the full sweep, split by subset so each run finishes within the session limit:

```python
# Run sth + tsh first (small subsets, ~1 hour total)
!python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc --pretrained_model_name_or_path MODELS/pllava-7b --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b --subset sth --num_frames 16 --save_path outputs/vidhalluc/full_baseline_sth --disable-pruning
!python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc --pretrained_model_name_or_path MODELS/pllava-7b --use_lora --lora_alpha 14 --weight_dir MODELS/pllava-7b --subset sth --num_frames 16 --save_path outputs/vidhalluc/full_pruned_sth --selected_layer 10 --alpha 0.4 --tau 0.8 --temporal_segment_ratio 0.25 --cluster_ratio 0.5
# ...similarly for tsh, ach_mcq, ach_binaryqa
```

Save outputs to Drive between sessions so they survive disconnects.

---

## Part 4 — What's fixed in this fork (vs upstream `Visual-AI/PruneVid`)

Upstream pins `transformers==4.37.1` in requirements but the code actually needs ≥4.40 for several attributes. Running upstream out of the box on any modern Python gives you a cascade of import / attribute errors. This fork has the following patches baked in so it runs on `transformers==4.40.2`:

1. **`models/pllava/configuration_pllava.py`** — added `prune_enabled` config flag (default True) so `--disable-pruning` is a real bypass, not a `ratio=1.0` hack.

2. **`models/pllava/modeling_pllava.py`**:
   - Propagates `prune_enabled` to text_config.
   - Gates `merge_frames_dynamic` call on `prune_enabled`. When disabled, returns identity-shaped metadata so the rest of forward is untouched.
   - Disabled the `if flag: labels = torch.full_like(attention_mask, ...)` synthetic-label block that crashed pruned inference with `IndexError: shape of the mask [1, 933] does not match the shape of the indexed tensor [1, 395, 32064]` (pre-pruning mask vs post-pruning logits).

3. **`models/pllava/llama.py`**:
   - Gates `VTPWindowCache.cache(...)` on `prune_enabled` for true disable.
   - `bias=config.mlp_bias` → `bias=getattr(config, 'mlp_bias', False)` to tolerate older config.json that predates the field.
   - Dropped `attention_mask=attention_mask` kwarg from the `BaseModelOutputWithPast(...)` return in `LlamaModelVTP.forward` (transformers 4.40 doesn't accept it).
   - Dropped `attention_mask=outputs.attention_mask` kwarg from the `CausalLMOutputWithPast(...)` return in `LlamaForCausalLMVTP.forward`.

4. **`tasks/eval/model_utils.py`**:
   - `load_pllava` takes a `prune_enabled=True` kwarg, passes it through to `PllavaConfig.from_pretrained`.
   - `from mmcv.runner import load_checkpoint` wrapped in `try/except` with a stub fallback, so mmcv is optional (the only consumer is the optical-flow model which VidHalluc doesn't use).

5. **`tasks/eval/vidhalluc/`** — new module:
   - `__init__.py`: `VidHallucDataset(EvalDataset)` flattening all 4 subsets, robust video-path resolver (handles TSH's nested unzip layout), `build_prompt` / `parse_answer` / `is_correct` helpers. `__getitem__` decodes videos in an isolated subprocess with a 45-second hard-kill timeout — handles VidHalluc's occasional corrupt h264 streams that freeze decord indefinitely inside C code (signal.alarm is not enough).
   - `pllava_eval_vidhalluc.py`: single-process eval script with **hard-coded greedy decoding** (`do_sample=False, num_beams=1, temperature=1.0, top_p=1.0, max_new_tokens=32`) so pruned vs baseline cannot drift on decoding params. Catches `TimeoutError`/`RuntimeError` from the video decoder and writes a skipped-placeholder record (`"skipped": true`) so the output indices stay aligned. Writes `predictions.jsonl` + `summary.json` + `run_meta.json`.
   - `compare.py`: prints baseline / pruned / delta tables with per-subset yes-rate deltas.

6. **`scripts/eval_vidhalluc.sh`** — bash runner. Smoke + full sweep, tmux-ready. Already wired with `--use_lora --lora_alpha 14 --weight_dir "$MODEL"` because the `ermu2001/pllava-7b` checkpoint is a PEFT-wrapped half-merged LoRA.

None of these changes affect the existing MVBench / VideoMME / EgoSchema / VCGBench / Recaption eval scripts — the `prune_enabled` flag defaults to `True`, so their behavior is unchanged.

---

## Part 5 — Key implementation notes

- **Model path**: PLLaVA-7B (`ermu2001/pllava-7b` on HuggingFace). LLaVA-OneVision is an upstream TODO; the stub loader imports symbols not present in this repo.
- **Disable-pruning**: a real config flag that short-circuits `merge_frames_dynamic` and `VTPWindowCache.cache` — not a ratio=1.0 hack. Ratios at 1.0 still run the clustering code and reorder tokens.
- **Decoding settings**: hard-coded in `pllava_eval_vidhalluc.py` (`FIXED_DECODING`) — greedy, `max_new_tokens=32`. Not CLI-configurable.
- **`--data-root` auto-detect** (in priority order): `VIDHALLUC_DATA_ROOT` env var → `<PruneVid>/../data/VidHalluc` → `/content/VidHalluc` (Colab) → `/content/drive/MyDrive/VidHalluc` → `<PruneVid>/VidHalluc` → legacy nested layout.
- **Conversation template**: `conv_plain_v1` (empty system prompt, USER/ASSISTANT roles). Not `eval_mvbench` — its "select the best option" system prompt biases yes/no answers.
