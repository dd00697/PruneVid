#!/usr/bin/env bash
# Run the PLLaVA / PruneVid VidHalluc comparison: pruned vs true baseline.
# Smoke test (20 examples of ach_binaryqa) runs first; then the full sweep.
#
# Override these env vars as needed:
#   MODEL         : HF repo or local path to the PLLaVA checkpoint
#   NUM_FRAMES    : frame count (must match between runs)
#   OUT_ROOT      : where to write outputs
#   DATA_ROOT     : VidHalluc data root (defaults to the auto-resolved path)
#   SKIP_FULL     : set to 1 to stop after smoke test

set -euo pipefail

# cd into the PruneVid repo root so relative imports work
cd "$(dirname "$0")/.."

MODEL="${MODEL:-MODELS/pllava-7b}"
NUM_FRAMES="${NUM_FRAMES:-16}"
OUT_ROOT="${OUT_ROOT:-outputs/vidhalluc}"
SKIP_FULL="${SKIP_FULL:-0}"

COMMON_ARGS=(
    --pretrained_model_name_or_path "$MODEL"
    --use_lora
    --lora_alpha 14
    --weight_dir "$MODEL"
    --num_frames "$NUM_FRAMES"
    --pooling_shape "16-12-12"
)

PRUNING_ARGS=(
    --selected_layer 10
    --alpha 0.4
    --tau 0.8
    --temporal_segment_ratio 0.25
    --cluster_ratio 0.5
    --softmax 1.0
    --head 8
)

if [[ -n "${DATA_ROOT:-}" ]]; then
    COMMON_ARGS+=(--data-root "$DATA_ROOT")
fi

echo "=== Smoke test: ach_binaryqa, 20 examples ==="

python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    "${COMMON_ARGS[@]}" \
    --subset ach_binaryqa \
    --max_samples 20 \
    --save_path "$OUT_ROOT/smoke_baseline" \
    --disable-pruning

python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    "${COMMON_ARGS[@]}" \
    "${PRUNING_ARGS[@]}" \
    --subset ach_binaryqa \
    --max_samples 20 \
    --save_path "$OUT_ROOT/smoke_pruned"

python -m tasks.eval.vidhalluc.compare \
    --baseline "$OUT_ROOT/smoke_baseline/summary.json" \
    --pruned   "$OUT_ROOT/smoke_pruned/summary.json" \
    --out      "$OUT_ROOT/smoke_comparison.md"

if [[ "$SKIP_FULL" == "1" ]]; then
    echo "SKIP_FULL=1, stopping after smoke test. Results in $OUT_ROOT"
    exit 0
fi

echo "=== Full run: all subsets ==="

python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    "${COMMON_ARGS[@]}" \
    --subset all \
    --save_path "$OUT_ROOT/full_baseline" \
    --disable-pruning

python -m tasks.eval.vidhalluc.pllava_eval_vidhalluc \
    "${COMMON_ARGS[@]}" \
    "${PRUNING_ARGS[@]}" \
    --subset all \
    --save_path "$OUT_ROOT/full_pruned"

python -m tasks.eval.vidhalluc.compare \
    --baseline "$OUT_ROOT/full_baseline/summary.json" \
    --pruned   "$OUT_ROOT/full_pruned/summary.json" \
    --out      "$OUT_ROOT/full_comparison.md"

echo "Done. Outputs under: $OUT_ROOT"
