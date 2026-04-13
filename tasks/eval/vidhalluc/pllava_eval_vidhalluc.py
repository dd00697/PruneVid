"""Run PLLaVA / PruneVid evaluation on VidHalluc.

Single-process eval designed to be invoked twice with identical args except
``--disable-pruning`` so the pruned and baseline numbers differ only in
whether the PruneVid visual-token pruning is active.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from tasks.eval.eval_utils import conv_templates
from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.vidhalluc import (
    SUBSETS,
    VidHallucDataset,
    build_prompt,
    is_correct,
    parse_answer,
)


logging.basicConfig()
logger = logging.getLogger("vidhalluc_eval")
logger.setLevel(logging.INFO)


# Fixed decoding settings. Intentionally NOT CLI-configurable so the pruned
# and baseline runs cannot drift on decoding. Only pruning should differ.
FIXED_DECODING = dict(
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_p=1.0,
    repetition_penalty=1.0,
    length_penalty=1.0,
    max_new_tokens=32,
)


def _default_data_root() -> str:
    # Colab-friendly candidate list. Pick the first one that contains
    # ach_binaryqa.json at its top level.
    prunevid = Path(__file__).resolve().parents[3]
    env = os.environ.get("VIDHALLUC_DATA_ROOT")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    candidates += [
        prunevid.parent / "data" / "VidHalluc",
        Path("/content/VidHalluc"),
        Path("/content/drive/MyDrive/VidHalluc"),
        prunevid / "VidHalluc",
        prunevid.parent.parent,
    ]
    for c in candidates:
        try:
            if (c / "ach_binaryqa.json").exists():
                return str(c.resolve())
        except OSError:
            continue
    # Fall back to the most likely sibling layout so argparse --help and
    # the dataset's error message surface a concrete path.
    return str((prunevid.parent / "data" / "VidHalluc").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=_default_data_root(),
                        help="VidHalluc root containing ach_*.json, sth.json, tsh.json, and data/{ACH,STH,TSH}/")
    parser.add_argument("--subset", type=str, default="all",
                        choices=list(SUBSETS) + ["all"])
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--weight_dir", type=str, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_alpha", type=int, default=14)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--pooling_shape", type=str, default="16-12-12",
                        help="hyphen-separated T-H-W, e.g. 16-12-12")
    # Pruning hyperparameters (defaults match scripts/eval.sh).
    parser.add_argument("--selected_layer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--temporal_segment_ratio", type=float, default=0.25)
    parser.add_argument("--cluster_ratio", type=float, default=0.5)
    parser.add_argument("--softmax", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--disable-pruning", dest="disable_pruning",
                        action="store_true",
                        help="Bypass PruneVid entirely (true baseline).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="plain")
    parser.add_argument("--print_res", action="store_true")
    return parser.parse_args()


def _parse_pooling_shape(s: str):
    parts = [int(x) for x in s.split("-")]
    if len(parts) != 3:
        raise ValueError(f"--pooling_shape must be T-H-W, got: {s}")
    return tuple(parts)


def _resolve_subsets(subset: str):
    if subset == "all":
        return list(SUBSETS)
    return [subset]


def _yes_rate(entries):
    if not entries:
        return None
    n = 0
    for e in entries:
        if e["pred"] and str(e["pred"]).strip().lower() == "yes":
            n += 1
    return n / len(entries)


def _summarize(records):
    totals = {"n": 0, "correct": 0}
    per_subset: Dict[str, Dict[str, Any]] = {}
    by_subset: Dict[str, list] = {}
    for r in records:
        by_subset.setdefault(r["subset"], []).append(r)
    for subset, items in by_subset.items():
        n = len(items)
        correct = sum(1 for it in items if it["correct"])
        entry = {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}
        if subset in ("ach_binaryqa", "sth"):
            entry["yes_rate"] = _yes_rate(items)
        per_subset[subset] = entry
        totals["n"] += n
        totals["correct"] += correct
    totals["accuracy"] = totals["correct"] / totals["n"] if totals["n"] else 0.0
    return {"totals": totals, "per_subset": per_subset}


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    pooling_shape = _parse_pooling_shape(args.pooling_shape)
    subsets = _resolve_subsets(args.subset)

    logger.info("Loading PLLaVA model from %s", args.pretrained_model_name_or_path)
    logger.info("Pruning enabled: %s", not args.disable_pruning)

    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha,
        pooling_shape=pooling_shape,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        head=args.head,
        softmax=args.softmax,
        tau=args.tau,
        cluster_ratio=args.cluster_ratio,
        temporal_segment_ratio=args.temporal_segment_ratio,
        prune_enabled=not args.disable_pruning,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logger.info("Building VidHalluc dataset from %s (subsets=%s)", args.data_root, subsets)
    dataset = VidHallucDataset(
        data_root=args.data_root,
        subsets=subsets,
        num_segments=args.num_frames,
        max_samples=args.max_samples,
    )
    dataset.set_rank_and_world_size(0, 1)
    logger.info("Dataset: %s", str(dataset))

    pred_path = os.path.join(args.save_path, "predictions.jsonl")
    records = []
    fout = open(pred_path, "w", encoding="utf-8")
    try:
        for idx in tqdm(range(len(dataset)), desc="VidHalluc"):
            ex = dataset[idx]
            user_query = build_prompt(ex)

            conv = conv_templates[args.conv_mode].copy()
            conv.user_query(user_query, pre_query_prompt=None, post_query_prompt=None, is_mm=True)

            try:
                raw, _ = pllava_answer(
                    conv=conv,
                    model=model,
                    processor=processor,
                    img_list=ex["images"],
                    print_res=args.print_res,
                    **FIXED_DECODING,
                )
            except Exception as exc:
                logger.exception("Generation failed for %s: %s", ex["sample_id"], exc)
                raw = ""

            pred = parse_answer(raw, ex["subset"])
            correct = is_correct(pred, ex["gold"], ex["subset"])
            rec = {
                "subset": ex["subset"],
                "sample_id": ex["sample_id"],
                "video_id": ex["video_id"],
                "video_path": ex["video_path"],
                "question": ex["question"],
                "options": ex["options"],
                "gold": ex["gold"],
                "raw_output": raw,
                "pred": pred,
                "correct": bool(correct),
            }
            records.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    finally:
        fout.close()

    summary = _summarize(records)
    summary["config"] = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "num_frames": args.num_frames,
        "pooling_shape": pooling_shape,
        "disable_pruning": bool(args.disable_pruning),
        "selected_layer": args.selected_layer,
        "alpha": args.alpha,
        "tau": args.tau,
        "temporal_segment_ratio": args.temporal_segment_ratio,
        "cluster_ratio": args.cluster_ratio,
        "softmax": args.softmax,
        "head": args.head,
        "conv_mode": args.conv_mode,
        "decoding": FIXED_DECODING,
        "subsets": subsets,
        "max_samples": args.max_samples,
        "missing_videos": dataset.missing_videos,
        "data_root": args.data_root,
    }
    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "cwd": os.getcwd(), "argv": sys.argv}, f, indent=2, ensure_ascii=False)

    print(f"\n=== VidHalluc results ({'pruned' if not args.disable_pruning else 'baseline'}) ===")
    print(f"Overall: {summary['totals']['correct']}/{summary['totals']['n']} = {summary['totals']['accuracy']:.4f}")
    for subset, s in summary["per_subset"].items():
        line = f"  {subset:16s} {s['correct']:4d}/{s['n']:4d} = {s['accuracy']:.4f}"
        if "yes_rate" in s and s["yes_rate"] is not None:
            line += f"   yes_rate={s['yes_rate']:.3f}"
        print(line)
    print(f"Wrote: {pred_path}")
    print(f"Wrote: {os.path.join(args.save_path, 'summary.json')}")


if __name__ == "__main__":
    main()
