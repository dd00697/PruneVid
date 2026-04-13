"""Compare two VidHalluc summary.json files (baseline vs pruned)."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x):
    return "   n/a" if x is None else f"{x:.4f}"


def _fmt_delta(a, b):
    if a is None or b is None:
        return "   n/a"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="summary.json with --disable-pruning")
    parser.add_argument("--pruned", required=True, help="summary.json with pruning enabled")
    parser.add_argument("--out", type=str, default=None,
                        help="optional markdown output path")
    args = parser.parse_args()

    base = _load(args.baseline)
    prun = _load(args.pruned)

    subsets = sorted(set(base["per_subset"].keys()) | set(prun["per_subset"].keys()))

    lines = []
    header = f"{'metric':20s}{'baseline':>12s}{'pruned':>12s}{'delta':>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    b_tot = base["totals"]["accuracy"]
    p_tot = prun["totals"]["accuracy"]
    lines.append(f"{'overall':20s}{_fmt(b_tot):>12s}{_fmt(p_tot):>12s}{_fmt_delta(b_tot, p_tot):>12s}"
                 f"   n={base['totals']['n']}->{prun['totals']['n']}")

    for s in subsets:
        b = base["per_subset"].get(s, {})
        p = prun["per_subset"].get(s, {})
        b_acc = b.get("accuracy")
        p_acc = p.get("accuracy")
        extra = ""
        if "yes_rate" in b or "yes_rate" in p:
            b_y = b.get("yes_rate")
            p_y = p.get("yes_rate")
            extra = f"   yes_rate: {_fmt(b_y)} -> {_fmt(p_y)} ({_fmt_delta(b_y, p_y)})"
        lines.append(f"{s:20s}{_fmt(b_acc):>12s}{_fmt(p_acc):>12s}{_fmt_delta(b_acc, p_acc):>12s}{extra}")

    report = "\n".join(lines)
    print(report)

    if args.out:
        md = ["# VidHalluc baseline vs pruned", "", "```", report, "```", ""]
        md.append("## Configurations")
        md.append("")
        md.append(f"- baseline: `{args.baseline}`")
        md.append(f"  - disable_pruning={base.get('config', {}).get('disable_pruning')}")
        md.append(f"- pruned: `{args.pruned}`")
        md.append(f"  - disable_pruning={prun.get('config', {}).get('disable_pruning')}")
        md.append(f"  - alpha={prun.get('config', {}).get('alpha')}, tau={prun.get('config', {}).get('tau')}, "
                  f"cluster_ratio={prun.get('config', {}).get('cluster_ratio')}, "
                  f"temporal_segment_ratio={prun.get('config', {}).get('temporal_segment_ratio')}, "
                  f"selected_layer={prun.get('config', {}).get('selected_layer')}")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
