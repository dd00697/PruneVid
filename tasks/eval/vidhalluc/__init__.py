"""VidHalluc evaluation dataset for PLLaVA / PruneVid.

Flattens the four VidHalluc annotation files into a unified example list,
resolves video paths under ``<data_root>/data/{ACH,STH,TSH}`` (with a fallback
to the nested ``data/VidHalluc/data/{...}`` layout), and reuses
``tasks.eval.eval_utils.EvalDataset.read_video`` for frame sampling so that
preprocessing is identical to MVBench / VideoMME evaluation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from tasks.eval.eval_utils import EvalDataset


SUBSETS = ("ach_binaryqa", "ach_mcq", "sth", "tsh")

_SUBSET_VIDEO_DIR = {
    "ach_binaryqa": "ACH",
    "ach_mcq": "ACH",
    "sth": "STH",
    "tsh": "TSH",
}

_SUBSET_FILENAME = {
    "ach_binaryqa": "ach_binaryqa.json",
    "ach_mcq": "ach_mcq.json",
    "sth": "sth.json",
    "tsh": "tsh.json",
}

_BINARY_INSTRUCTION = "Answer the question with only 'Yes' or 'No'."
_MCQ_INSTRUCTION = (
    "Select the best answer from the options below. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
_TSH_INSTRUCTION = (
    "Which of the two actions occurs first in the video? "
    "Answer with two letters indicating the order "
    "(AB if Action A comes first, BA if Action B comes first)."
)


def _candidate_video_roots(data_root: str, subset: str):
    name = _SUBSET_VIDEO_DIR[subset]
    # Official unzip puts ACH/STH directly under data_root/data/, but TSH
    # sometimes unpacks one level deeper (data_root/data/VidHalluc/data/TSH).
    return [
        os.path.join(data_root, "data", name),
        os.path.join(data_root, "data", "VidHalluc", "data", name),
    ]


def _video_path(data_root: str, subset: str, video_id: str) -> str:
    for root in _candidate_video_roots(data_root, subset):
        candidate = os.path.join(root, f"{video_id}.mp4")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_candidate_video_roots(data_root, subset)[0], f"{video_id}.mp4")


class VidHallucDataset(EvalDataset):
    """VidHalluc dataset spanning ach_binaryqa / ach_mcq / sth / tsh."""

    def __init__(
        self,
        data_root: str,
        subsets: Sequence[str],
        num_segments: int,
        max_samples: Optional[int] = None,
    ):
        super().__init__(num_segments=num_segments, test_ratio=None)
        self.data_root = os.path.abspath(data_root)
        self.subsets = list(subsets)
        self.data_list: List[Dict[str, Any]] = []
        self.missing_videos: Dict[str, int] = {}

        for subset in self.subsets:
            if subset not in SUBSETS:
                raise ValueError(f"Unknown VidHalluc subset: {subset}")
            loader = getattr(self, f"_load_{subset}")
            loader()

        if max_samples is not None and max_samples > 0:
            self.data_list = self.data_list[:max_samples]

    def __len__(self) -> int:
        return len(self.data_list)

    def __str__(self) -> str:
        counts: Dict[str, int] = {}
        for ex in self.data_list:
            counts[ex["subset"]] = counts.get(ex["subset"], 0) + 1
        lines = [f"VidHallucDataset: {len(self.data_list)} examples"]
        for k, v in counts.items():
            lines.append(f"  {k}: {v}")
        if self.missing_videos:
            lines.append("  missing videos:")
            for k, v in self.missing_videos.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.data_list[index]
        images = self.read_video(ex["video_path"])
        out = dict(ex)
        out["images"] = images
        return out

    def _append(self, subset: str, video_id: str, question: str, gold: str,
                options: Optional[Dict[str, str]] = None,
                sample_id: Optional[str] = None):
        vp = _video_path(self.data_root, subset, video_id)
        if not os.path.exists(vp):
            self.missing_videos[subset] = self.missing_videos.get(subset, 0) + 1
            return
        if sample_id is None:
            sample_id = f"{subset}:{video_id}:{len(self.data_list)}"
        self.data_list.append({
            "subset": subset,
            "sample_id": sample_id,
            "video_id": video_id,
            "video_path": vp,
            "question": question,
            "options": options,
            "gold": gold,
        })

    def _load_ach_binaryqa(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["ach_binaryqa"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_id, q_list in data.items():
            if not isinstance(q_list, list):
                continue
            for q_idx, q_entry in enumerate(q_list):
                question = q_entry.get("q", "")
                answers = q_entry.get("a", {}) or {}
                for video_id, gold in answers.items():
                    sid = f"ach_binaryqa:{group_id}:{q_idx}:{video_id}"
                    self._append("ach_binaryqa", video_id, question, str(gold), sample_id=sid)

    def _load_ach_mcq(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["ach_mcq"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_id, vids in data.items():
            if not isinstance(vids, dict):
                continue
            for video_id, entry in vids.items():
                question = entry.get("Question", "")
                choices = entry.get("Choices", {}) or {}
                gold = entry.get("Correct Answer", "")
                sid = f"ach_mcq:{group_id}:{video_id}"
                self._append(
                    "ach_mcq", video_id, question, str(gold).strip(),
                    options=dict(choices), sample_id=sid,
                )

    def _load_sth(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["sth"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for video_id, entry in data.items():
            gold = entry.get("Scene change", "")
            question = "Is there a scene change in this video?"
            sid = f"sth:{video_id}"
            self._append("sth", video_id, question, str(gold), sample_id=sid)

    def _load_tsh(self):
        path = os.path.join(self.data_root, _SUBSET_FILENAME["tsh"])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, entry in data.items():
            video_id = entry.get("video", "")
            question = entry.get("Question", "")
            gold = entry.get("Correct Answer", "")
            sid = f"tsh:{idx}:{video_id}"
            self._append("tsh", video_id, question, str(gold).strip().upper(), sample_id=sid)


def build_prompt(example: Dict[str, Any]) -> str:
    """Build the per-subset user query text (without conversation wrapping)."""
    subset = example["subset"]
    q = example["question"]
    if subset in ("ach_binaryqa", "sth"):
        return f"{_BINARY_INSTRUCTION}\nQuestion: {q}"
    if subset == "ach_mcq":
        opts = example.get("options") or {}
        opt_lines = "\n".join(f"{k}. {v}" for k, v in opts.items())
        return f"{_MCQ_INSTRUCTION}\nQuestion: {q}\n{opt_lines}"
    if subset == "tsh":
        return f"{q}\n{_TSH_INSTRUCTION}"
    raise ValueError(f"Unknown subset: {subset}")


_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_MCQ_RE = re.compile(r"\b([ABCD])\b")
_TSH_RE = re.compile(r"\b(AB|BA)\b", re.IGNORECASE)


def parse_answer(raw: str, subset: str) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if subset in ("ach_binaryqa", "sth"):
        m = _YESNO_RE.search(text)
        if not m:
            return None
        return m.group(1).capitalize()
    if subset == "ach_mcq":
        m = _MCQ_RE.search(text.upper())
        if not m:
            return None
        return m.group(1)
    if subset == "tsh":
        m = _TSH_RE.search(text.upper())
        if not m:
            return None
        return m.group(1).upper()
    raise ValueError(f"Unknown subset: {subset}")


def is_correct(pred: Optional[str], gold: str, subset: str) -> bool:
    if pred is None:
        return False
    if subset == "ach_mcq":
        return pred.strip().upper() == str(gold).strip().upper()
    if subset == "tsh":
        return pred.strip().upper() == str(gold).strip().upper()
    return pred.strip().lower() == str(gold).strip().lower()
