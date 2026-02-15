"""Dataset inspection utilities for ANIMA UI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
PATTERN_REPEAT_FOLDER = re.compile(r"^\d+_.+")


def normalize_path(path: str) -> str:
    if path is None:
        return ""
    cleaned = path.strip()
    if not cleaned:
        return ""
    return os.path.normpath(cleaned)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def detect_image_dirs(base_path: str) -> List[str]:
    normalized = normalize_path(base_path)
    if not normalized:
        return []

    root = Path(normalized)
    if not root.exists() or not root.is_dir():
        return []

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    matched = [p for p in subdirs if PATTERN_REPEAT_FOLDER.match(p.name)]

    if not matched:
        return [str(root)]

    matched.sort(key=lambda p: p.name.lower())
    return [str(p) for p in matched]


def _caption_state(image_path: Path) -> Dict[str, Any]:
    base = image_path.with_suffix("")
    for ext in (".txt", ".tags"):
        caption_path = base.with_suffix(ext)
        if caption_path.exists() and caption_path.is_file():
            try:
                content = caption_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"has_caption": False, "caption_file": str(caption_path), "reason": "not_utf8"}

            if content.strip() == "":
                return {"has_caption": False, "caption_file": str(caption_path), "reason": "blank"}

            return {"has_caption": True, "caption_file": str(caption_path), "reason": None}

    return {"has_caption": False, "caption_file": None, "reason": "missing"}


@dataclass
class DirStats:
    image_dir: str
    image_count: int
    caption_valid_count: int
    caption_invalid_count: int


def inspect_dataset(base_path: str) -> Dict[str, Any]:
    normalized = normalize_path(base_path)
    result: Dict[str, Any] = {
        "input_path": base_path,
        "normalized_path": normalized,
        "exists": False,
        "image_dirs": [],
        "stats": [],
        "image_count": 0,
        "caption_valid_count": 0,
        "caption_invalid_count": 0,
    }

    if not normalized:
        result["error"] = "empty_path"
        return result

    root = Path(normalized)
    if not root.exists() or not root.is_dir():
        result["error"] = "path_not_found"
        return result

    result["exists"] = True
    image_dirs = detect_image_dirs(normalized)
    result["image_dirs"] = image_dirs

    all_stats: List[DirStats] = []
    for img_dir_str in image_dirs:
        img_dir = Path(img_dir_str)
        images = [p for p in img_dir.iterdir() if _is_image(p)]

        valid = 0
        invalid = 0
        for image_path in images:
            state = _caption_state(image_path)
            if state["has_caption"]:
                valid += 1
            else:
                invalid += 1

        all_stats.append(
            DirStats(
                image_dir=str(img_dir),
                image_count=len(images),
                caption_valid_count=valid,
                caption_invalid_count=invalid,
            )
        )

    result["stats"] = [asdict(s) for s in all_stats]
    result["image_count"] = sum(s.image_count for s in all_stats)
    result["caption_valid_count"] = sum(s.caption_valid_count for s in all_stats)
    result["caption_invalid_count"] = sum(s.caption_invalid_count for s in all_stats)
    return result

