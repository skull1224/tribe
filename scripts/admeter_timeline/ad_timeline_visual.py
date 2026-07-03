from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageFilter
from pytesseract import Output

from ad_timeline_common import (
    FRAME_COUNT,
    AdRow,
    PromptVectors,
    TableRow,
    base_row,
    brand_terms,
    clip_cache_path,
)


def frame_rows(ad: AdRow, vectors: PromptVectors) -> list[TableRow]:
    with tempfile.TemporaryDirectory(prefix="ad_timeline_frames_") as tmp:
        paths = extract_frames(ad, Path(tmp))
        scores = visual_scores(ad, vectors)
        terms = brand_terms(ad.advertiser)
        return [
            _frame_row(ad, path, i, terms, scores)
            for i, path in enumerate(paths[:FRAME_COUNT])
        ]


def extract_frames(ad: AdRow, frame_dir: Path) -> list[Path]:
    pattern = frame_dir / "frame_%03d.jpg"
    fps = max(FRAME_COUNT / ad.duration_sec, 0.05)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(ad.video_path),
        "-vf",
        f"fps={fps},scale=640:-1",
        "-frames:v",
        str(FRAME_COUNT),
        "-q:v",
        "2",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return sorted(frame_dir.glob("frame_*.jpg"))


def visual_scores(ad: AdRow, vectors: PromptVectors) -> dict[str, np.ndarray]:
    path = clip_cache_path(ad)
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        frame_embeddings = data["frame_embeddings"].astype(np.float32)
    scores = {}
    for concept, pair in vectors.items():
        pos, neg = pair
        scores[concept] = (frame_embeddings @ pos.T).mean(axis=1) - (
            frame_embeddings @ neg.T
        ).mean(axis=1)
    return scores


def _frame_row(
    ad: AdRow,
    path: Path,
    frame_index: int,
    terms: set[str],
    scores: dict[str, np.ndarray],
) -> TableRow:
    start = ad.duration_sec * frame_index / FRAME_COUNT
    end = ad.duration_sec * (frame_index + 1) / FRAME_COUNT
    text, words, conf, brand_hit = ocr_frame(path, terms)
    row = base_row(ad) | {
        "frame_index": str(frame_index),
        "start_sec": f"{start:.3f}",
        "end_sec": f"{end:.3f}",
        "center_sec": f"{(start + end) / 2:.3f}",
        "time_norm": f"{((frame_index + 0.5) / FRAME_COUNT):.4f}",
        "ocr_text": text,
        "ocr_word_count": str(words),
        "ocr_conf_mean": f"{conf:.6f}",
        "ocr_brand_hit": str(brand_hit),
    }
    row.update({key: f"{value:.6f}" for key, value in color_frame(path).items()})
    for concept, values in scores.items():
        if frame_index < len(values):
            row[f"clip_{concept}_score"] = f"{float(values[frame_index]):.6f}"
    return row


def ocr_frame(path: Path, terms: set[str]) -> tuple[str, int, float, int]:
    with Image.open(path) as image:
        prepared = image.convert("L").filter(ImageFilter.SHARPEN)
        try:
            data = pytesseract.image_to_data(
                prepared,
                lang="eng",
                config="--psm 6",
                output_type=Output.DICT,
                timeout=15,
            )
        except RuntimeError:
            return "", 0, np.nan, 0
    tokens = []
    confidences = []
    for text, conf_value in zip(
        data.get("text", []), data.get("conf", []), strict=False
    ):
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        try:
            conf = float(conf_value)
        except (TypeError, ValueError):
            continue
        if cleaned and conf >= 30:
            tokens.append(cleaned)
            confidences.append(conf)
    normalized = " ".join(re.split(r"[^a-z0-9]+", " ".join(tokens).lower()))
    brand_hit = int(any(term in normalized for term in terms))
    mean_conf = float(np.mean(confidences)) if confidences else np.nan
    return " ".join(tokens), len(tokens), mean_conf, brand_hit


def color_frame(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return {
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "brightness_mean": float(value.mean()),
        "warm_pixel_fraction": float(((hue <= 0.12) | (hue >= 0.92)).mean()),
        "cool_pixel_fraction": float(((hue >= 0.45) & (hue <= 0.72)).mean()),
    }
