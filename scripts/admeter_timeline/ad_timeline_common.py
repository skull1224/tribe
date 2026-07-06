from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor


ROOT = Path(__file__).resolve().parents[2]
COVERAGE = ROOT / "tmp/clip_visual_baseline/clip_video_coverage.csv"
OUT_DIR = ROOT / "tmp/ad_timeline_features"
CLIP_EMBED_DIR = ROOT / "tmp/clip_visual_baseline/clip_frame_embeddings"
CLAP_EMBED_DIR = ROOT / "tmp/clap_audio_baseline/clap_audio_embeddings"
FRAME_COUNT = 8
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SEGMENT_SEC = 10.0
STOP_TERMS = {"the", "and", "for", "with", "usa", "today", "super", "bowl"}
TableRow = dict[str, str]
PromptVectors = dict[str, tuple[np.ndarray, np.ndarray]]

VISUAL_PROMPTS = {
    "people_present": (
        [
            "a person is visible",
            "people appear in the scene",
            "a human actor is on screen",
        ],
        [
            "no people are visible",
            "an empty product shot",
            "a graphic screen with no person",
        ],
    ),
    "face_closeup": (
        ["a close up of a human face", "a person looking at the camera"],
        ["a wide shot with no visible face", "a product package without a face"],
    ),
    "funny_playful": (
        ["a funny playful commercial scene", "a comedic advertising moment"],
        ["a serious plain product shot", "a sad dramatic scene"],
    ),
    "sentimental_warm": (
        ["a warm emotional family scene", "a sentimental touching advertisement"],
        ["a cold technical product demonstration", "a fast action scene"],
    ),
    "energetic_action": (
        ["an energetic action packed scene", "fast movement and excitement"],
        ["a calm static scene", "a quiet still product shot"],
    ),
    "dramatic_dark": (
        ["a dark dramatic cinematic scene", "a tense serious mood"],
        ["a bright cheerful scene", "a playful colorful scene"],
    ),
    "product_packshot": (
        ["a product packshot", "a product close up", "brand packaging is visible"],
        ["people talking with no product visible", "a landscape without a product"],
    ),
    "logo_text": (
        [
            "a brand logo is visible",
            "large text appears on screen",
            "a title card with words",
        ],
        ["a scene with no visible text", "people in a scene without logos"],
    ),
}

AUDIO_PROMPTS = {
    "upbeat_audio": (
        ["upbeat cheerful music", "happy energetic advertising music"],
        ["quiet sad audio", "slow tense music"],
    ),
    "calm_audio": (
        ["calm gentle music", "quiet soft voiceover"],
        ["loud energetic music", "intense action sound"],
    ),
    "dramatic_audio": (
        ["dramatic cinematic music", "tense suspenseful sound"],
        ["happy playful music", "calm quiet audio"],
    ),
    "humorous_audio": (
        ["funny playful sound effects", "comedic voices and sounds"],
        ["serious narration", "dramatic music"],
    ),
    "speech_voiceover": (
        ["people speaking", "narration voiceover", "dialogue in a commercial"],
        ["instrumental music without speech", "ambient sound with no talking"],
    ),
    "music_forward": (
        ["background music", "a song playing", "commercial music bed"],
        ["spoken dialogue without music", "silence"],
    ),
}


@dataclass(frozen=True, slots=True)
class AdRow:
    year: int
    rank: int
    advertiser: str
    title: str
    source_key: str
    video_path: Path
    duration_sec: float

    @property
    def key(self) -> str:
        return f"{self.year}|{self.rank}|{self.source_key}"


def read_ads() -> list[AdRow]:
    ads = []
    with COVERAGE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["video_path"])
            if row.get("has_video") != "True" or not path.exists():
                continue
            duration = float(row.get("clip_duration_sec") or 0.0)
            if duration > 0:
                ads.append(
                    AdRow(
                        year=int(row["year"]),
                        rank=int(row["rank"]),
                        advertiser=row["advertiser"],
                        title=row["title"],
                        source_key=row["source_key"],
                        video_path=path,
                        duration_sec=duration,
                    )
                )
    return ads


def safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:80]


def brand_terms(advertiser: str) -> set[str]:
    terms = {t for t in re.split(r"[^a-z0-9]+", advertiser.lower()) if len(t) >= 3}
    return terms - STOP_TERMS


def clip_cache_path(ad: AdRow) -> Path:
    digest = hashlib.sha1(str(ad.video_path).encode("utf-8")).hexdigest()[:12]
    name = (
        f"{ad.year}_rank{ad.rank:02d}_frames08_"
        f"{safe_key(ad.source_key)}_{digest}.npz"
    )
    return CLIP_EMBED_DIR / name


def clap_cache_path(ad: AdRow) -> Path:
    path_digest = hashlib.sha1(str(ad.video_path).encode("utf-8")).hexdigest()[:12]
    model_digest = hashlib.sha1(CLAP_MODEL.encode("utf-8")).hexdigest()[:8]
    name = (
        f"{ad.year}_rank{ad.rank:02d}_seg{CLAP_SEGMENT_SEC:g}_"
        f"{safe_key(ad.source_key)}_{path_digest}_{model_digest}.npz"
    )
    return CLAP_EMBED_DIR / name


def base_row(ad: AdRow) -> TableRow:
    return {
        "year": str(ad.year),
        "rank": str(ad.rank),
        "advertiser": ad.advertiser,
        "title": ad.title,
        "source_key": ad.source_key,
        "duration_sec": f"{ad.duration_sec:.3f}",
        "video_path": str(ad.video_path),
    }


def write_csv(path: Path, rows: list[TableRow]) -> None:
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def encode_clip_text() -> PromptVectors:
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    return {
        concept: (
            _clip_text_features(model, processor, pair[0]),
            _clip_text_features(model, processor, pair[1]),
        )
        for concept, pair in VISUAL_PROMPTS.items()
    }


def _clip_text_features(
    model: CLIPModel, processor: CLIPProcessor, prompts: list[str]
) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        features = model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.detach().cpu().numpy().astype(np.float32)


def encode_clap_text() -> PromptVectors:
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    model = ClapModel.from_pretrained(CLAP_MODEL).eval()
    return {
        concept: (
            _clap_text_features(model, processor, pair[0]),
            _clap_text_features(model, processor, pair[1]),
        )
        for concept, pair in AUDIO_PROMPTS.items()
    }


def _clap_text_features(
    model: ClapModel, processor: ClapProcessor, prompts: list[str]
) -> np.ndarray:
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        features = model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return features.detach().cpu().numpy().astype(np.float32)
