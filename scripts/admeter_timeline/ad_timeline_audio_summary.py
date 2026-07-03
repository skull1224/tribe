from __future__ import annotations

import json

import numpy as np

from ad_timeline_common import (
    CLAP_SEGMENT_SEC,
    AdRow,
    PromptVectors,
    TableRow,
    base_row,
    clap_cache_path,
)


VISUAL_MOOD_FIELDS = [
    "clip_funny_playful_score",
    "clip_sentimental_warm_score",
    "clip_energetic_action_score",
    "clip_dramatic_dark_score",
]
AUDIO_MOOD_FIELDS = [
    "clap_upbeat_audio_score",
    "clap_calm_audio_score",
    "clap_dramatic_audio_score",
    "clap_humorous_audio_score",
]


def audio_rows(ad: AdRow, vectors: PromptVectors) -> list[TableRow]:
    path = clap_cache_path(ad)
    if not path.exists():
        return []
    with np.load(path, allow_pickle=True) as data:
        embeddings = data["segment_embeddings"].astype(np.float32)
        meta = json.loads(str(data["meta"].item()))
    duration = float(meta.get("duration_sec", ad.duration_sec))
    segment_sec = float(meta.get("segment_sec", CLAP_SEGMENT_SEC))
    return [
        _audio_row(ad, embedding, i, duration, segment_sec, vectors)
        for i, embedding in enumerate(embeddings)
    ]


def _audio_row(
    ad: AdRow,
    embedding: np.ndarray,
    index: int,
    duration: float,
    segment_sec: float,
    vectors: PromptVectors,
) -> TableRow:
    start = index * segment_sec
    end = min(duration, (index + 1) * segment_sec)
    row = base_row(ad) | {
        "segment_index": str(index),
        "start_sec": f"{start:.3f}",
        "end_sec": f"{end:.3f}",
        "center_sec": f"{(start + end) / 2:.3f}",
        "time_norm": f"{((start + end) / 2 / duration):.4f}",
    }
    for concept, pair in vectors.items():
        pos, neg = pair
        score = (embedding @ pos.T).mean() - (embedding @ neg.T).mean()
        row[f"clap_{concept}_score"] = f"{float(score):.6f}"
    return row


def summary_rows(frames: list[TableRow], audio: list[TableRow]) -> list[TableRow]:
    frame_groups = group_rows(frames)
    audio_groups = group_rows(audio)
    out = []
    items = sorted(
        frame_groups.items(),
        key=lambda item: (int(item[1][0]["year"]), int(item[1][0]["rank"])),
    )
    for key, rows_for_ad in items:
        out.append(summary_row(rows_for_ad, audio_groups.get(key, [])))
    return out


def group_rows(rows: list[TableRow]) -> dict[str, list[TableRow]]:
    grouped: dict[str, list[TableRow]] = {}
    for row in rows:
        key = row["year"] + "|" + row["rank"] + "|" + row["source_key"]
        grouped.setdefault(key, []).append(row)
    return grouped


def summary_row(frames: list[TableRow], audio: list[TableRow]) -> TableRow:
    first = frames[0]
    person_time, person_score = peak(frames, "clip_people_present_score")
    sat_time, sat_score = peak(frames, "saturation_mean")
    face_time, face_score = peak(frames, "clip_face_closeup_score")
    mood = best_family(frames, VISUAL_MOOD_FIELDS)
    audio_mood = best_family(audio, AUDIO_MOOD_FIELDS) if audio else ""
    return {
        "year": first["year"],
        "rank": first["rank"],
        "advertiser": first["advertiser"],
        "title": first["title"],
        "source_key": first["source_key"],
        "ocr_brand_time_ranges": ranges(frames, "ocr_brand_hit", 1.0),
        "people_peak_time": person_time,
        "people_peak_score": person_score,
        "people_time_ranges_score_gt_0_02": ranges(
            frames, "clip_people_present_score", 0.02
        ),
        "people_strong_ranges_score_gt_0_055": ranges(
            frames, "clip_people_present_score", 0.055
        ),
        "face_closeup_peak_time": face_time,
        "face_closeup_peak_score": face_score,
        "face_closeup_ranges_score_gt_0_00": ranges(
            frames, "clip_face_closeup_score", 0.0
        ),
        "saturation_mean": (
            f"{np.mean([float(r['saturation_mean']) for r in frames]):.6f}"
        ),
        "saturation_peak_time": sat_time,
        "saturation_peak_score": sat_score,
        "high_saturation_ranges_gt_0_45": ranges(frames, "saturation_mean", 0.45),
        "visual_top_score_family": mood.removeprefix("clip_").removesuffix("_score"),
        "audio_top_score_family": audio_mood.removeprefix("clap_").removesuffix(
            "_score"
        ),
    }


def ranges(rows: list[TableRow], field: str, threshold: float) -> str:
    intervals = [
        (float(row["start_sec"]), float(row["end_sec"]))
        for row in rows
        if field in row and float(row.get(field, "nan")) >= threshold
    ]
    return "; ".join(f"{start:.1f}-{end:.1f}s" for start, end in intervals)


def peak(rows: list[TableRow], field: str) -> tuple[str, str]:
    valid = [row for row in rows if field in row and row[field] != ""]
    if not valid:
        return "", ""
    top = max(valid, key=lambda row: float(row[field]))
    return f"{float(top['center_sec']):.1f}s", top[field]


def best_family(rows: list[TableRow], fields: list[str]) -> str:
    if not rows:
        return ""
    usable = [field for field in fields if field in rows[0]]
    if not usable:
        return ""
    return max(
        usable,
        key=lambda field: np.mean([float(row[field]) for row in rows if field in row]),
    )
