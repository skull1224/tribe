from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rsa_expanded_stats import corr_fast, corr_many, permutation_rank_matrix, rank_behavior


ROOT: Final = Path("/Users/seong-yeob/PycharmProjects/tribe")
SOURCE: Final = ROOT / "tmp/acr_content_baseline/features_common_clip_clap_asr_acr_tribe.csv"
CONTENT: Final = ROOT / "tmp/content_timing_baseline/clip_content_timing_features.csv"
OCR: Final = ROOT / "tmp/ocr_timing_baseline/ocr_timing_features.csv"
SENSORY: Final = ROOT / "tmp/feedback_priority_analysis/low_level_sensory_features.csv"
OUT_DIR: Final = ROOT / "tmp/rsa_expanded_comparison"
KEY_COLS: Final = ["year", "rank", "source_key"]
RANDOM_SEED: Final = 20260704
N_PERM: Final = 100
PCA_K: Final = 50

ROI_3: Final = ["cortical_dlPFC", "subcortical_amygdala", "cortical_vmPFC_proxy"]
ROI_5: Final = ROI_3 + [
    "subcortical_hippocampus",
    "subcortical_ventral_striatum_accumbens",
]
ROI_LABELS: Final = {
    "cortical_dlPFC": "dlPFC",
    "subcortical_amygdala": "Amygdala",
    "cortical_vmPFC_proxy": "vmPFC",
    "subcortical_hippocampus": "Hippocampus",
    "subcortical_ventral_striatum_accumbens": "Ventral striatum",
}
ROI_WINDOWS: Final = [
    "whole",
    "pct25_75",
    "pct50_100",
    "last5",
    "std",
    "rolling_peak_10s",
    "rolling_abs_peak_10s",
]
WINDOW_LABELS: Final = {
    "whole": "Whole",
    "pct25_75": "25-75%",
    "pct50_100": "50-100%",
    "last5": "Last 5s",
    "std": "Temporal SD",
    "rolling_peak_10s": "Peak 10s",
    "rolling_abs_peak_10s": "Abs peak 10s",
}

Behavior = Literal["continuous_admeter_distance", "top_bottom25_category"]
Space = Literal["standardized", "pca50"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    group: str
    name: str
    cols: tuple[str, ...]
    roi: str = ""
    window: str = ""


def prefixed(df: pd.DataFrame, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in exclude or not col.startswith(prefixes):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            cols.append(col)
    return cols


def available_rois(df: pd.DataFrame) -> list[str]:
    rois: list[str] = []
    for col in df.columns:
        if "__" not in col:
            continue
        roi, window = col.rsplit("__", 1)
        if window in ROI_WINDOWS and roi.startswith(("cortical_", "subcortical_")):
            rois.append(roi)
    return list(dict.fromkeys(rois))


def roi_cols(rois: list[str], signed: bool = False) -> list[str]:
    prefix = "" if signed else "abs_"
    return [f"{prefix}{roi}__{window}" for roi in rois for window in ROI_WINDOWS]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, low_memory=False)
    if CONTENT.exists():
        content = pd.read_csv(CONTENT)
        df = df.merge(content[KEY_COLS + prefixed(content, ("ct_",))], on=KEY_COLS, how="left")
    if OCR.exists():
        ocr = pd.read_csv(OCR)
        ocr_cols = KEY_COLS + prefixed(ocr, ("ocr_",), ("ocr_error", "ocr_text_joined"))
        df = df.merge(ocr[ocr_cols], on=KEY_COLS, how="left")
    if SENSORY.exists():
        sensory = pd.read_csv(SENSORY)
        sensory_cols = KEY_COLS + prefixed(sensory, ("low_",), ("low_error", "low_has_video_file"))
        df = df.merge(sensory[sensory_cols], on=KEY_COLS, how="left")
    abs_data = {
        f"abs_{col}": df[col].abs()
        for col in roi_cols(available_rois(df), signed=True)
    }
    return pd.concat([df, pd.DataFrame(abs_data, index=df.index)], axis=1).copy()


def feature_specs(df: pd.DataFrame) -> list[FeatureSpec]:
    clip = prefixed(df, ("clip_mean__",))
    clap = prefixed(df, ("clap_mean__",))
    asr = prefixed(df, ("asr_hash__", "asr_meta__"))
    acr = prefixed(df, ("acr_visual__", "acr_meta__"))
    content = prefixed(df, ("ct_",))
    ocr = prefixed(df, ("ocr_",), ("ocr_error", "ocr_text_joined"))
    sensory = prefixed(df, ("low_",), ("low_error", "low_has_video_file"))
    low_video = [c for c in sensory if c.startswith(("low_brightness", "low_saturation", "low_motion", "low_dark", "low_high"))]
    low_audio = [c for c in sensory if c.startswith("low_audio")]
    tribe_3 = [c for c in roi_cols(ROI_3) if c in df.columns]
    tribe_5 = [c for c in roi_cols(ROI_5) if c in df.columns]
    tribe_5_signed = [c for c in roi_cols(ROI_5, signed=True) if c in df.columns]
    tribe_all = [f"abs_{c}" for c in roi_cols(available_rois(df), signed=True) if f"abs_{c}" in df.columns]
    specs = [
        FeatureSpec("Baseline control", "Duration only", ("duration_sec",)),
        FeatureSpec("Baseline control", "Low-level video sensory", tuple(low_video)),
        FeatureSpec("Baseline control", "Low-level audio sensory", tuple(low_audio)),
        FeatureSpec("Baseline control", "Low-level sensory all", tuple(sensory)),
        FeatureSpec("Baseline modality", "CLIP visual embedding", tuple(clip)),
        FeatureSpec("Baseline modality", "CLAP audio embedding", tuple(clap)),
        FeatureSpec("Baseline modality", "ASR text embedding/meta", tuple(asr)),
        FeatureSpec("Baseline modality", "ACR content tags", tuple(acr)),
        FeatureSpec("Baseline cue", "OCR timing", tuple(ocr)),
        FeatureSpec("Baseline cue", "CLIP content timing all", tuple(content)),
        FeatureSpec("TRIBE set", "TRIBE 3ROI abs", tuple(tribe_3)),
        FeatureSpec("TRIBE set", "TRIBE 5ROI abs", tuple(tribe_5)),
        FeatureSpec("TRIBE set", "TRIBE 5ROI signed", tuple(tribe_5_signed)),
        FeatureSpec("TRIBE set", "TRIBE all available ROI abs", tuple(tribe_all)),
        FeatureSpec("Combined", "CLIP + TRIBE 5ROI", tuple(clip + tribe_5)),
        FeatureSpec("Combined", "CLAP + TRIBE 5ROI", tuple(clap + tribe_5)),
        FeatureSpec("Combined", "ASR + TRIBE 5ROI", tuple(asr + tribe_5)),
        FeatureSpec("Combined", "ACR + TRIBE 5ROI", tuple(acr + tribe_5)),
        FeatureSpec("Combined", "OCR + TRIBE 5ROI", tuple(ocr + tribe_5)),
        FeatureSpec("Combined", "Content timing + TRIBE 5ROI", tuple(content + tribe_5)),
        FeatureSpec("Combined", "CLIP+CLAP+ASR+ACR", tuple(clip + clap + asr + acr)),
        FeatureSpec("Combined", "CLIP+CLAP+ASR+ACR+TRIBE", tuple(clip + clap + asr + acr + tribe_5)),
        FeatureSpec("Combined", "All non-TRIBE baselines", tuple(clip + clap + asr + acr + content + ocr + sensory)),
        FeatureSpec("Combined", "All non-TRIBE baselines + TRIBE", tuple(clip + clap + asr + acr + content + ocr + sensory + tribe_5)),
    ]
    for roi in ROI_5:
        cols = [f"abs_{roi}__{window}" for window in ROI_WINDOWS if f"abs_{roi}__{window}" in df.columns]
        specs.append(FeatureSpec("TRIBE ROI", f"TRIBE ROI {ROI_LABELS[roi]}", tuple(cols), roi=roi))
    for window in ROI_WINDOWS:
        cols = [f"abs_{roi}__{window}" for roi in ROI_5 if f"abs_{roi}__{window}" in df.columns]
        specs.append(FeatureSpec("TRIBE time window", f"TRIBE window {WINDOW_LABELS[window]}", tuple(cols), window=window))
    for roi in ROI_5:
        for window in ROI_WINDOWS:
            col = f"abs_{roi}__{window}"
            if col in df.columns:
                specs.append(FeatureSpec("TRIBE ROI-window cell", f"{ROI_LABELS[roi]} {WINDOW_LABELS[window]}", (col,), roi, window))
    return [spec for spec in specs if spec.cols]


def top_bottom_labels(work: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(work), np.nan, dtype=float)
    for _, sub in work.groupby("year", sort=False):
        low = sub["rating_z_within_year"].quantile(0.25)
        high = sub["rating_z_within_year"].quantile(0.75)
        labels[work.index.get_indexer(sub.index[sub["rating_z_within_year"] <= low])] = 0.0
        labels[work.index.get_indexer(sub.index[sub["rating_z_within_year"] >= high])] = 1.0
    return labels


def transformed_matrix(work: pd.DataFrame, cols: tuple[str, ...], space: Space) -> np.ndarray:
    x = SimpleImputer(strategy="median").fit_transform(work[list(cols)].to_numpy(float))
    x = StandardScaler().fit_transform(x)
    if space == "pca50" and x.shape[1] > PCA_K:
        n_components = min(PCA_K, x.shape[0] - 1, x.shape[1])
        return PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(x)
    return x


def rsa_row(df: pd.DataFrame, spec: FeatureSpec, behavior: Behavior, space: Space, seed: int) -> dict[str, str | int | float] | None:
    cols = tuple(dict.fromkeys(col for col in spec.cols if col in df.columns))
    mask = df["rating_z_within_year"].notna() & df[list(cols)].notna().any(axis=1)
    work = df.loc[mask, ["year", "rating_z_within_year", *cols]].copy()
    if behavior == "top_bottom25_category":
        labels = top_bottom_labels(work)
        keep = ~np.isnan(labels)
        work = work.loc[keep].copy()
        y = labels[keep]
    else:
        y = work["rating_z_within_year"].to_numpy(float)
    usable_cols = tuple(col for col in cols if work[col].notna().any())
    if len(work) < 10 or len(np.unique(y)) < 2 or not usable_cols:
        return None
    x = transformed_matrix(work, usable_cols, space)
    neural = np.nan_to_num(pdist(x, metric="euclidean" if x.shape[1] == 1 else "correlation"), nan=1.0, posinf=1.0, neginf=1.0)
    i, j = np.triu_indices(len(y), 1)
    behavioral = (y[i] != y[j]).astype(float) if behavior == "top_bottom25_category" else np.abs(y[i] - y[j])
    between_mask = behavioral > 0
    within = float(neural[~between_mask].mean()) if behavior == "top_bottom25_category" else np.nan
    between = float(neural[between_mask].mean()) if behavior == "top_bottom25_category" else np.nan
    neural_rank = rank_behavior(neural, "continuous_admeter_distance")
    behavior_rank = rank_behavior(behavioral, behavior)
    observed = corr_fast(neural_rank, behavior_rank)
    null_ranks = permutation_rank_matrix(y, behavior, i, j, N_PERM, seed)
    null = corr_many(neural_rank, null_ranks)
    return {
        "sample": "common_clip_clap_asr_acr_tribe",
        "behavior_model": behavior,
        "feature_group": spec.group,
        "feature_set": spec.name,
        "space": space,
        "n": int(len(work)),
        "n_features": int(len(usable_cols)),
        "effective_features": int(x.shape[1]),
        "neural_distance": "euclidean" if x.shape[1] == 1 else "correlation",
        "rsa_spearman": float(observed),
        "p_perm_two_sided": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (N_PERM + 1)),
        "n_perm": N_PERM,
        "pair_count": int(len(neural)),
        "mean_within_neural_distance": within,
        "mean_between_neural_distance": between,
        "between_minus_within": float(between - within) if np.isfinite(within) else np.nan,
        "roi": spec.roi,
        "window": spec.window,
    }


def run_rsa(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for spec_idx, spec in enumerate(feature_specs(df)):
        spaces: tuple[Space, ...] = ("standardized", "pca50") if len(spec.cols) > PCA_K else ("standardized",)
        for space in spaces:
            for behavior in ("continuous_admeter_distance", "top_bottom25_category"):
                row = rsa_row(df, spec, behavior, space, RANDOM_SEED + spec_idx)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def preferred_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.copy()
    work["preferred_space"] = np.where(work["n_features"] > PCA_K, "pca50", "standardized")
    return work[work["space"].eq(work["preferred_space"])].drop(columns=["preferred_space"])
