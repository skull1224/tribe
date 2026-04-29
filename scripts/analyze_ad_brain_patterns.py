#!/usr/bin/env python3
"""Analyze TRIBE ad brain-response outputs.

The script uses per-ad manifest.csv files for ROI frequency summaries and,
when available, per-ad NPZ files containing a ``preds`` array for pattern
similarity and mean-response outputs.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/Users/seong-yeob/Downloads/tribeproto")
ROI_RE = re.compile(r"(?P<region>.+?) \((?P<score>[-+]?\d+(?:\.\d+)?)\)$")
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}$")
FSAVERAGE_SIZES = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fsaverage": 163842,
}


@dataclass(frozen=True)
class AdInput:
    name: str
    folder: Path
    manifest: Path | None
    npz: Path | None
    included: bool
    excluded_reason: str
    npz_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TRIBE ad brain-response manifests and NPZ outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing one folder per advertisement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV, report, and figure outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top ROIs to show in figures and report sections.",
    )
    parser.add_argument(
        "--include-threshold-folders",
        action="store_true",
        help="Include folders whose names end with _threshold90.",
    )
    return parser.parse_args()


def discover_inputs(root: Path, include_threshold_folders: bool) -> list[AdInput]:
    data_npz = discover_data_npz(root)
    seen_data_keys: set[str] = set()
    inputs: list[AdInput] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name in {"analysis", "data"}:
            continue

        manifest_path = folder / "manifest.csv"
        manifest = manifest_path if manifest_path.is_file() else None
        npz_files = sorted(folder.glob("*.npz"))
        folder_key = ad_key(folder.name)
        seen_data_keys.add(folder_key)
        matched_data_npz = data_npz.get(folder_key, [])

        npz: Path | None = None
        if len(npz_files) == 1:
            npz = npz_files[0]
            npz_status = "found"
        elif len(npz_files) == 0:
            if len(matched_data_npz) == 1:
                npz = matched_data_npz[0]
                npz_status = "found_in_data"
            elif len(matched_data_npz) > 1:
                npz_status = "multiple_found"
            else:
                npz_status = "missing"
        else:
            npz_status = "multiple_found"

        excluded_reason = ""
        if folder.name.endswith("_threshold90") and not include_threshold_folders:
            included = False
            excluded_reason = "threshold_folder_excluded"
        elif manifest is None and npz is None:
            included = False
            excluded_reason = "missing_manifest_and_npz"
        elif npz_status == "multiple_found":
            included = True
            excluded_reason = "npz_ambiguous_for_pattern_analysis"
        else:
            included = True

        inputs.append(
            AdInput(
                name=folder.name,
                folder=folder,
                manifest=manifest,
                npz=npz,
                included=included,
                excluded_reason=excluded_reason,
                npz_status=npz_status,
            )
        )
    data_dir = root / "data"
    for key, paths in sorted(data_npz.items()):
        if key in seen_data_keys:
            continue
        if len(paths) == 1:
            inputs.append(
                AdInput(
                    name=key,
                    folder=data_dir,
                    manifest=None,
                    npz=paths[0],
                    included=True,
                    excluded_reason="",
                    npz_status="found_in_data_only",
                )
            )
        else:
            inputs.append(
                AdInput(
                    name=key,
                    folder=data_dir,
                    manifest=None,
                    npz=None,
                    included=False,
                    excluded_reason="multiple_matching_data_npz",
                    npz_status="multiple_found",
                )
            )
    return inputs


def discover_data_npz(root: Path) -> dict[str, list[Path]]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        return {}
    mapping: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*_predictions.npz")):
        mapping.setdefault(ad_key(path.stem), []).append(path)
    return mapping


def ad_key(name: str) -> str:
    key = Path(name).stem
    if key.endswith("_predictions"):
        key = key[: -len("_predictions")]
    if key.endswith("_threshold90"):
        key = key[: -len("_threshold90")]
    key = TIMESTAMP_RE.sub("", key)
    return key


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [column.lstrip("\ufeff") for column in df.columns]
    required = {"index", "time_sec", "prediction_index", "top_regions"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def parse_top_regions(raw: object) -> list[tuple[str, float]]:
    if pd.isna(raw):
        return []
    entries: list[tuple[str, float]] = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        match = ROI_RE.match(part)
        if match is None:
            continue
        entries.append((match.group("region"), float(match.group("score"))))
    return entries


def build_roi_long_table(
    inputs: Iterable[AdInput],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roi_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for ad_input in inputs:
        if not ad_input.included or ad_input.manifest is None:
            continue

        manifest = read_manifest(ad_input.manifest)
        frame_count = len(manifest)
        prediction_count = int(manifest["prediction_index"].nunique())
        manifest_rows.append(
            {
                "ad": ad_input.name,
                "manifest_path": str(ad_input.manifest),
                "frame_count": frame_count,
                "prediction_count": prediction_count,
                "first_time_sec": manifest["time_sec"].min(),
                "last_time_sec": manifest["time_sec"].max(),
            }
        )

        for _, row in manifest.iterrows():
            parsed_regions = parse_top_regions(row["top_regions"])
            for rank, (region, score) in enumerate(parsed_regions, start=1):
                roi_rows.append(
                    {
                        "ad": ad_input.name,
                        "frame_index": int(row["index"]),
                        "time_sec": float(row["time_sec"]),
                        "prediction_index": int(row["prediction_index"]),
                        "rank": rank,
                        "region": region,
                        "score": score,
                    }
                )

    return pd.DataFrame(roi_rows), pd.DataFrame(manifest_rows)


def summarize_roi_tables(
    roi_long: pd.DataFrame, manifest_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if roi_long.empty:
        columns = [
            "region",
            "appearances",
            "ads_present",
            "total_frames",
            "frame_fraction",
            "mean_score",
            "max_score",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame()

    total_frames = int(manifest_summary["frame_count"].sum())
    overall = (
        roi_long.groupby("region")
        .agg(
            appearances=("region", "size"),
            ads_present=("ad", "nunique"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    overall["total_frames"] = total_frames
    overall["frame_fraction"] = overall["appearances"] / max(total_frames, 1)
    overall = overall[
        [
            "region",
            "appearances",
            "ads_present",
            "total_frames",
            "frame_fraction",
            "mean_score",
            "max_score",
            "mean_rank",
        ]
    ].sort_values(
        ["appearances", "mean_score", "max_score"],
        ascending=[False, False, False],
    )

    ad_frames = manifest_summary.set_index("ad")["frame_count"]
    by_ad = (
        roi_long.groupby(["ad", "region"])
        .agg(
            appearances=("region", "size"),
            unique_predictions=("prediction_index", "nunique"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    by_ad["frame_count"] = by_ad["ad"].map(ad_frames).astype(int)
    by_ad["frame_fraction"] = by_ad["appearances"] / by_ad["frame_count"]
    by_ad = by_ad.sort_values(
        ["ad", "appearances", "mean_score", "max_score"],
        ascending=[True, False, False, False],
    )
    by_ad["rank_within_ad"] = by_ad.groupby("ad").cumcount() + 1
    by_ad = by_ad[
        [
            "ad",
            "rank_within_ad",
            "region",
            "appearances",
            "unique_predictions",
            "frame_count",
            "frame_fraction",
            "mean_score",
            "max_score",
            "mean_rank",
        ]
    ]
    return overall, by_ad


def load_preds(path: Path) -> tuple[np.ndarray, str]:
    with np.load(path) as data:
        if "preds" in data:
            key = "preds"
        elif "predictions" in data:
            key = "predictions"
        else:
            raise ValueError(
                f"{path} does not contain a 'preds' or 'predictions' array"
            )
        preds = np.asarray(data[key])
    if preds.ndim != 2:
        raise ValueError(f"{path} has preds.ndim={preds.ndim}, expected 2")
    return preds, key


def infer_mesh(feature_dim: int) -> str | None:
    for mesh, hemi_size in FSAVERAGE_SIZES.items():
        if feature_dim == hemi_size * 2:
            return mesh
    return None


def normalize_for_corr(patterns: np.ndarray) -> np.ndarray:
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def compute_npz_outputs(
    inputs: Iterable[AdInput],
    manifest_summary: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    status_rows: list[dict[str, object]] = []
    mean_patterns: dict[str, np.ndarray] = {}
    roi_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    roi_maps: dict[int, dict[str, np.ndarray] | None] = {}

    prediction_counts = {}
    if not manifest_summary.empty:
        prediction_counts = (
            manifest_summary.set_index("ad")["prediction_count"].to_dict()
        )

    patterns_dir = output_dir / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    for ad_input in inputs:
        status = {
            "ad": ad_input.name,
            "npz_path": str(ad_input.npz) if ad_input.npz else "",
            "npz_status": ad_input.npz_status,
            "array_key": "",
            "preds_shape": "",
            "feature_mesh": "",
            "validation_status": "",
        }

        if not ad_input.included:
            status["validation_status"] = f"skipped:{ad_input.excluded_reason}"
            status_rows.append(status)
            continue
        if ad_input.npz is None:
            status["validation_status"] = "npz_missing"
            status_rows.append(status)
            continue

        try:
            preds, array_key = load_preds(ad_input.npz)
        except Exception as exc:  # noqa: BLE001 - report all data issues.
            status["validation_status"] = f"npz_error:{exc}"
            warnings.append(f"{ad_input.name}: {exc}")
            status_rows.append(status)
            continue

        expected_predictions = prediction_counts.get(ad_input.name)
        validation_notes = ["ok"]
        if expected_predictions is not None and preds.shape[0] < expected_predictions:
            validation_notes.append(
                "preds_rows_lt_manifest_predictions:"
                f"{preds.shape[0]}<{expected_predictions}"
            )
        elif (
            expected_predictions is not None
            and preds.shape[0] != expected_predictions
        ):
            validation_notes.append(
                "preds_rows_ne_manifest_predictions:"
                f"{preds.shape[0]}!={expected_predictions}"
            )

        mesh = infer_mesh(preds.shape[1])
        status["array_key"] = array_key
        status["preds_shape"] = "x".join(str(dim) for dim in preds.shape)
        status["feature_mesh"] = mesh or "unknown"
        status["validation_status"] = ";".join(validation_notes)

        mean_pattern = preds.mean(axis=0)
        mean_patterns[ad_input.name] = mean_pattern
        np.save(
            patterns_dir / f"{safe_name(ad_input.name)}_mean_pattern.npy",
            mean_pattern,
        )

        roi_map = roi_maps.get(preds.shape[1])
        if preds.shape[1] not in roi_maps:
            roi_map, roi_warning = build_destrieux_roi_map(
                preds.shape[1], output_dir / "mne_subjects"
            )
            roi_maps[preds.shape[1]] = roi_map
            if roi_warning:
                warnings.append(roi_warning)
        if roi_map:
            roi_rows.extend(
                summarize_pattern_by_roi(
                    mean_pattern,
                    roi_map,
                    ad=ad_input.name,
                    source="ad_mean",
                )
            )

        status_rows.append(status)

    if not mean_patterns:
        return (
            pd.DataFrame(status_rows),
            pd.DataFrame(),
            pd.DataFrame(roi_rows),
            pd.DataFrame(),
            warnings,
        )

    dims = {pattern.shape[0] for pattern in mean_patterns.values()}
    if len(dims) != 1:
        warnings.append(
            "Skipping ad similarity because NPZ feature dimensions differ: "
            + ", ".join(str(dim) for dim in sorted(dims))
        )
        return (
            pd.DataFrame(status_rows),
            pd.DataFrame(),
            pd.DataFrame(roi_rows),
            pd.DataFrame(),
            warnings,
        )

    ad_names = list(mean_patterns)
    stacked = np.vstack([mean_patterns[ad] for ad in ad_names])
    normalized = normalize_for_corr(stacked)
    similarity = normalized @ normalized.T
    similarity_df = pd.DataFrame(similarity, index=ad_names, columns=ad_names)

    common_pattern = stacked.mean(axis=0)
    np.save(patterns_dir / "common_mean_pattern.npy", common_pattern)
    common_roi = pd.DataFrame()
    roi_map = roi_maps.get(common_pattern.shape[0])
    if roi_map:
        common_roi = pd.DataFrame(
            summarize_pattern_by_roi(
                common_pattern,
                roi_map,
                ad="common_mean",
                source="common_mean",
            )
        )

    return (
        pd.DataFrame(status_rows),
        similarity_df,
        pd.DataFrame(roi_rows),
        common_roi,
        warnings,
    )


def build_destrieux_roi_map(
    feature_dim: int, subjects_dir: Path
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    mesh = infer_mesh(feature_dim)
    if mesh is None:
        return None, (
            f"Skipping Destrieux ROI summary for feature_dim={feature_dim}; "
            "dimension does not match a known bilateral fsaverage mesh."
        )

    try:
        import mne
    except Exception as exc:  # noqa: BLE001 - optional dependency path.
        return None, f"Skipping Destrieux ROI summary; MNE import failed: {exc}"

    hemi_size = feature_dim // 2
    subjects_dir.mkdir(parents=True, exist_ok=True)
    try:
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
        labels = mne.read_labels_from_annot(
            "fsaverage",
            parc="aparc.a2009s",
            hemi="both",
            subjects_dir=subjects_dir,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - keep analysis usable without atlas.
        return None, f"Skipping Destrieux ROI summary; atlas load failed: {exc}"

    roi_map: dict[str, np.ndarray] = {}
    for label in labels:
        name = str(label.name)
        if name.startswith("unknown") or name.startswith("Medial_wall"):
            continue
        if name.endswith("-lh"):
            offset = 0
        elif name.endswith("-rh"):
            offset = hemi_size
        else:
            continue
        vertices = np.asarray(label.vertices)
        vertices = vertices[vertices < hemi_size]
        if vertices.size == 0:
            continue
        roi_map[destrieux_display_name(name)] = vertices + offset

    if not roi_map:
        return None, "Skipping Destrieux ROI summary; no ROI vertices matched mesh."
    return roi_map, None


def destrieux_display_name(label_name: str) -> str:
    if label_name.endswith("-lh"):
        prefix = "L "
        base = label_name[:-3]
    elif label_name.endswith("-rh"):
        prefix = "R "
        base = label_name[:-3]
    else:
        prefix = ""
        base = label_name

    replacements = [
        ("G_and_S_", "Gyrus and Sulcus "),
        ("G_", "Gyrus "),
        ("S_", "Sulcus "),
    ]
    for old, new in replacements:
        if base.startswith(old):
            base = new + base[len(old) :]
            break
    base = base.replace("_", " ")
    return prefix + base


def summarize_pattern_by_roi(
    pattern: np.ndarray,
    roi_map: dict[str, np.ndarray],
    ad: str,
    source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region, vertices in roi_map.items():
        values = pattern[vertices]
        rows.append(
            {
                "ad": ad,
                "source": source,
                "region": region,
                "n_vertices": int(vertices.size),
                "mean_response": float(np.mean(values)),
                "max_response": float(np.max(values)),
                "min_response": float(np.min(values)),
                "positive_vertex_fraction": float(np.mean(values > 0)),
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["mean_response"], row["max_response"]),
        reverse=True,
    )


def safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("_") or "ad"


def plot_overall_bar(overall: pd.DataFrame, output_path: Path, top_n: int) -> None:
    if overall.empty:
        return
    data = overall.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.36 * len(data))))
    ax.barh(data["region"], data["appearances"], color="#2f6f73")
    ax.set_xlabel("Top-region appearances across frames")
    ax.set_ylabel("ROI")
    ax.set_title("Most frequent TRIBE-predicted active regions")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.7, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_roi_heatmap(
    by_ad: pd.DataFrame,
    output_path: Path,
    top_regions: list[str],
) -> None:
    if by_ad.empty or not top_regions:
        return
    matrix = (
        by_ad[by_ad["region"].isin(top_regions)]
        .pivot_table(
            index="ad",
            columns="region",
            values="frame_fraction",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(columns=top_regions, fill_value=0.0)
    )
    if matrix.empty:
        return

    fig_width = max(12, 0.6 * len(matrix.columns))
    fig_height = max(4, 0.55 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title("ROI frame fraction by advertisement")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frame fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similarity(similarity: pd.DataFrame, output_path: Path) -> None:
    if similarity.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(similarity.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(similarity.columns)))
    ax.set_xticklabels(similarity.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(similarity.index)))
    ax.set_yticklabels(similarity.index, fontsize=8)
    ax.set_title("Mean brain-pattern correlation between ads")
    for i in range(len(similarity.index)):
        for j in range(len(similarity.columns)):
            ax.text(
                j,
                i,
                f"{similarity.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_common_brain_map(pattern: np.ndarray, output_path: Path) -> str | None:
    mesh = infer_mesh(pattern.shape[0])
    if mesh is None:
        return (
            "Skipping common brain map; common pattern dimension does not "
            "match a known bilateral fsaverage mesh."
        )

    try:
        from nilearn import datasets, plotting
    except Exception as exc:  # noqa: BLE001 - optional plotting path.
        return f"Skipping common brain map; Nilearn import failed: {exc}"

    hemi_size = pattern.shape[0] // 2
    try:
        fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    except Exception as exc:  # noqa: BLE001 - keep CSV analysis usable.
        return f"Skipping common brain map; fsaverage load failed: {exc}"

    vmax = float(np.nanpercentile(np.abs(pattern), 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(pattern))) or 1.0

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    views = [
        ("left", "lateral", pattern[:hemi_size], "Left lateral"),
        ("right", "lateral", pattern[hemi_size:], "Right lateral"),
        ("left", "medial", pattern[:hemi_size], "Left medial"),
        ("right", "medial", pattern[hemi_size:], "Right medial"),
    ]
    for index, (hemi, view, stat_map, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage[f"infl_{hemi}"],
            stat_map=stat_map,
            bg_map=fsaverage[f"sulc_{hemi}"],
            hemi=hemi,
            view=view,
            axes=ax,
            figure=fig,
            cmap="coolwarm",
            colorbar=False,
            symmetric_cbar=True,
            vmin=-vmax,
            vmax=vmax,
            title=title,
        )
    fig.suptitle("Common mean TRIBE-predicted brain pattern", fontsize=14)
    mappable = plt.cm.ScalarMappable(
        cmap="coolwarm",
        norm=plt.Normalize(vmin=-vmax, vmax=vmax),
    )
    mappable.set_array([])
    fig.colorbar(mappable, ax=fig.axes, fraction=0.025, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return None


def write_report(
    output_path: Path,
    root: Path,
    inputs: list[AdInput],
    manifest_summary: pd.DataFrame,
    overall: pd.DataFrame,
    by_ad: pd.DataFrame,
    npz_status: pd.DataFrame,
    similarity: pd.DataFrame,
    npz_roi_by_ad: pd.DataFrame,
    npz_roi_common: pd.DataFrame,
    warnings: list[str],
    top_n: int,
) -> None:
    included = [ad_input for ad_input in inputs if ad_input.included]
    excluded = [ad_input for ad_input in inputs if not ad_input.included]
    lines: list[str] = [
        "# TRIBE Ad Brain-Pattern Analysis",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Input root: `{root}`",
        "- Interpretation note: these are TRIBE model-predicted fMRI-like "
        "responses, not direct experimental fMRI measurements.",
        "",
        "## Input Inventory",
        "",
        f"- Included ad folders: {len(included)}",
        f"- Excluded folders: {len(excluded)}",
    ]

    if excluded:
        lines.append("- Exclusion details:")
        for item in excluded:
            lines.append(f"  - `{item.name}`: {item.excluded_reason}")
    lines.extend(["", "## Manifest ROI Summary", ""])

    if manifest_summary.empty:
        lines.append("- No manifest.csv files were available for ROI summaries.")
    else:
        total_frames = int(manifest_summary["frame_count"].sum())
        lines.append(
            f"- Parsed {len(manifest_summary)} manifest files and {total_frames} "
            "frame rows."
        )
        lines.append("- Overall most frequent ROIs:")
        for _, row in overall.head(top_n).iterrows():
            lines.append(
                "  - "
                f"{row['region']}: {int(row['appearances'])} appearances, "
                f"mean score {row['mean_score']:.2f}, "
                f"frame fraction {row['frame_fraction']:.3f}"
            )

    lines.extend(["", "## Advertisement-Level Highlights", ""])
    if by_ad.empty:
        lines.append("- No per-ad ROI table was generated.")
    else:
        for ad in sorted(by_ad["ad"].unique()):
            top_rows = by_ad[by_ad["ad"] == ad].head(5)
            lines.append(f"### {ad}")
            for _, row in top_rows.iterrows():
                lines.append(
                    "- "
                    f"{row['region']}: {int(row['appearances'])}/"
                    f"{int(row['frame_count'])} frames, "
                    f"mean score {row['mean_score']:.2f}"
                )
            lines.append("")

    lines.extend(["## NPZ Pattern Analysis", ""])
    if npz_status.empty:
        lines.append("- No NPZ inventory was generated.")
    else:
        found = int((npz_status["validation_status"].str.startswith("ok")).sum())
        missing = int((npz_status["validation_status"] == "npz_missing").sum())
        lines.append(f"- Valid NPZ prediction files: {found}")
        lines.append(f"- Included ads missing NPZ files: {missing}")
        lines.append(
            "- NPZ files should contain either a `preds` or `predictions` "
            "array with shape `n_predictions x n_features`."
        )
        lines.append("")
        lines.append("| Ad | NPZ status | Array key | Preds shape | Validation |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in npz_status.iterrows():
            lines.append(
                f"| {row['ad']} | {row['npz_status']} | "
                f"{row['array_key']} | {row['preds_shape']} | "
                f"{row['validation_status']} |"
            )

    lines.extend(["", "## Similarity", ""])
    if similarity.empty:
        lines.append(
            "- Advertisement similarity was skipped because no valid NPZ "
            "prediction arrays were available."
        )
    else:
        lines.append(
            "- `ad_similarity.csv` contains Pearson correlations between "
            "each ad's mean prediction vector."
        )

    lines.extend(["", "## NPZ Destrieux ROI Summary", ""])
    if npz_roi_by_ad.empty:
        lines.append(
            "- NPZ ROI summaries were skipped because no valid NPZ arrays were "
            "available or the feature dimension could not be mapped to a known "
            "fsaverage mesh."
        )
    else:
        lines.append(
            "- `npz_roi_summary_by_ad.csv` ranks Destrieux ROIs from each ad's "
            "mean prediction vector."
        )
        lines.append("- Common mean-pattern top ROIs:")
        for _, row in npz_roi_common.head(top_n).iterrows():
            lines.append(
                "  - "
                f"{row['region']}: mean response "
                f"{row['mean_response']:.4f}, peak {row['max_response']:.4f}"
            )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `roi_summary_overall.csv`",
            "- `roi_summary_by_ad.csv`",
            "- `manifest_summary.csv`",
            "- `npz_status.csv`",
            "- `ad_similarity.csv` when NPZ data is available",
            "- `npz_roi_summary_by_ad.csv` when NPZ data is available",
            "- `npz_roi_summary_common.csv` when NPZ data is available",
            "- `figures/overall_top_rois.png`",
            "- `figures/roi_heatmap_by_ad.png`",
            "- `figures/ad_similarity_heatmap.png` when NPZ data is available",
            "- `figures/common_mean_brain_map.png` when NPZ data is available",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inventory(
    inputs: list[AdInput],
    manifest_summary: pd.DataFrame,
) -> pd.DataFrame:
    manifest_counts = {}
    if not manifest_summary.empty:
        manifest_counts = manifest_summary.set_index("ad")["frame_count"].to_dict()
    rows = []
    for item in inputs:
        rows.append(
            {
                "ad": item.name,
                "folder": str(item.folder),
                "included": item.included,
                "excluded_reason": item.excluded_reason,
                "manifest_path": str(item.manifest) if item.manifest else "",
                "manifest_rows": manifest_counts.get(item.name, ""),
                "npz_path": str(item.npz) if item.npz else "",
                "npz_status": item.npz_status,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "analysis"
    )
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(root, args.include_threshold_folders)
    roi_long, manifest_summary = build_roi_long_table(inputs)
    overall, by_ad = summarize_roi_tables(roi_long, manifest_summary)
    npz_status, similarity, npz_roi_by_ad, npz_roi_common, warnings = (
        compute_npz_outputs(inputs, manifest_summary, output_dir)
    )
    inventory = write_inventory(inputs, manifest_summary)

    inventory.to_csv(output_dir / "input_inventory.csv", index=False)
    manifest_summary.to_csv(output_dir / "manifest_summary.csv", index=False)
    roi_long.to_csv(output_dir / "roi_observations_long.csv", index=False)
    overall.to_csv(output_dir / "roi_summary_overall.csv", index=False)
    by_ad.to_csv(output_dir / "roi_summary_by_ad.csv", index=False)
    npz_status.to_csv(output_dir / "npz_status.csv", index=False)
    npz_roi_by_ad.to_csv(output_dir / "npz_roi_summary_by_ad.csv", index=False)
    npz_roi_common.to_csv(output_dir / "npz_roi_summary_common.csv", index=False)
    if not similarity.empty:
        similarity.to_csv(output_dir / "ad_similarity.csv")
    else:
        pd.DataFrame().to_csv(output_dir / "ad_similarity.csv", index=False)

    top_regions = overall.head(args.top_n)["region"].tolist()
    plot_overall_bar(
        overall,
        figures_dir / "overall_top_rois.png",
        top_n=args.top_n,
    )
    plot_roi_heatmap(
        by_ad,
        figures_dir / "roi_heatmap_by_ad.png",
        top_regions=top_regions,
    )
    plot_similarity(similarity, figures_dir / "ad_similarity_heatmap.png")
    common_pattern_path = output_dir / "patterns" / "common_mean_pattern.npy"
    if common_pattern_path.is_file():
        common_pattern = np.load(common_pattern_path)
        brain_map_warning = plot_common_brain_map(
            common_pattern,
            figures_dir / "common_mean_brain_map.png",
        )
        if brain_map_warning:
            warnings.append(brain_map_warning)

    write_report(
        output_dir / "report.md",
        root,
        inputs,
        manifest_summary,
        overall,
        by_ad,
        npz_status,
        similarity,
        npz_roi_by_ad,
        npz_roi_common,
        warnings,
        args.top_n,
    )

    print(f"analysis_dir: {output_dir}")
    print(f"included_ads: {sum(item.included for item in inputs)}")
    print(f"roi_observations: {len(roi_long)}")
    npz_valid = int((npz_status["validation_status"].str.startswith("ok")).sum())
    print(f"npz_valid: {npz_valid}")
    print(f"report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
