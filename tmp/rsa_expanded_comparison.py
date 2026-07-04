#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "seaborn",
# ]
# ///
# --- How to run ---
# .venv/bin/python tmp/rsa_expanded_comparison.py

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rsa_expanded_comparison_core import (
    N_PERM,
    OUT_DIR,
    PCA_K,
    ROI_5,
    ROI_LABELS,
    ROI_WINDOWS,
    WINDOW_LABELS,
    Behavior,
    load_data,
    preferred_metrics,
    run_rsa,
)


BEHAVIOR_TITLES = {
    "continuous_admeter_distance": "Continuous AdMeter distance",
    "top_bottom25_category": "Top/bottom 25% category",
}


def add_chart_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.98, title, ha="left", va="top", fontsize=13, fontweight="semibold", color="#1F2430")
    fig.text(left, 0.925, subtitle, ha="left", va="top", fontsize=9, color="#6F768A")


def plot_ranked(metrics: pd.DataFrame, behavior: Behavior, output_name: str) -> None:
    plot_df = (
        metrics[
            metrics["behavior_model"].eq(behavior)
            & ~metrics["feature_group"].eq("TRIBE ROI-window cell")
        ]
        .sort_values("rsa_spearman", ascending=False)
        .head(26)
        .sort_values("rsa_spearman", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10.5, 9))
    sns.barplot(data=plot_df, x="rsa_spearman", y="feature_set", color="#A3BEFA", edgecolor="#2E4780", ax=ax)
    ax.axvline(0, color="#1F2430", linewidth=1)
    ax.set_xlabel("RSA Spearman rho")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E6E8F0")
    add_chart_header(
        fig,
        ax,
        f"Expanded RSA ranking: {BEHAVIOR_TITLES[behavior]}",
        "Preferred space: PCA50 for >50 feature columns; otherwise standardized features.",
    )
    sns.despine(ax=ax)
    fig.savefig(OUT_DIR / output_name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_roi_window_heatmaps(metrics: pd.DataFrame) -> None:
    cell = metrics[
        metrics["feature_group"].eq("TRIBE ROI-window cell") & metrics["space"].eq("standardized")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, behavior in zip(axes, ["continuous_admeter_distance", "top_bottom25_category"]):
        sub = cell[cell["behavior_model"].eq(behavior)].copy()
        sub["roi_label"] = sub["roi"].map(ROI_LABELS)
        sub["window_label"] = sub["window"].map(WINDOW_LABELS)
        matrix = sub.pivot(index="roi_label", columns="window_label", values="rsa_spearman")
        matrix = matrix.reindex(
            index=[ROI_LABELS[roi] for roi in ROI_5],
            columns=[WINDOW_LABELS[window] for window in ROI_WINDOWS],
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap=sns.blend_palette(["#FFFFFF", "#CEDFFE", "#A3BEFA", "#5477C4"], as_cmap=True),
            linewidths=1,
            linecolor="#FFFFFF",
            cbar=ax is axes[-1],
            ax=ax,
        )
        ax.set_title(BEHAVIOR_TITLES[behavior], fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    add_chart_header(
        fig,
        axes[0],
        "TRIBE 5ROI window-level RSA",
        "Single ROI-window feature RSA against AdMeter distance and top/bottom 25% category structure.",
    )
    fig.savefig(OUT_DIR / "rsa_5roi_window_heatmaps.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_outputs(metrics: pd.DataFrame) -> None:
    preferred = preferred_metrics(metrics)
    inventory = metrics[["feature_group", "feature_set", "n_features"]].drop_duplicates()
    top = (
        preferred.sort_values(["behavior_model", "rsa_spearman"], ascending=[True, False])
        .groupby("behavior_model")
        .head(12)
    )
    metrics.to_csv(OUT_DIR / "rsa_expanded_metrics.csv", index=False)
    preferred.to_csv(OUT_DIR / "rsa_expanded_preferred_space_metrics.csv", index=False)
    inventory.to_csv(OUT_DIR / "rsa_feature_set_inventory.csv", index=False)
    top.to_csv(OUT_DIR / "rsa_top_summary.csv", index=False)
    lines = [
        "# Expanded RSA comparison",
        "",
        f"- Source sample: common CLIP+CLAP+ASR+ACR+TRIBE file, max n={int(metrics['n'].max())}.",
        f"- Permutation test: {N_PERM} label shuffles per feature set and behavior model.",
        "- `standardized` uses imputed, z-scored features.",
        f"- `pca50` is additionally reported for feature sets with more than {PCA_K} columns.",
        "- Preferred-space summaries use PCA50 for >50-column feature sets and standardized space otherwise.",
        "",
        "## Top preferred-space RSA rows",
        "",
        "```",
        top.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")
    plot_ranked(preferred, "continuous_admeter_distance", "rsa_ranked_continuous.png")
    plot_ranked(preferred, "top_bottom25_category", "rsa_ranked_topbottom25.png")
    plot_roi_window_heatmaps(preferred)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = run_rsa(load_data())
    write_outputs(metrics)
    print(metrics.sort_values("rsa_spearman", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
