# -*- coding: utf-8 -*-
"""
make_figure1_docking.py — Figure 1: five-seed mean docking affinities.

Reads focused_100_seed_summary.csv from Stage 5 and produces the connected
dot-plot showing WT vs W730A mean ± SD docking scores for BMS-986205 and
ERDRP-0519.

Usage:
    python make_figure1_docking.py [--seed-summary PATH] [--out-dir PATH]

Output:  eval_results_10ns_direct/comparison_figures/figure1_docking_affinities.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Style constants (mirrors make_comparison_figures.py) ────────────────────
PUB_DPI   = 600
LINE_W    = 1.5

# Compound display config — colours match CASE_META in make_comparison_figures.py
COMPOUND_META = {
    "BMS-986205": {
        "wt_color":  "#009E73",   # C_BMS_WT  green
        "mut_color": "#CC79A7",   # D_BMS_MUT pink
        "marker":    "^",
        "label":     "BMS-986205",
        "x_wt":      1.0,
        "x_mut":     2.0,
    },
    "ERDRP-0519": {
        "wt_color":  "#0072B2",   # A_ERDRP_WT blue
        "mut_color": "#E69F00",   # B_ERDRP_MUT orange
        "marker":    "o",
        "label":     "ERDRP-0519",
        "x_wt":      4.0,
        "x_mut":     5.0,
    },
}

# Map CSV name values to compound keys
NAME_MAP = {
    "BMS-986205": "BMS-986205",
    "ERDRP-0519": "ERDRP-0519",
}

DEFAULT_SEED_SUMMARY = (
    Path(__file__).resolve().parent.parent.parent.parent  # nipahv-rdrp-md/
    / "Stages" / "Stage 5" / "results" / "focused_100_seed_summary.csv"
)


def apply_pub_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8,
        "axes.labelsize":     9,
        "axes.titlesize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "lines.antialiased":  True,
        "figure.dpi":         150,
        "savefig.dpi":        PUB_DPI,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def save_pub_figure(fig, path_stem):
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem + ext, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def load_seed_data(seed_summary_path):
    """Return dict: {compound_name: {receptor_state: (mean, sd)}}."""
    df = pd.read_csv(seed_summary_path)
    data = {}
    for _, row in df.iterrows():
        name = row["name"]
        if name not in NAME_MAP:
            continue
        key  = NAME_MAP[name]
        state = "WT" if row["receptor_state"] == "WT" else "W730A"
        data.setdefault(key, {})[state] = (
            float(row["mean_affinity"]),
            float(row["sd_affinity"]),
        )
    return data


def plot_figure1(seed_data, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # ── Draw one connected pair per compound ──────────────────────────────
    for comp, meta in COMPOUND_META.items():
        if comp not in seed_data:
            print(f"  WARNING: {comp} not found in seed summary - skipping")
            continue
        states = seed_data[comp]
        if "WT" not in states or "W730A" not in states:
            print(f"  WARNING: {comp} missing WT or W730A row - skipping")
            continue

        wt_mean,  wt_sd  = states["WT"]
        mut_mean, mut_sd = states["W730A"]
        x_wt  = meta["x_wt"]
        x_mut = meta["x_mut"]
        mk    = meta["marker"]

        # Connecting line between WT and W730A points
        delta = mut_mean - wt_mean
        line_color = meta["wt_color"]
        ax.plot([x_wt, x_mut], [wt_mean, mut_mean],
                color=line_color, linewidth=LINE_W, linestyle="-",
                zorder=2, alpha=0.75)

        # WT point + error bar
        ax.errorbar(x_wt, wt_mean, yerr=wt_sd,
                    fmt=mk, color=meta["wt_color"], markersize=7,
                    capsize=4, capthick=1.0, elinewidth=1.0,
                    label=f"{comp} (WT)", zorder=3)

        # W730A point + error bar
        ax.errorbar(x_mut, mut_mean, yerr=mut_sd,
                    fmt=mk, color=meta["mut_color"], markersize=7,
                    capsize=4, capthick=1.0, elinewidth=1.0,
                    markerfacecolor=meta["mut_color"],
                    label=f"{comp} (W730A)", zorder=3)

        # Δ annotation — position midway, offset left
        mid_x = (x_wt + x_mut) / 2
        mid_y = (wt_mean + mut_mean) / 2
        sign  = "+" if delta >= 0 else ""
        ax.annotate(
            f"Δ {sign}{delta:.2f}",
            xy=(mid_x, mid_y),
            xytext=(mid_x - 0.35, mid_y + 0.08),
            fontsize=7,
            color="dimgrey",
            ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # ── X-axis ticks & labels ─────────────────────────────────────────────
    tick_positions = [
        COMPOUND_META["BMS-986205"]["x_wt"],
        COMPOUND_META["BMS-986205"]["x_mut"],
        COMPOUND_META["ERDRP-0519"]["x_wt"],
        COMPOUND_META["ERDRP-0519"]["x_mut"],
    ]
    tick_labels = ["WT", "W730A", "WT", "W730A"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Compound name labels below the two pairs
    for comp, meta in COMPOUND_META.items():
        mid = (meta["x_wt"] + meta["x_mut"]) / 2
        ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] < -8.5 else -8.55,
                meta["label"], ha="center", va="top", fontsize=8,
                fontweight="bold", color="black")

    # ── Axes decoration ───────────────────────────────────────────────────
    ax.set_ylabel("Docking affinity (kcal/mol)")
    ax.set_xlim(0.4, 5.6)

    # Leave enough headroom above and below
    all_means = []
    all_sds   = []
    for comp, states in seed_data.items():
        if comp not in COMPOUND_META:
            continue
        for m, s in states.values():
            all_means.append(m)
            all_sds.append(s)
    if all_means:
        y_min = min(m - s for m, s in zip(all_means, all_sds)) - 0.35
        y_max = max(m + s for m, s in zip(all_means, all_sds)) + 0.35
        ax.set_ylim(y_min, y_max)

    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axhline(-7.0, color="silver", linewidth=0.7, linestyle=":",
               label="−7.0 kcal/mol threshold", zorder=1)

    # Vertical separator between the two compounds
    ax.axvline(3.0, color="lightgrey", linewidth=0.8, linestyle="--", zorder=1)

    ax.legend(loc="lower left", fontsize=7, framealpha=0.85, ncol=2)
    ax.set_title("Five-seed mean docking affinities: WT vs W730A", fontsize=9)

    fig.tight_layout()
    out_stem = str(out_dir / "figure1_docking_affinities")
    save_pub_figure(fig, out_stem)
    print(f"  figure1_docking_affinities.png / .pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: five-seed docking affinities WT vs W730A"
    )
    parser.add_argument(
        "--seed-summary", default=None,
        help="Path to focused_100_seed_summary.csv (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: eval_results_10ns_direct/comparison_figures/)",
    )
    args = parser.parse_args()

    # Resolve seed summary path
    seed_path = (
        Path(args.seed_summary).expanduser().resolve()
        if args.seed_summary
        else DEFAULT_SEED_SUMMARY
    )
    if not seed_path.exists():
        print(f"ERROR: seed summary not found: {seed_path}")
        sys.exit(1)

    # Resolve output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir.parent / "eval_results_10ns_direct" / "comparison_figures"

    out_dir.mkdir(parents=True, exist_ok=True)

    apply_pub_style()
    print(f"Reading seed summary: {seed_path}")
    print(f"Writing to:          {out_dir}")
    print()

    seed_data = load_seed_data(seed_path)
    plot_figure1(seed_data, out_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
