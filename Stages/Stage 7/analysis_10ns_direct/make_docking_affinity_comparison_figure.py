#!/usr/bin/env python3
"""
Render a publication-quality 5-seed docking affinity comparison figure.

Source of truth:
  Stages/Stage 5/results/matrix/*.pdbqt

Output:
  Stages/Stage 7/eval_results_10ns_direct/comparison_figures/
    - docking_affinity_comparison.png
    - docking_affinity_comparison.pdf
    - docking_affinity_summary.csv
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stage7_eval_common import apply_pub_style, save_pub_figure


CONDITION_META = {
    "ERDRP_WT": {
        "ligand": "ERDRP-0519",
        "receptor": "WT",
        "label": "ERDRP-0519\nWT",
        "color": "#0072B2",
        "edge": "#004F7A",
    },
    "ERDRP_MUT": {
        "ligand": "ERDRP-0519",
        "receptor": "W730A",
        "label": "ERDRP-0519\nW730A",
        "color": "#56B4E9",
        "edge": "#2A7AA6",
    },
    "BMS_WT": {
        "ligand": "BMS-986205",
        "receptor": "WT",
        "label": "BMS-986205\nWT",
        "color": "#009E73",
        "edge": "#006B4F",
    },
    "BMS_MUT": {
        "ligand": "BMS-986205",
        "receptor": "W730A",
        "label": "BMS-986205\nW730A",
        "color": "#CC79A7",
        "edge": "#934F79",
    },
}

PAIR_ORDER = [("ERDRP_WT", "ERDRP_MUT"), ("BMS_WT", "BMS_MUT")]
PLOT_ORDER = ["ERDRP_WT", "ERDRP_MUT", "BMS_WT", "BMS_MUT"]
SEED_RE = re.compile(r"^(BMS|ERDRP)_(WT|MUT)_seed_(\d+)\.pdbqt$")


def parse_args():
    parser = argparse.ArgumentParser(description="Render docking affinity comparison figure.")
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("Stages/Stage 5/results/matrix"),
        help="Directory containing Stage 5 docking matrix PDBQT files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Stages/Stage 7/eval_results_10ns_direct/comparison_figures"),
        help="Directory for figure outputs.",
    )
    return parser.parse_args()


def extract_best_affinity(pdbqt_path: Path) -> float:
    best = None
    with pdbqt_path.open("r") as handle:
        for line in handle:
            if not line.startswith("REMARK VINA RESULT"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            value = float(parts[3])
            best = value if best is None else min(best, value)
    if best is None:
        raise RuntimeError(f"No Vina affinity found in {pdbqt_path}")
    return best


def load_seed_matrix(matrix_dir: Path):
    values = {key: {} for key in CONDITION_META}
    for pdbqt_path in sorted(matrix_dir.glob("*.pdbqt")):
        match = SEED_RE.match(pdbqt_path.name)
        if not match:
            continue
        condition = f"{match.group(1)}_{match.group(2)}"
        seed = int(match.group(3))
        values[condition][seed] = extract_best_affinity(pdbqt_path)

    missing = [key for key, entries in values.items() if not entries]
    if missing:
        raise FileNotFoundError(
            f"Missing docking results for condition(s): {', '.join(missing)} in {matrix_dir}"
        )
    return values


def summarize(values):
    summary = {}
    for key, seed_map in values.items():
        ordered = sorted(seed_map.items())
        scores = np.array([score for _, score in ordered], dtype=float)
        summary[key] = {
            "seeds": [seed for seed, _ in ordered],
            "scores": scores,
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n": int(len(scores)),
        }
    return summary


def write_summary_csv(summary, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "ligand", "receptor", "n", "mean_kcal_mol", "sd_kcal_mol", "seed_values"])
        for condition in PLOT_ORDER:
            meta = CONDITION_META[condition]
            row = summary[condition]
            seed_values = "; ".join(
                f"{seed}:{score:.3f}" for seed, score in zip(row["seeds"], row["scores"])
            )
            writer.writerow([
                condition,
                meta["ligand"],
                meta["receptor"],
                row["n"],
                f"{row['mean']:.4f}",
                f"{row['sd']:.4f}",
                seed_values,
            ])


def make_figure(summary, out_stem: Path):
    apply_pub_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(PLOT_ORDER))

    for condition, xpos in zip(PLOT_ORDER, x):
        meta = CONDITION_META[condition]
        row = summary[condition]

        ax.bar(
            xpos,
            row["mean"],
            width=0.72,
            color=meta["color"],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.errorbar(
            xpos,
            row["mean"],
            yerr=row["sd"],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=4,
            capthick=0.8,
            zorder=4,
        )

        offsets = np.linspace(-0.11, 0.11, row["n"])
        ax.scatter(
            xpos + offsets,
            row["scores"],
            s=18,
            facecolors="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_ylabel("Docking affinity (kcal/mol)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax.text(
        0.005,
        1.02,
        "Bars, mean affinity; whiskers, s.d.; points, individual docking seeds (n = 5).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#4B5563",
    )

    handles = [plt.Rectangle((0, 0), 1, 1, color=CONDITION_META[key]["color"]) for key in PLOT_ORDER]
    legend_labels = [
        "A: ERDRP-0519 / WT",
        "B: ERDRP-0519 / W730A",
        "C: BMS-986205 / WT",
        "D: BMS-986205 / W730A",
    ]
    ax.legend(
        handles,
        legend_labels,
        fontsize=6,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    delta_labels = []
    for wt_key, mut_key in PAIR_ORDER:
        ligand = CONDITION_META[wt_key]["ligand"]
        delta = summary[mut_key]["mean"] - summary[wt_key]["mean"]
        delta_labels.append(f"{ligand}: Δ(W730A - WT) = {delta:+.2f} kcal/mol")

    fig.text(0.125, 0.03, delta_labels[0], ha="left", va="bottom", fontsize=6.5, color="#4B5563")
    fig.text(0.125, 0.005, delta_labels[1], ha="left", va="bottom", fontsize=6.5, color="#4B5563")

    fig.tight_layout(rect=[0, 0.08, 1, 0.90], pad=1.0)
    save_pub_figure(fig, str(out_stem))


def main():
    args = parse_args()
    values = load_seed_matrix(args.matrix_dir)
    summary = summarize(values)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, args.out_dir / "docking_affinity_summary.csv")
    make_figure(summary, args.out_dir / "docking_affinity_comparison")
    print(f"Wrote figure to {args.out_dir}")


if __name__ == "__main__":
    main()
