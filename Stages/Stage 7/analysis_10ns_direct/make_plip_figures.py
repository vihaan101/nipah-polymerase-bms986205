# -*- coding: utf-8 -*-
"""
make_plip_figures.py — Generate publication-quality standalone PDFs for the BMS
PLIP 3D interaction renders (Supplementary Figs. S1/S2).

Each case produces one image-only figure PDF (G1-compliant: no combined panels).
The per-pose interaction inventories are NOT rendered here — they live as editable
Markdown tables (Supplementary Tables S7/S8) directly in
CBC_Supplementary_Information_Draft.md. The `interactions` dict below is retained
only as a structured provenance record of the values transcribed into those tables
from the PLIP report files (`*_plip_report.txt`).

Usage:
    python make_plip_figures.py [--out-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_DPI = 600
# Pre-upscale factor: 1200×800 → 4800×3200, then embed at native res.
# This removes matplotlib's blurring pass; the PDF viewer renders crisp.
UPSCALE = 4

CASES = [
    {
        "key": "BMS_WT",
        "png": "BMS_WT_seed42_plip.png",
        "out": "BMS_WT_seed42_plip",
        "title": "C: BMS-986205 / WT (Success)",
        "subtitle": "PLIP — BMS_WT_SEED42",
        "color": "#009E73",
        "interactions": {
            "Hydrophobic": [
                ("LEU723", "3.93 Å"), ("LEU723", "3.93 Å"), ("PHE726", "3.73 Å"),
                ("TRP730", "3.60 Å"), ("TRP806", "3.80 Å"), ("THR810", "3.94 Å"),
                ("LEU870", "3.39 Å"),
            ],
            "H-bond": [
                ("HIS875 (side)", "D–A 3.86 Å, 164.9°"),
            ],
            "π-Stacking": [],
            "π-Cation":   [],
        },
    },
    {
        "key": "BMS_MUT",
        "png": "BMS_MUT_seed42_plip.png",
        "out": "BMS_MUT_seed42_plip",
        "title": "D: BMS-986205 / W730A (Resistance)",
        "subtitle": "PLIP — BMS_MUT_SEED42",
        "color": "#CC79A7",
        "interactions": {
            "Hydrophobic": [
                ("LEU723", "3.99 Å"), ("GLN803", "3.72 Å"), ("TRP806", "3.80 Å"),
                ("LEU870", "3.86 Å"), ("LEU870", "3.66 Å"),
            ],
            "H-bond": [
                ("GLY800 (bb)", "D–A 3.83 Å, 143.5°"),
                ("HIS875 (side)", "D–A 3.33 Å, 132.5°"),
            ],
            "π-Stacking": [
                ("PHE726", "4.36 Å, 17.8°, parallel"),
            ],
            "π-Cation": [
                ("HIS875", "4.38 Å, offset 1.95 Å"),
            ],
        },
    },
]

def apply_pub_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         7,
        "axes.labelsize":    7,
        "axes.titlesize":    7,
        "lines.antialiased": True,
        "figure.dpi":        150,
        "savefig.dpi":       PUB_DPI,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def _load_and_crop(img_path, pad=20, upscale=UPSCALE):
    """Load PNG, autocrop white border, upscale with LANCZOS for crisp embedding."""
    pil = Image.open(str(img_path)).convert("RGBA")
    arr = np.array(pil, dtype=np.float32) / 255.0

    # Mask: non-transparent, non-white pixels
    mask = (arr[:, :, 3] > 0.05) & ~np.all(arr[:, :, :3] > 0.93, axis=2)
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        cropped = pil
    else:
        h, w = arr.shape[:2]
        r0 = max(0,     rows[0]  - pad)
        r1 = min(h - 1, rows[-1] + pad)
        c0 = max(0,     cols[0]  - pad)
        c1 = min(w - 1, cols[-1] + pad)
        cropped = pil.crop((c0, r0, c1 + 1, r1 + 1))

    if upscale != 1:
        new_w = cropped.width  * upscale
        new_h = cropped.height * upscale
        cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    return np.array(cropped, dtype=np.uint8)


def make_plip_figure(case_meta, out_dir):
    """Standalone 3D visualization figure (image only — G1 compliant main figure)."""
    img_path = out_dir / case_meta["png"]
    if not img_path.exists():
        print(f"  WARNING: PNG not found: {img_path}")
        return

    fig, ax_img = plt.subplots(figsize=(7.0, 5.0))
    img = _load_and_crop(img_path)
    ax_img.imshow(img, interpolation="none")
    ax_img.axis("off")

    ax_img.text(
        0.02, 0.02, case_meta["title"],
        transform=ax_img.transAxes,
        fontsize=7, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=case_meta["color"],
                  alpha=0.88, linewidth=0),
        va="bottom", ha="left",
    )

    out_stem = str(out_dir / case_meta["out"])
    fig.savefig(out_stem + ".pdf", dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {case_meta['out']}.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate PLIP figure PDFs")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir.parent / "eval_results_10ns_direct" / "comparison_figures"

    if not out_dir.exists():
        print(f"ERROR: output dir not found: {out_dir}")
        sys.exit(1)

    apply_pub_style()
    print(f"Writing to: {out_dir}")
    for case in CASES:
        make_plip_figure(case, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
