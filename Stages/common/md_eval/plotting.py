"""Publication-quality matplotlib helpers for MD eval figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_DPI = 300
MARKER_EVERY = 20


def apply_pub_style() -> None:
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": PUB_DPI,
        "savefig.dpi": PUB_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pub_figure(fig, path_stem: str) -> None:
    """Save figure as PNG and PDF, then close."""
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem + ext, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
