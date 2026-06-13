# -*- coding: utf-8 -*-
"""
inspect_figures.py — Use playwright/Chromium to screenshot all PDF figures
and report visual compliance issues.

Strategy:
  1. Convert each PDF to PNG using macOS `sips` (converts the first page).
  2. Build a local HTML page embedding all PNGs at full width.
  3. Use playwright to render and screenshot the HTML page per figure.
  4. Run heuristic pixel checks and report findings.

Usage:
    python inspect_figures.py [--pdf-dir PATH] [--out-dir PATH]

Defaults:
  --pdf-dir  eval_results_10ns_direct/comparison_figures_legacy/
  --out-dir  <pdf-dir>/screenshots/
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ISSUES = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pdf_to_png(pdf_path: Path, out_png: Path) -> bool:
    """Convert PDF to PNG using macOS sips. Returns True on success."""
    result = subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(out_png)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and out_png.exists()


def build_html(title: str, img_path: Path) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: white; }}
    img {{ max-width: 1400px; display: block; }}
  </style>
</head>
<body>
  <img src="{img_path.as_uri()}">
</body>
</html>"""


def check_screenshot_pixels(png_path: Path, name: str):
    """Flag figures that are mostly blank or unusually monochromatic."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(str(png_path)).convert("RGB")
        arr = np.array(img, dtype=float)

        white_frac = (arr > 240).all(axis=2).mean()
        if white_frac > 0.97:
            ISSUES.setdefault(name, []).append(
                f"Mostly blank ({white_frac:.0%} white) — possible render failure")
        elif white_frac > 0.88:
            ISSUES.setdefault(name, []).append(
                f"Very light figure ({white_frac:.0%} white) — verify content is visible")

        # Colour diversity check: low std across channels → near-grayscale figure
        channel_std = arr.reshape(-1, 3).std(axis=0)
        if channel_std.max() < 8:
            ISSUES.setdefault(name, []).append(
                "Very low colour variance — figure may be entirely grayscale or blank")

    except ImportError:
        pass  # PIL not available; skip pixel check


# ── Main inspection loop ──────────────────────────────────────────────────────

def inspect_all(pdf_dir: Path, out_dir: Path):
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    print(f"Screenshots → {out_dir}\n")

    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()

            for pdf_path in pdfs:
                name = pdf_path.stem
                tmp_png = tmpdir / f"{name}_raw.png"
                out_screenshot = out_dir / f"{name}.png"

                # Step 1: PDF → PNG via sips
                if not pdf_to_png(pdf_path, tmp_png):
                    print(f"  ✗  {pdf_path.name}: sips conversion failed")
                    ISSUES[name] = ["PDF→PNG conversion failed via sips"]
                    continue

                # Step 2: Build HTML wrapper and load in browser
                html_content = build_html(name, tmp_png)
                html_path = tmpdir / f"{name}.html"
                html_path.write_text(html_content, encoding="utf-8")

                try:
                    page.goto(html_path.as_uri(), wait_until="load", timeout=10000)
                    page.wait_for_timeout(400)
                    page.screenshot(path=str(out_screenshot), full_page=True)
                    print(f"  ✓  {pdf_path.name}  →  {out_screenshot.name}")
                    check_screenshot_pixels(out_screenshot, name)
                except Exception as e:
                    print(f"  ✗  {pdf_path.name}: {e}")
                    ISSUES[name] = [f"Playwright error: {e}"]

            context.close()
            browser.close()


def print_report():
    print("\n" + "=" * 64)
    print("VISUAL INSPECTION REPORT")
    print("=" * 64)
    if not ISSUES:
        print("No issues detected across all figures.")
    else:
        for name, issues in sorted(ISSUES.items()):
            print(f"\n{name}:")
            for issue in issues:
                print(f"  • {issue}")
    print("=" * 64)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Playwright PDF figure inspector")
    parser.add_argument("--pdf-dir", default=None,
                        help="Directory of PDFs (default: comparison_figures_legacy/)")
    parser.add_argument("--out-dir", default=None,
                        help="Screenshots output dir (default: <pdf-dir>/screenshots/)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    pdf_dir = (
        Path(args.pdf_dir).expanduser().resolve() if args.pdf_dir
        else script_dir.parent / "eval_results_10ns_direct" / "comparison_figures_legacy"
    )
    if not pdf_dir.exists():
        print(f"ERROR: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    out_dir = (
        Path(args.out_dir).expanduser().resolve() if args.out_dir
        else pdf_dir / "screenshots"
    )

    inspect_all(pdf_dir, out_dir)
    print_report()
    print(f"\nScreenshots saved to: {out_dir}")


if __name__ == "__main__":
    main()
