#!/usr/bin/env python3
"""Export seed-42 WT/MUT poses for ERDRP (and BMS) into Stage 5 results/matrix/."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import resolve_vina_path  # noqa: E402

STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
STAGE5_RESULTS = Path(__file__).resolve().parent / "results"
MATRIX_DIR = STAGE5_RESULTS / "matrix"

WT_RECEPTOR = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"
ERDRP_LIGAND = STAGE1_DIR / "data" / "ligands" / "ERDRP.pdbqt"
BMS_LIGAND = STAGE3_LIGAND = PROJECT_ROOT / "Stages" / "Stage 3" / "data" / "ligands" / "BMS-986205.pdbqt"
LEAD_MANIFEST = STAGE4_DIR / "results" / "verification" / "lead_selection.json"

DEFAULT_SEED = 42


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def parse_affinity(pdbqt: Path) -> float | None:
    with open(pdbqt) as fh:
        for line in fh:
            if "REMARK VINA RESULT" in line:
                return float(line.split()[3])
    return None


def run_dock(vina: Path, box: dict, receptor: Path, ligand: Path, out: Path, seed: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(vina),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        str(box["center_x"]),
        "--center_y",
        str(box["center_y"]),
        "--center_z",
        str(box["center_z"]),
        "--size_x",
        str(box["size_x"]),
        "--size_y",
        str(box["size_y"]),
        "--size_z",
        str(box["size_z"]),
        "--exhaustiveness",
        str(box.get("exhaustiveness", 16)),
        "--num_modes",
        "1",
        "--scoring",
        "vinardo",
        "--seed",
        str(seed),
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def copy_lead_pose(job_id: str, receptor_key: str, dest: Path) -> None:
    """Use Stage 4 manifest WT/MUT poses when available (BMS)."""
    if not LEAD_MANIFEST.exists():
        raise FileNotFoundError(f"Stage 4 lead manifest required for {job_id}: {LEAD_MANIFEST}")
    manifest = json.loads(LEAD_MANIFEST.read_text())
    key = "pose_path_wt" if receptor_key == "WT" else "pose_path_mut"
    pose_value = manifest.get(key) or manifest.get("pose_path")
    if not pose_value:
        raise RuntimeError(f"Lead manifest missing {key}")
    src = Path(pose_value)
    if not src.is_absolute():
        src = STAGE4_DIR / "results" / "verification" / Path(pose_value).name
        if not src.exists():
            src = PROJECT_ROOT / pose_value
    require_file(src, f"BMS {receptor_key} pose")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage 6 matrix poses (seed 42).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--vina-path", type=Path, default=None)
    parser.add_argument("--skip-bms", action="store_true", help="Only export ERDRP poses")
    parser.add_argument("--use-lead-manifest-for-bms", action="store_true", default=True)
    args = parser.parse_args()

    vina = resolve_vina_path(args.vina_path)
    box_path = require_file(STAGE1_DIR / "config" / "docking_box.json", "docking box")
    box = json.loads(box_path.read_text())
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("ERDRP_WT", WT_RECEPTOR, ERDRP_LIGAND, False),
        ("ERDRP_MUT", MUT_RECEPTOR, ERDRP_LIGAND, False),
    ]
    if not args.skip_bms:
        jobs.extend(
            [
                ("BMS_WT", WT_RECEPTOR, BMS_LIGAND, True),
                ("BMS_MUT", MUT_RECEPTOR, BMS_LIGAND, True),
            ]
        )

    manifest_rows = []
    for job_id, receptor, ligand, from_lead in jobs:
        out = MATRIX_DIR / f"{job_id}_seed_{args.seed}.pdbqt"
        if from_lead and args.use_lead_manifest_for_bms and LEAD_MANIFEST.exists():
            receptor_key = "WT" if "WT" in job_id else "MUT"
            copy_lead_pose(job_id, receptor_key, out)
        else:
            require_file(receptor, f"{job_id} receptor")
            require_file(ligand, f"{job_id} ligand")
            run_dock(vina, box, receptor, ligand, out, args.seed)
        aff = parse_affinity(out)
        manifest_rows.append({"job_id": job_id, "path": str(out), "affinity": aff, "seed": args.seed})

    summary_path = MATRIX_DIR / "matrix_export_manifest.json"
    summary_path.write_text(json.dumps(manifest_rows, indent=2) + "\n")
    print(f"Wrote {len(manifest_rows)} matrix poses under {MATRIX_DIR}")


if __name__ == "__main__":
    main()
