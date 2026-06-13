#!/usr/bin/env python3
"""Shared docking helpers for locked compound-selection workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
DEFAULT_VINA_PATH = PROJECT_ROOT / "scripts" / "vina"
GHOST_BACKBONE_AND_CB = {"N", "CA", "C", "O", "CB"}
TARGET_RES_ID = 730
CHAIN_ID = "A"

SEVERE_GHOST_CLASH_THRESHOLD = 2.5
MODERATE_GHOST_CLASH_THRESHOLD = 3.2
SEVERE_GHOST_CLASH_PENALTY = 2.0
MODERATE_GHOST_CLASH_PENALTY = 0.5
MUTATION_SCORE_VERSION = "mutation_composite_score_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def atomic_write_json(path: Path, data: dict | list) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_vina_path(vina_path: str | Path | None = None) -> Path:
    candidate = Path(vina_path or os.environ.get("VINA_PATH") or DEFAULT_VINA_PATH).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Vina executable not found at {candidate}. Pass --vina-path or set VINA_PATH."
        )
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Vina executable is not executable: {candidate}")
    return candidate


def vina_version(vina_path: Path) -> str:
    for args in (["--version"], []):
        try:
            result = subprocess.run(
                [str(vina_path), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            continue
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        if output:
            return output.splitlines()[0]
    return "unknown"


def load_docking_box(path: Path | None = None) -> dict:
    box_path = path or STAGE1_DIR / "config" / "docking_box.json"
    with open(box_path, "r", encoding="utf-8") as handle:
        box = json.load(handle)
    required = {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
    missing = required - set(box)
    if missing:
        raise KeyError(f"Missing docking box fields in {box_path}: {sorted(missing)}")
    return box


def build_vina_command(
    vina_path: Path,
    receptor: Path,
    ligand: Path,
    output: Path,
    box: dict,
    *,
    seed: int = 42,
    cpu: int = 1,
    exhaustiveness: int | None = None,
) -> list[str]:
    return [
        str(vina_path),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness if exhaustiveness is not None else box.get("exhaustiveness", 16)),
        "--num_modes", "1",
        "--scoring", "vinardo",
        "--seed", str(seed),
        "--cpu", str(cpu),
        "--out", str(output),
    ]


def run_vina_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def parse_affinity(pdbqt_file: Path) -> float | None:
    if not pdbqt_file.exists():
        return None
    best = None
    with open(pdbqt_file, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "REMARK VINA RESULT" in line:
                value = float(line.split()[3])
                best = value if best is None else min(best, value)
    return best


def get_ghost_sidechain_coords(wt_pdb: Path, chain_id: str = CHAIN_ID, resid: int = TARGET_RES_ID) -> np.ndarray:
    try:
        from Bio.PDB import PDBParser
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required for ghost-sidechain geometry checks. "
            "Install biopython in the active environment before paired screening."
        ) from exc
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB missing: {wt_pdb}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb))
    try:
        residue = structure[0][chain_id][resid]
    except KeyError as exc:
        raise RuntimeError(f"Residue {chain_id}:{resid} not found in {wt_pdb}") from exc
    if residue.resname != "TRP":
        raise RuntimeError(f"Expected TRP at {chain_id}:{resid}, found {residue.resname}")
    coords = [atom.get_coord() for atom in residue if atom.name not in GHOST_BACKBONE_AND_CB]
    if len(coords) != 9:
        raise RuntimeError(f"Expected 9 ghost sidechain atoms, found {len(coords)}")
    return np.array(coords)


def ligand_model1_coords(pdbqt_file: Path) -> np.ndarray:
    coords = []
    with open(pdbqt_file, "r", encoding="utf-8", errors="ignore") as handle:
        in_model1 = False
        saw_model = False
        for line in handle:
            if line.startswith("MODEL"):
                saw_model = True
                in_model1 = line.split()[1:2] == ["1"]
                continue
            if line.startswith("ENDMDL") and in_model1:
                break
            if (in_model1 or not saw_model) and line.startswith(("ATOM", "HETATM")):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
    if not coords:
        raise RuntimeError(f"No ligand coordinates found in {pdbqt_file}")
    return np.array(coords)


def min_distance_to_ghost(ligand_pdbqt: Path, ghost_coords: np.ndarray) -> float:
    return float(np.min(distance.cdist(ligand_model1_coords(ligand_pdbqt), ghost_coords)))


def pose_center(pdbqt_file: Path) -> np.ndarray:
    return np.mean(ligand_model1_coords(pdbqt_file), axis=0)


def pose_center_shift(wt_pdbqt: Path, mut_pdbqt: Path) -> float:
    return float(np.linalg.norm(pose_center(mut_pdbqt) - pose_center(wt_pdbqt)))


def ghost_warning_band(ghost_clash_dist: float) -> str:
    if ghost_clash_dist < SEVERE_GHOST_CLASH_THRESHOLD:
        return "severe"
    if ghost_clash_dist < MODERATE_GHOST_CLASH_THRESHOLD:
        return "moderate"
    return "none"


def ghost_penalty(ghost_clash_dist: float) -> float:
    band = ghost_warning_band(ghost_clash_dist)
    if band == "severe":
        return SEVERE_GHOST_CLASH_PENALTY
    if band == "moderate":
        return MODERATE_GHOST_CLASH_PENALTY
    return 0.0


def _z_inverse(values: Iterable[float]) -> list[float]:
    clean = [float(v) for v in values]
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / len(clean)
    sd = math.sqrt(variance)
    if sd == 0:
        return [0.0 for _ in clean]
    return [-(v - mean) / sd for v in clean]


def add_mutation_composite_scores(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    metric_columns = {
        "wt_affinity": [row["wt_affinity"] for row in rows],
        "mut_affinity": [row["mut_affinity"] for row in rows],
        "affinity_loss": [max(row["delta_affinity"], 0.0) for row in rows],
        "abs_delta_dist": [abs(row["delta_dist"]) for row in rows],
        "pose_center_shift": [row["pose_center_shift"] for row in rows],
    }
    z = {name: _z_inverse(values) for name, values in metric_columns.items()}
    scored = []
    for idx, row in enumerate(rows):
        result = dict(row)
        result["score_version"] = MUTATION_SCORE_VERSION
        result["ghost_warning"] = ghost_warning_band(float(row["ghost_clash_dist"]))
        result["ghost_penalty"] = ghost_penalty(float(row["ghost_clash_dist"]))
        result["mutation_composite_score"] = (
            z["wt_affinity"][idx]
            + z["mut_affinity"][idx]
            + z["affinity_loss"][idx]
            + z["abs_delta_dist"][idx]
            + z["pose_center_shift"][idx]
            - result["ghost_penalty"]
        )
        scored.append(result)
    scored.sort(key=lambda item: (-item["mutation_composite_score"], item["name"]))
    for rank, row in enumerate(scored, start=1):
        row["rank_mutation_aware"] = rank
    return scored


def validate_workers(requested_workers: int, cpu_count_value: int | None = None) -> int:
    available = cpu_count_value or (os.cpu_count() or 1)
    if requested_workers < 1:
        raise ValueError("--workers must be >= 1")
    max_workers = max(1, available - 1)
    return min(requested_workers, max_workers)


def config_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def valid_output(path: Path, expected_config_hash: str | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if expected_config_hash is None:
        return True
    meta = path.with_suffix(path.suffix + ".meta.json")
    if not meta.exists():
        return False
    try:
        with open(meta, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return False
    return data.get("config_hash") == expected_config_hash


def write_output_meta(path: Path, expected_config_hash: str, extra: dict | None = None) -> None:
    payload = {"config_hash": expected_config_hash}
    if extra:
        payload.update(extra)
    atomic_write_json(path.with_suffix(path.suffix + ".meta.json"), payload)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
