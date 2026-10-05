"""MCS-based ligand RMSD helpers shared by Stage 1 redocking validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS


def mcs_rmsd_between_sdfs(reference_sdf: Path, docked_sdf: Path, timeout: int = 30) -> float:
    """Heavy-atom RMSD without superimposition using MCS atom correspondence."""
    ref_mol = Chem.MolFromMolFile(str(reference_sdf), removeHs=True)
    dock_mol = Chem.MolFromMolFile(str(docked_sdf), removeHs=True)
    if ref_mol is None or dock_mol is None:
        raise ValueError(f"Could not load SDFs for RMSD: {reference_sdf}, {docked_sdf}")

    mcs_result = rdFMCS.FindMCS(
        [ref_mol, dock_mol],
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=timeout,
    )
    if mcs_result.numAtoms < ref_mol.GetNumAtoms():
        raise ValueError(
            f"MCS incomplete ({mcs_result.numAtoms}/{ref_mol.GetNumAtoms()} atoms matched)"
        )

    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    dock_match = dock_mol.GetSubstructMatch(mcs_mol)
    ref_conf = ref_mol.GetConformer()
    dock_conf = dock_mol.GetConformer()

    sq_diffs = []
    for r_idx, d_idx in zip(ref_match, dock_match):
        rp = np.array(ref_conf.GetAtomPosition(r_idx))
        dp = np.array(dock_conf.GetAtomPosition(d_idx))
        sq_diffs.append(float(np.sum((rp - dp) ** 2)))
    return float(np.sqrt(np.mean(sq_diffs)))
