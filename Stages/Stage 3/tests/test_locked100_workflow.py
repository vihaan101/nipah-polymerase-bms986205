from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
STAGE3_DIR = TESTS_DIR.parent
PROJECT_ROOT = STAGE3_DIR.parents[1]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

import docking_utils  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECT_LOCKED = _load_module(STAGE3_DIR / "select_library_100_locked.py", "select_library_100_locked")
SCREEN = _load_module(STAGE3_DIR / "screen_library_100.py", "screen_library_100")


class Locked100Stage3Tests(unittest.TestCase):
    def test_mutation_composite_score_handles_zero_variance_and_ghost_penalty(self) -> None:
        rows = [
            {
                "name": "A",
                "wt_affinity": -7.0,
                "mut_affinity": -7.5,
                "delta_affinity": -0.5,
                "delta_dist": 0.2,
                "pose_center_shift": 1.0,
                "ghost_clash_dist": 3.5,
            },
            {
                "name": "B",
                "wt_affinity": -7.0,
                "mut_affinity": -7.5,
                "delta_affinity": -0.5,
                "delta_dist": 0.2,
                "pose_center_shift": 1.0,
                "ghost_clash_dist": 2.0,
            },
        ]

        scored = docking_utils.add_mutation_composite_scores(rows)

        by_name = {row["name"]: row for row in scored}
        self.assertEqual(by_name["A"]["score_version"], "mutation_composite_score_v1")
        self.assertEqual(by_name["A"]["mutation_composite_score"], 0.0)
        self.assertEqual(by_name["B"]["ghost_warning"], "severe")
        self.assertEqual(by_name["B"]["mutation_composite_score"], -2.0)
        self.assertEqual(by_name["A"]["rank_mutation_aware"], 1)

    def test_build_vina_command_pins_cpu_one(self) -> None:
        cmd = docking_utils.build_vina_command(
            Path("/tmp/vina"),
            Path("wt.pdbqt"),
            Path("lig.pdbqt"),
            Path("out.pdbqt"),
            {"center_x": 1, "center_y": 2, "center_z": 3, "size_x": 4, "size_y": 5, "size_z": 6},
            seed=123,
            cpu=1,
        )

        self.assertIn("--cpu", cmd)
        self.assertEqual(cmd[cmd.index("--cpu") + 1], "1")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "123")

    def test_ligand_model1_coords_parses_model_and_plain_pdbqt(self) -> None:
        atom = "ATOM      1  C   LIG     1       1.000   2.000   3.000  1.00  0.00           C\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            modeled = Path(tmpdir) / "modeled.pdbqt"
            plain = Path(tmpdir) / "plain.pdbqt"
            modeled.write_text("MODEL 1\n" + atom + "ENDMDL\n", encoding="utf-8")
            plain.write_text(atom, encoding="utf-8")

            self.assertEqual(docking_utils.ligand_model1_coords(modeled).shape, (1, 3))
            self.assertEqual(docking_utils.ligand_model1_coords(plain).shape, (1, 3))

    def test_selection_reason_records_top_threshold_and_bms_override(self) -> None:
        row = pd.Series({
            "library_id": "LIB100_0001",
            "wt_affinity": -8.5,
            "bms_inclusion_mode": "clinical_hypothesis_override",
        })

        reason = SCREEN.selection_reason(row, {"LIB100_0001"}, -8.0)

        self.assertEqual(reason, "top20_wt;wt_threshold;prespecified_bms")

    def test_wt_worker_preserves_bms_inclusion_mode_for_carry_forward(self) -> None:
        original_dock_one = SCREEN.dock_one
        try:
            SCREEN.dock_one = lambda *_args, **_kwargs: {"affinity": -6.0, "output": "/tmp/out.pdbqt"}
            row = {
                "library_id": "LIB100_0042",
                "name": "BMS-986205",
                "canonical_smiles": "CC",
                "smiles": "CC",
                "inchikey": "TEST",
                "bms_inclusion_mode": "clinical_hypothesis_override",
            }

            result = SCREEN._wt_worker((row, {"library_sha256": "abc"}, "/tmp", False))

            self.assertEqual(result["bms_inclusion_mode"], "clinical_hypothesis_override")
            reason = SCREEN.selection_reason(pd.Series(result), set(), -8.0)
            self.assertEqual(reason, "prespecified_bms")
        finally:
            SCREEN.dock_one = original_dock_one

    def test_run_wt_raises_after_writing_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            library = tmp / "library.csv"
            pd.DataFrame([
                {
                    "library_id": "LIB100_0001",
                    "name": "A",
                    "canonical_smiles": "CC",
                    "smiles": "CC",
                    "inchikey": "A",
                    "bms_inclusion_mode": "",
                }
            ]).to_csv(library, index=False)
            original_results = SCREEN.STAGE3_RESULTS
            original_worker = SCREEN._wt_worker
            original_executor = SCREEN.ProcessPoolExecutor
            original_as_completed = SCREEN.as_completed
            try:
                SCREEN.STAGE3_RESULTS = tmp / "results"
                SCREEN._wt_worker = lambda _payload: {"library_id": "LIB100_0001", "name": "A", "error": "boom"}

                class _Future:
                    def result(self):
                        return {"library_id": "LIB100_0001", "name": "A", "error": "boom"}

                class _Executor:
                    def __init__(self, max_workers):
                        self.max_workers = max_workers

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        return False

                    def submit(self, *_args, **_kwargs):
                        return _Future()

                SCREEN.ProcessPoolExecutor = _Executor
                SCREEN.as_completed = lambda futures: futures
                args = types.SimpleNamespace(
                    library=library,
                    workers=1,
                    resume=False,
                    wt_threshold=-8.0,
                    mode="wt",
                )

                with self.assertRaisesRegex(RuntimeError, "WT screening had 1 failed"):
                    SCREEN.run_wt(args, {
                        "library_sha256": "abc",
                        "vina_path": "/tmp/vina",
                        "vina_version": "test",
                        "seed": 42,
                    })

                self.assertTrue((SCREEN.STAGE3_RESULTS / "library_100_wt_failures.csv").exists())
            finally:
                SCREEN.STAGE3_RESULTS = original_results
                SCREEN._wt_worker = original_worker
                SCREEN.ProcessPoolExecutor = original_executor
                SCREEN.as_completed = original_as_completed

    def test_stable_library_id_format(self) -> None:
        self.assertEqual(SELECT_LOCKED.stable_library_id(1), "LIB100_0001")
        self.assertEqual(SELECT_LOCKED.stable_library_id(100), "LIB100_0100")

    def test_dockability_rejects_multifragment_and_metals(self) -> None:
        from rdkit import Chem

        self.assertEqual(
            SELECT_LOCKED.dockability_failure_reason(Chem, Chem.MolFromSmiles("CC.CC")),
            "multi_fragment_structure",
        )
        self.assertEqual(
            SELECT_LOCKED.dockability_failure_reason(Chem, Chem.MolFromSmiles("CC[Ti]")),
            "unsupported_docking_elements:Ti",
        )
        self.assertIsNone(
            SELECT_LOCKED.dockability_failure_reason(Chem, Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")),
        )

    def test_validate_workers_caps_to_cpu_minus_one(self) -> None:
        self.assertEqual(docking_utils.validate_workers(14, cpu_count_value=16), 14)
        self.assertEqual(docking_utils.validate_workers(28, cpu_count_value=16), 15)
        with self.assertRaises(ValueError):
            docking_utils.validate_workers(0, cpu_count_value=16)


if __name__ == "__main__":
    unittest.main()
