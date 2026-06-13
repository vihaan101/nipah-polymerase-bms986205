from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
STAGE5_DIR = TESTS_DIR.parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECT = _load_module(STAGE5_DIR / "select_focused_validation_candidates.py", "select_focused_validation_candidates")
RUN = _load_module(STAGE5_DIR / "run_focused_five_seed_validation.py", "run_focused_five_seed_validation")


class Focused100Stage5Tests(unittest.TestCase):
    def test_select_candidates_adds_bms_when_absent_from_top_n(self) -> None:
        ranked = pd.DataFrame([
            {"library_id": "LIB100_0001", "name": "A", "rank_mutation_aware": 1},
            {"library_id": "LIB100_0002", "name": "B", "rank_mutation_aware": 2},
            {"library_id": "LIB100_0003", "name": "C", "rank_mutation_aware": 3},
            {"library_id": "LIB100_0004", "name": "BMS-986205", "rank_mutation_aware": 8},
        ])

        selected = SELECT.select_candidates(ranked, top_n=2)

        self.assertEqual(list(selected["library_id"]), ["LIB100_0001", "LIB100_0002", "LIB100_0004"])
        bms = selected[selected["name"] == "BMS-986205"].iloc[0]
        self.assertEqual(bms["candidate_role"], "prespecified_bms")

    def test_select_candidates_does_not_duplicate_bms(self) -> None:
        ranked = pd.DataFrame([
            {"library_id": "LIB100_0001", "name": "BMS-986205", "rank_mutation_aware": 1},
            {"library_id": "LIB100_0002", "name": "A", "rank_mutation_aware": 2},
        ])

        selected = SELECT.select_candidates(ranked, top_n=2)

        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(selected["name"] == "BMS-986205"), 1)

    def test_summarize_seed_matrix_computes_delta_error_and_rank(self) -> None:
        matrix = pd.DataFrame([
            {"library_id": "LIB100_0001", "name": "A", "receptor_state": "WT", "seed": 1, "affinity": -7.0},
            {"library_id": "LIB100_0001", "name": "A", "receptor_state": "WT", "seed": 2, "affinity": -7.2},
            {"library_id": "LIB100_0001", "name": "A", "receptor_state": "W730A", "seed": 1, "affinity": -7.4},
            {"library_id": "LIB100_0001", "name": "A", "receptor_state": "W730A", "seed": 2, "affinity": -7.6},
            {"library_id": "COMPARATOR_ERDRP", "name": "ERDRP-0519", "receptor_state": "WT", "seed": 1, "affinity": -6.0},
            {"library_id": "COMPARATOR_ERDRP", "name": "ERDRP-0519", "receptor_state": "WT", "seed": 2, "affinity": -6.2},
            {"library_id": "COMPARATOR_ERDRP", "name": "ERDRP-0519", "receptor_state": "W730A", "seed": 1, "affinity": -6.1},
            {"library_id": "COMPARATOR_ERDRP", "name": "ERDRP-0519", "receptor_state": "W730A", "seed": 2, "affinity": -6.3},
        ])

        summary = RUN.summarize(matrix)
        a_wt = summary[(summary["name"] == "A") & (summary["receptor_state"] == "WT")].iloc[0]
        a_mut = summary[(summary["name"] == "A") & (summary["receptor_state"] == "W730A")].iloc[0]

        self.assertAlmostEqual(a_wt["mean_affinity"], -7.1)
        self.assertAlmostEqual(a_wt["delta_affinity"], -0.4)
        self.assertGreater(a_wt["propagated_error"], 0)
        self.assertEqual(a_wt["rank_stability"], 1)
        self.assertEqual(a_mut["rank_stability"], 1)


if __name__ == "__main__":
    unittest.main()
