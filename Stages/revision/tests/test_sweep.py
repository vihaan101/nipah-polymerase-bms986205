"""Frozen-scale projection and the a priori composite sweep."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_revision  # noqa: E402


def _library():
    return [
        {
            "name": "Other",
            "wt_affinity": -20.0,
            "mut_affinity": -1.0,
            "delta_affinity": 0.0,
            "delta_dist": 0.0,
            "pose_center_shift": 0.0,
            "ghost_clash_dist": 4.0,
            "rank_mutation_aware": 1,
        },
        {
            "name": "BMS-986205",
            "wt_affinity": -1.0,
            "mut_affinity": -20.0,
            "delta_affinity": 0.0,
            "delta_dist": 0.0,
            "pose_center_shift": 0.0,
            "ghost_clash_dist": 4.0,
            "rank_mutation_aware": 2,
        },
        {
            "name": "Filler",
            "wt_affinity": 0.0,
            "mut_affinity": 0.0,
            "delta_affinity": 5.0,
            "delta_dist": 5.0,
            "pose_center_shift": 5.0,
            "ghost_clash_dist": 4.0,
            "rank_mutation_aware": 3,
        },
    ]


class FrozenScaleAndSweepTests(unittest.TestCase):
    def test_projection_uses_library_scale_and_does_not_rerank_it(self) -> None:
        library = [
            {
                "name": "A",
                "wt_affinity": 0.0,
                "mut_affinity": 0.0,
                "delta_affinity": 0.0,
                "delta_dist": 0.0,
                "pose_center_shift": 0.0,
                "ghost_clash_dist": 4.0,
                "rank_mutation_aware": 1,
            },
            {
                "name": "B",
                "wt_affinity": 2.0,
                "mut_affinity": 2.0,
                "delta_affinity": 2.0,
                "delta_dist": 2.0,
                "pose_center_shift": 2.0,
                "ghost_clash_dist": 4.0,
                "rank_mutation_aware": 2,
            },
        ]
        ligand = dict(library[0])
        ligand["name"] = "ERDRP-0519"
        projected = run_revision.project_on_frozen_scale(library, [ligand])
        self.assertEqual([row["rank_mutation_aware"] for row in library], [1, 2])
        self.assertEqual(len(projected), 1)
        self.assertAlmostEqual(projected[0]["mutation_composite_score"], 5.0)
        self.assertEqual(
            [job["ligand"] for job in run_revision.nine_vxv_jobs()],
            ["ERDRP-0519", "ERDRP-0519", "BMS-986205", "BMS-986205"],
        )
        self.assertEqual(run_revision.require_trp_before_mutation("TRP"), "TRP")
        with self.assertRaises(ValueError):
            run_revision.require_trp_before_mutation("ALA")

    def test_weight_sweep_changes_bms_rank(self) -> None:
        settings = run_revision.sweep_composite(_library())
        weights = [row for row in settings if row["grid"] == "weights"]
        penalties = [row for row in settings if row["grid"] == "penalties"]
        cutoffs = [row for row in settings if row["grid"] == "cutoffs"]
        self.assertEqual(len(weights), 4 ** 5)
        self.assertEqual(len(penalties), 4 * 3)
        self.assertEqual(len(cutoffs), 3 * 3)
        by_weight = {row["weights"]: row for row in weights}
        self.assertEqual(by_weight[(1, 1, 1, 1, 1)]["bms_rank"], 1)
        self.assertEqual(by_weight[(2, 1, 1, 1, 1)]["bms_rank"], 2)
        self.assertFalse(by_weight[(2, 1, 1, 1, 1)]["delta_affinity_positive_above_bms"])
        self.assertGreaterEqual(by_weight[(1, 1, 1, 1, 1)]["spearman"], -1.0)

    def test_sweep_fails_when_ranked_csv_is_missing(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = run_revision.main(["--sweep"], project_root=root)
            written = root / "Stages" / "revision" / "results" / "sweep" / "sweep.csv"
            self.assertEqual(code, 1)
            self.assertFalse(written.exists())


def _library_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "name,wt_affinity,mut_affinity,delta_affinity,delta_dist,pose_center_shift,ghost_clash_dist,rank_mutation_aware",
            "ERDRP-0519,-8,-6,2,1,0.5,4,1",
            "BMS-986205,-7,-5,2,1,0.5,4,2",
            "Other,-1,-1,0,0,0,4,3",
            "",
        ]),
        encoding="utf-8",
    )


def _structures(site="TRP"):
    return {
        "vxv_records": [
            {"chain": "A", "het": " ", "resid": 729, "resname": "MET"},
            {"chain": "A", "het": " ", "resid": 730, "resname": site},
            {"chain": "A", "het": " ", "resid": 731, "resname": "GLY"},
            {"chain": "A", "het": "W", "resname": "HOH", "resid": 1},
            {"chain": "B", "het": " ", "resid": 1, "resname": "ALA"},
        ],
        "knz_records": [
            {"chain": "A", "het": " ", "resid": 729, "resname": "MET"},
            {"chain": "A", "het": " ", "resid": 730, "resname": "TRP"},
            {"chain": "A", "het": " ", "resid": 731, "resname": "GLY"},
        ],
        "reference_ligand": {
            "ERDRP-0519": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            "BMS-986205": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        },
    }


class NineVxvCommandTests(unittest.TestCase):
    def test_9vxv_flag_compares_poses_and_projects_on_frozen_scale(self) -> None:
        peaks = []
        inflight = []
        mutated = []
        seen = []

        def dock_runner(job):
            seen.append(job)
            inflight.append(job["seed"])
            peaks.append(len(inflight))

            class _Handle:
                def wait(self):
                    inflight.pop()
                    return {
                        "affinity": -8.0 if job["state"] == "WT" else -6.0,
                        "pocket_ca": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "ligand_xyz": [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
                        "site_dist": 3.0 if job["state"] == "WT" else 5.0,
                        "ghost_clash_dist": 4.0,
                    }

            return _Handle()

        def mutate(site):
            mutated.append(site["resname"])
            return {"resname": "ALA"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "Stages" / "Stage 3" / "results" / "library_100_mutation_ranked.csv"
            _library_csv(library)
            before = library.read_bytes()
            code = run_revision.main(
                ["--9vxv"],
                project_root=root,
                structure_source=lambda: _structures(),
                dock_runner=dock_runner,
                mutate=mutate,
            )
            comparison = json.loads(
                (root / "Stages" / "revision" / "results" / "9vxv" / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(library.read_bytes(), before)
        self.assertEqual(code, 0)
        self.assertEqual(max(peaks), 12)
        self.assertEqual(mutated, ["TRP"])
        self.assertNotIn("HOH", comparison["cleaned_resnames"])
        self.assertNotIn("ALA", comparison["cleaned_resnames"])
        ligands = {row["ligand"] for row in comparison["ingredients"]}
        self.assertEqual(ligands, {"ERDRP-0519", "BMS-986205"})
        sources = {(row["ligand"], row["source"]) for row in comparison["ingredients"]}
        self.assertIn(("BMS-986205", "9KNZ"), sources)
        self.assertIn(("BMS-986205", "9VXV"), sources)
        self.assertIn(("ERDRP-0519", "9VXV"), sources)
        rmsd = comparison["ligand_rmsd"]["ERDRP-0519"]["WT"]
        self.assertAlmostEqual(rmsd, 1.0)
        self.assertEqual(len(comparison["projected"]), 2)
        self.assertIn("mutation_composite_score", comparison["projected"][0])
        command = " ".join(seen[0]["command"])
        self.assertIn("--size_x", command)
        self.assertIn("26", command)
        self.assertEqual(
            {(job["ligand"], job["state"]) for job in seen},
            {("ERDRP-0519", "WT"), ("ERDRP-0519", "W730A"), ("BMS-986205", "WT"), ("BMS-986205", "W730A")},
        )

    def test_9vxv_stops_before_mutation_when_site_is_not_trp(self) -> None:
        mutated = []

        def mutate(site):
            mutated.append(site)
            return {"resname": "ALA"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _library_csv(root / "Stages" / "Stage 3" / "results" / "library_100_mutation_ranked.csv")
            code = run_revision.main(
                ["--9vxv"],
                project_root=root,
                structure_source=lambda: _structures("ALA"),
                dock_runner=lambda job: None,
                mutate=mutate,
            )
        self.assertEqual(code, 1)
        self.assertEqual(mutated, [])


if __name__ == "__main__":
    unittest.main()
