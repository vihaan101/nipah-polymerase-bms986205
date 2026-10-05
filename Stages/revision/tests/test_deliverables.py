"""Deliverable index and methods note."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_revision  # noqa: E402


class DeliverableTests(unittest.TestCase):
    def test_tables_list_figure_groups_and_methods_note_has_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_revision.write_deliverables(out)
            table = (out / "figure_groups.csv").read_text(encoding="utf-8")
            note = (out / "methods_note.md").read_text(encoding="utf-8")

        for group in ("convergence", "replicate_spread", "entropy", "9vxv", "sweep"):
            self.assertIn(group, table)
        self.assertIn("0.1 ns", note)
        self.assertIn("25_000_000", note)
        self.assertIn("0, 0.5, 1, 2", note)
        self.assertIn("0, 1, 2, 4", note)
        self.assertIn("0, 0.5, 1", note)
        self.assertIn("2.0, 2.5, 3.0", note)
        self.assertIn("3.0, 3.2, 3.5", note)
        self.assertIn("2.5", note)
        self.assertIn("3.2", note)
        lines = note.splitlines()
        weight = next(line for line in lines if "descriptor weights" in line)
        penalty = next(line for line in lines if "severe ghost penalties" in line)
        self.assertIn("2.5", weight)
        self.assertIn("3.2", weight)
        self.assertIn("penalty", weight)
        self.assertIn("2.5", penalty)
        self.assertIn("3.2", penalty)
        self.assertFalse(any(line.strip().startswith("- at the current cutoffs") for line in lines))

    def test_all_does_not_schedule_production(self) -> None:
        steps = run_revision.selected_steps(run_revision.parse_args(["--all", "--parallel", "4"]))
        self.assertEqual(steps, ["eval", "diagnostics", "9vxv", "sweep", "deliverables"])

    def test_combined_flags_keep_pipeline_order(self) -> None:
        steps = run_revision.selected_steps(run_revision.parse_args(["--deliverables", "--production", "--resume"]))
        self.assertEqual(steps, ["production", "deliverables"])
        self.assertTrue(run_revision.parse_args(["--resume"]).resume)

    def test_main_production_uses_the_injected_runner(self) -> None:
        commands = []

        def runner(task, command):
            commands.append(command)
            return 0

        run_revision.main(
            ["--production", "--parallel", "4"],
            production_runner=runner,
            n_gpus=1,
            sentinels={},
        )
        self.assertEqual(len(commands), 20)
        self.assertTrue(all("--resume" not in command for command in commands))
        self.assertTrue(all("run_production_50ns_direct.py" in " ".join(command) for command in commands))

    def test_eval_jobs_are_cpu_wave_then_gpu_wave(self) -> None:
        jobs = run_revision.eval_jobs(cpu_count=8, n_gpus=2, parallel=None, project_root=Path("/repo"))
        names = [job["module"] for job in jobs]
        self.assertLess(names.index("md_eval.analyze_plif"), names.index("md_eval.analyze_mmgbsa"))
        self.assertEqual(jobs[0]["env"]["STAGE7_PLIF_N_JOBS"], "2")
        self.assertEqual(jobs[-1]["env"]["STAGE7_DIRECT_RESULTS_ROOT"], "/repo/stage7_50ns_direct/results")
        self.assertEqual(jobs[-1]["start_method"], "spawn")

    def test_deliverables_embed_series_from_analysis_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diag = root / "Stages" / "revision" / "results" / "diagnostics"
            vxv = root / "Stages" / "revision" / "results" / "9vxv"
            sweep = root / "Stages" / "revision" / "results" / "sweep"
            diag.mkdir(parents=True)
            vxv.mkdir(parents=True)
            sweep.mkdir(parents=True)
            (diag / "pocket_rmsd.csv").write_text(
                "case,replicate,pocket_backbone_rmsd_A\nA_ERDRP_WT,1,8.7500\n",
                encoding="utf-8",
            )
            (diag / "spread.json").write_text(
                '{"cases":{"C_BMS_WT":{"mean":3.5}}}',
                encoding="utf-8",
            )
            (diag / "entropy.json").write_text(
                '{"A_ERDRP_WT_rep1":{"delta_e":[0.25,0.75]}}',
                encoding="utf-8",
            )
            (vxv / "comparison.json").write_text(
                '{"ligand_rmsd":{"ERDRP-0519":{"WT":1.5}}}',
                encoding="utf-8",
            )
            (sweep / "sweep.csv").write_text("bms_rank\n13\n", encoding="utf-8")
            code = run_revision.main(["--deliverables"], project_root=root)
            out = root / "Stages" / "revision" / "results" / "deliverables"
            convergence = (out / "convergence.svg").read_text(encoding="utf-8")
            spread = (out / "replicate_spread.svg").read_text(encoding="utf-8")
            entropy = (out / "entropy.svg").read_text(encoding="utf-8")
            nine = (out / "9vxv.svg").read_text(encoding="utf-8")
            sweep_svg = (out / "sweep.svg").read_text(encoding="utf-8")
            note = (out / "methods_note.md").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertIn("8.7500", convergence)
        self.assertNotIn(">convergence<", convergence)
        self.assertIn("3.5", spread)
        self.assertIn("0.25", entropy)
        self.assertIn("1.5", nine)
        self.assertIn("13", sweep_svg)
        self.assertIn("25_000_000", note)

    def test_all_does_not_start_md_jobs(self) -> None:
        started = []

        def production_runner(task, command, log_path=None):
            started.append((task["case"], task["replicate"]))
            return 0

        def eval_worker(task):
            class _Handle:
                def wait(self):
                    return 0, ""

            return _Handle()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / "Stages" / "Stage 3" / "results" / "library_100_mutation_ranked.csv"
            library.parent.mkdir(parents=True)
            library.write_text(
                "name,wt_affinity,mut_affinity,delta_affinity,delta_dist,pose_center_shift,ghost_clash_dist,rank_mutation_aware\n"
                "BMS-986205,-7,-5,2,1,0.5,4,1\nOther,-1,-1,0,0,0,4,2\n",
                encoding="utf-8",
            )
            run_revision.main(
                ["--all", "--parallel", "2"],
                project_root=root,
                n_gpus=1,
                production_runner=production_runner,
                eval_worker=eval_worker,
                frame_source=lambda case, replicate: [],
                pool_factory=lambda max_workers: _InlinePool(),
                structure_source=lambda: {"vxv_records": [], "knz_records": [], "reference_ligand": {}},
                dock_runner=lambda job: None,
                mutate=lambda site: {"resname": "ALA"},
            )
        self.assertEqual(started, [])


class _InlinePool:
    def __init__(self, max_workers=1):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def map(self, fn, items):
        return [fn(item) for item in items]


if __name__ == "__main__":
    unittest.main()
