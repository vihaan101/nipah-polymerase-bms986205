"""Diagnostics on synthetic CSVs. No trajectory or GPU."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_revision  # noqa: E402


class DiagnosticTests(unittest.TestCase):
    def test_pocket_rmsd_series_reports_half_drift(self) -> None:
        with self._csv("pocket.csv", ["time_ns", "pocket_backbone_rmsd_A"], [
            ["0", "1"],
            ["1", "1"],
            ["2", "1"],
            ["3", "3"],
        ]) as path:
            result = run_revision.pocket_rmsd_diagnostic(path)

        self.assertEqual(result["selection"], "backbone and (around 10.0 resname UNK)")
        self.assertEqual(result["series"], [1.0, 1.0, 1.0, 3.0])
        self.assertEqual(result["drift"]["first_mean"], 1.0)
        self.assertEqual(result["drift"]["second_mean"], 2.0)
        self.assertEqual(result["drift"]["mean_shift"], 1.0)

    def test_block_sign_matches_when_both_halves_agree(self) -> None:
        same = run_revision.block_sign_check([1, 1, 2, 2], [0, 0, 0, 0], block_size=1)
        flipped = run_revision.block_sign_check([1, 1, -2, -2], [0, 0, 0, 0], block_size=1)
        self.assertTrue(same["same_sign"])
        self.assertFalse(flipped["same_sign"])
        self.assertEqual(
            run_revision.COMPARISON_PAIRS["wt_mutant"],
            (("A_ERDRP_WT", "B_ERDRP_MUT"), ("C_BMS_WT", "D_BMS_MUT")),
        )

    def test_replicate_spread_calls_out_bms_wt(self) -> None:
        spread = run_revision.replicate_spread({
            "C_BMS_WT": [1.0, 2.0, 3.0, 4.0, 5.0],
            "D_BMS_MUT": [3.0, 3.0, 3.0, 3.0, 3.0],
        })
        bms = spread["cases"]["C_BMS_WT"]
        self.assertEqual(bms["mean"], 3.0)
        self.assertEqual(bms["min"], 1.0)
        self.assertEqual(bms["max"], 5.0)
        self.assertAlmostEqual(bms["sd"], 2 ** 0.5)
        gap = spread["gaps"][("C_BMS_WT", "D_BMS_MUT")]
        self.assertEqual(gap["gap"], 0.0)
        self.assertGreater(gap["pooled_sd"], 0.0)
        self.assertEqual(spread["callout"], "C_BMS_WT")

    def test_mean_gap_is_not_folded_into_pooled_sd(self) -> None:
        spread = run_revision.replicate_spread({
            "A_ERDRP_WT": [0.0, 0.0, 0.0, 0.0, 0.0],
            "B_ERDRP_MUT": [10.0, 10.0, 10.0, 10.0, 10.0],
        })
        gap = spread["gaps"][("A_ERDRP_WT", "B_ERDRP_MUT")]
        self.assertEqual(gap["gap"], 10.0)
        self.assertEqual(gap["pooled_sd"], 0.0)

    def test_entropy_extreme_frame_dominates_logsumexp(self) -> None:
        flat = [0.0] * 500
        flat_share = run_revision.entropy_diagnostic(flat)
        self.assertEqual(flat_share["n_frames"], 500)
        self.assertEqual(flat_share["spacing_ns"], 0.1)
        self.assertAlmostEqual(flat_share["extreme_frame_share"], 1 / 500)
        self.assertEqual(flat_share["subsample_n"], 100)

        spiked = [0.0] * 499 + [50.0]
        spiked_share = run_revision.entropy_diagnostic(spiked)
        self.assertGreater(spiked_share["extreme_frame_share"], 0.5)
        self.assertGreater(spiked_share["sd_before_logsumexp"], 0.0)

    def test_directory_summary_uses_replicate_mmgbsa_csvs(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for replicate, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1):
                path = root / f"C_BMS_WT_mmgbsa_rep{replicate}.csv"
                path.write_text(f"time_ns,dG_bind_kcal_mol\n0,{value}\n1,{value}\n", encoding="utf-8")
            summary = run_revision.diagnostics_from_directory(root)
        self.assertEqual(summary["spread"]["callout"], "C_BMS_WT")
        self.assertEqual(summary["spread"]["cases"]["C_BMS_WT"]["mean"], 3.0)

    def _csv(self, name, header, rows):
        import tempfile

        class _Holder:
            def __enter__(self_inner):
                self_inner.tmp = tempfile.TemporaryDirectory()
                path = Path(self_inner.tmp.name) / name
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)
                self_inner.path = path
                return path

            def __exit__(self_inner, *_args):
                self_inner.tmp.cleanup()

        return _Holder()


def _mmgbsa(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["time_ns,dG_bind_kcal_mol"]
    for index, value in enumerate(values):
        lines.append(f"{index * 0.1:.1f},{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frames():
    def frame(shift):
        return [
            {"name": "CA", "resname": "HIS", "resid": 875, "xyz": (0.0, 0.0, shift)},
            {"name": "CA", "resname": "TRP", "resid": 806, "xyz": (30.0, 0.0, 0.0)},
            {"name": "CA", "resname": "ALA", "resid": 730, "xyz": (20.0, 0.0, 0.0)},
            {"name": "C", "resname": "UNK", "resid": 1, "xyz": (0.0, 0.0, 0.0)},
        ]
    return [frame(0.0), frame(0.0), frame(4.0), frame(4.0)]


class DiagnosticsCommandTests(unittest.TestCase):
    def test_diagnostics_flag_writes_reports(self) -> None:
        import tempfile
        submitted = []

        class _Pool:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def map(self, fn, items):
                items = list(items)
                submitted.append((self.max_workers, items))
                return [fn(item) for item in items]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mmgbsa = root / "Stages" / "revision" / "results" / "eval" / "mmgbsa_50ns_direct"
            decomp = root / "Stages" / "revision" / "results" / "eval" / "decomp_50ns_direct"
            decomp.mkdir(parents=True)
            (decomp / "A_ERDRP_WT_decomp_timeseries.csv").write_text(
                "time_ns,resid_875\n0.0,-3.25\n0.1,-3.25\n",
                encoding="utf-8",
            )
            for case in run_revision.CASES:
                for replicate in run_revision.REPLICATES:
                    if case == "A_ERDRP_WT":
                        values = [-5.0] * 10 + [5.0] * 10
                    elif case == "B_ERDRP_MUT":
                        values = [4.0] * 20
                    elif case == "C_BMS_WT":
                        values = [float(replicate)] * 20
                    else:
                        values = [0.0] * 20
                    _mmgbsa(mmgbsa / f"{case}_mmgbsa_rep{replicate}.csv", values)
            code = run_revision.main(
                ["--diagnostics", "--parallel", "2"],
                project_root=root,
                n_gpus=1,
                frame_source=lambda case, replicate: _frames(),
                pool_factory=lambda max_workers: _Pool(max_workers),
            )
            out = root / "Stages" / "revision" / "results" / "diagnostics"
            pocket = (out / "pocket_rmsd.csv").read_text(encoding="utf-8")
            drift = json.loads((out / "pocket_drift.json").read_text(encoding="utf-8"))
            spread = json.loads((out / "spread.json").read_text(encoding="utf-8"))
            blocks = json.loads((out / "block_sign.json").read_text(encoding="utf-8"))
            entropy = json.loads((out / "entropy.json").read_text(encoding="utf-8"))
            occupancy = json.loads((out / "occupancy.json").read_text(encoding="utf-8"))
            overlay = (out / "decomp_overlay.csv").read_text(encoding="utf-8")
            wording = (out / "wording.txt").read_text(encoding="utf-8").lower()
            self.assertEqual(code, 0)
            self.assertIn("backbone and (around 10.0 resname UNK)", pocket)
            self.assertIn("4.0000", pocket)
            self.assertIn("first_mean", drift["A_ERDRP_WT"]["1"])
            self.assertEqual(spread["callout"], "C_BMS_WT")
            self.assertEqual(spread["cases"]["C_BMS_WT"]["mean"], 3.0)
            self.assertEqual(spread["cases"]["C_BMS_WT"]["min"], 1.0)
            self.assertEqual(spread["cases"]["C_BMS_WT"]["max"], 5.0)
            gap = spread["gaps"]["A_ERDRP_WT|B_ERDRP_MUT"]
            self.assertEqual(gap["gap"], 4.0)
            self.assertEqual(gap["pooled_sd"], 0.0)
            self.assertFalse(blocks["A_ERDRP_WT|B_ERDRP_MUT"]["same_sign"])
            self.assertTrue(blocks["C_BMS_WT|D_BMS_MUT"]["same_sign"])
            self.assertIn("A_ERDRP_WT|C_BMS_WT", blocks)
            self.assertIn("B_ERDRP_MUT|D_BMS_MUT", blocks)
            sample = entropy["A_ERDRP_WT_rep1"]
            self.assertEqual(sample["spacing_ns"], 0.1)
            self.assertEqual(sample["stride"], 25)
            self.assertEqual(len(sample["delta_e"]), 20)
            self.assertGreater(sample["sd_before_logsumexp"], 0.0)
            self.assertGreater(sample["extreme_frame_share"], 0.0)
            self.assertEqual(sample["subsample_n"], 20)
            self.assertEqual(occupancy["A_ERDRP_WT"]["1"]["875"], [1, 1, 1, 1])
            self.assertEqual(occupancy["A_ERDRP_WT"]["1"]["806"], [0, 0, 0, 0])
            self.assertIn("-3.25", overlay)
            self.assertIn("descriptive", wording)
            self.assertIn("not an affinity claim", wording)
            self.assertEqual(submitted[0][0], 2)
            self.assertEqual(sorted(submitted[0][1]), sorted(run_revision.CASES))

    def test_diagnostics_fails_when_highlight_residue_is_absent(self) -> None:
        import tempfile

        class _Pool:
            def __init__(self, max_workers):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def map(self, fn, items):
                return [fn(item) for item in items]

        def frames(case, replicate):
            return [[
                {"name": "CA", "resname": "HIS", "resid": 875, "xyz": (0.0, 0.0, 0.0)},
                {"name": "CA", "resname": "TRP", "resid": 806, "xyz": (1.0, 0.0, 0.0)},
                {"name": "C", "resname": "UNK", "resid": 1, "xyz": (0.0, 0.0, 0.0)},
            ]]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = run_revision.main(
                ["--diagnostics"],
                project_root=root,
                frame_source=frames,
                pool_factory=lambda max_workers: _Pool(max_workers),
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
