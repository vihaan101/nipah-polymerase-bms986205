from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


TESTS_DIR = Path(__file__).resolve().parent
STAGE7_DIR = TESTS_DIR.parent
MODULE_PATH = STAGE7_DIR / "analysis_10ns_direct" / "verify_eval_ready_trajectories.py"


def _load_verifier_module():
    fake_validation = types.ModuleType("stage7_production_validation")
    fake_validation.TIER_TIME_RANGES = {"10ns_direct": (4.0, 10000.0)}

    def _validate_trajectory(*_args, **_kwargs):
        return None

    fake_validation.validate_trajectory = _validate_trajectory
    sys.modules["stage7_production_validation"] = fake_validation

    analysis_dir = str(MODULE_PATH.parent)
    if analysis_dir not in sys.path:
        sys.path.append(analysis_dir)

    spec = importlib.util.spec_from_file_location("stage7_verify_eval_ready_trajectories", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier_module()


class Stage710nsDirectVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_load_stage6_cases = VERIFIER.load_stage6_cases
        self.original_resolve_stage6_path = VERIFIER.resolve_stage6_path
        self.original_inspect_dcd = VERIFIER._inspect_dcd
        self.original_run_contract_validation = VERIFIER._run_contract_validation

    def tearDown(self) -> None:
        VERIFIER.load_stage6_cases = self.original_load_stage6_cases
        VERIFIER.resolve_stage6_path = self.original_resolve_stage6_path
        VERIFIER._inspect_dcd = self.original_inspect_dcd
        VERIFIER._run_contract_validation = self.original_run_contract_validation

    def test_replicate_is_evaluation_ready_when_contract_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            topology = tmp / "stage6" / "A_ERDRP_WT" / "topology.pdb"
            topology.parent.mkdir(parents=True)
            topology.write_text("topology")
            self._write_replicate(results_root, "A_ERDRP_WT", 1, include_log=True)

            VERIFIER.load_stage6_cases = lambda: {
                "A_ERDRP_WT": {"topology_pdb": str(topology)},
            }
            VERIFIER.resolve_stage6_path = lambda value: value
            VERIFIER._inspect_dcd = lambda *_args: {
                "n_frames": 2500,
                "first_time_ps": 4.0,
                "last_time_ps": 10000.0,
                "frames_scanned": 2500,
            }
            VERIFIER._run_contract_validation = lambda *_args: None

            result = VERIFIER.inspect_replicate(results_root, "A_ERDRP_WT", 1, VERIFIER.load_stage6_cases())

            self.assertEqual(result["status"], "evaluation_ready")
            self.assertEqual(result["checks"]["dcd_readable"], "pass")
            self.assertEqual(result["checks"]["frame_count"], "pass")
            self.assertEqual(result["failure_reasons"], [])

    def test_missing_required_dcd_marks_replicate_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            topology = tmp / "stage6" / "A_ERDRP_WT" / "topology.pdb"
            topology.parent.mkdir(parents=True)
            topology.write_text("topology")
            self._write_replicate(results_root, "A_ERDRP_WT", 1, include_dcd=False)

            VERIFIER.load_stage6_cases = lambda: {
                "A_ERDRP_WT": {"topology_pdb": str(topology)},
            }
            VERIFIER.resolve_stage6_path = lambda value: value

            result = VERIFIER.inspect_replicate(results_root, "A_ERDRP_WT", 1, VERIFIER.load_stage6_cases())

            self.assertEqual(result["status"], "not_ready")
            self.assertIn("missing required file: production.dcd", result["failure_reasons"])
            self.assertEqual(result["checks"]["production.dcd_present"], "fail")

    def test_missing_log_is_warning_not_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            topology = tmp / "stage6" / "A_ERDRP_WT" / "topology.pdb"
            topology.parent.mkdir(parents=True)
            topology.write_text("topology")
            self._write_replicate(results_root, "A_ERDRP_WT", 1, include_log=False)

            VERIFIER.load_stage6_cases = lambda: {
                "A_ERDRP_WT": {"topology_pdb": str(topology)},
            }
            VERIFIER.resolve_stage6_path = lambda value: value
            VERIFIER._inspect_dcd = lambda *_args: {
                "n_frames": 2500,
                "first_time_ps": 4.0,
                "last_time_ps": 10000.0,
                "frames_scanned": 2500,
            }
            VERIFIER._run_contract_validation = lambda *_args: None

            result = VERIFIER.inspect_replicate(results_root, "A_ERDRP_WT", 1, VERIFIER.load_stage6_cases())

            self.assertEqual(result["status"], "not_ready")
            self.assertEqual(result["checks"]["production.log_present"], "fail")
            self.assertIn("missing required file: production.log", result["failure_reasons"])

    def test_verify_results_builds_summary_and_ready_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_root = tmp / "results"
            topology = tmp / "stage6" / "A_ERDRP_WT" / "topology.pdb"
            topology.parent.mkdir(parents=True)
            topology.write_text("topology")
            self._write_replicate(results_root, "A_ERDRP_WT", 1, include_log=True)
            self._write_replicate(results_root, "A_ERDRP_WT", 2, include_dcd=False)
            for rep in (3, 4, 5):
                self._write_replicate(results_root, "A_ERDRP_WT", rep, include_log=True)

            VERIFIER.load_stage6_cases = lambda: {
                "A_ERDRP_WT": {"topology_pdb": str(topology)},
            }
            VERIFIER.resolve_stage6_path = lambda value: value

            def _inspect(_topology, dcd_path):
                if "replicate_2" in str(dcd_path):
                    raise AssertionError("replicate_2 should have failed before inspection")
                return {
                    "n_frames": 2500,
                    "first_time_ps": 4.0,
                    "last_time_ps": 10000.0,
                    "frames_scanned": 2500,
                }

            VERIFIER._inspect_dcd = _inspect
            VERIFIER._run_contract_validation = lambda *_args: None

            manifest = VERIFIER.verify_results(results_root, ["A_ERDRP_WT"])

            self.assertEqual(manifest["summary"]["evaluation_ready"], 4)
            self.assertEqual(manifest["summary"]["not_ready"], 1)
            self.assertEqual(
                manifest["evaluation_ready_replicates"]["A_ERDRP_WT"],
                ["replicate_1", "replicate_3", "replicate_4", "replicate_5"],
            )

    def test_markdown_render_includes_status_table(self) -> None:
        manifest = {
            "generated_at": "2026-04-26T00:00:00Z",
            "source_root": "/tmp/results",
            "tier": "10ns_direct",
            "summary": {"evaluation_ready": 1, "not_ready": 1},
            "cases": {
                "A_ERDRP_WT": {
                    "replicate_1": {
                        "status": "evaluation_ready",
                        "metrics": {"n_frames": 2500, "first_time_ps": 4.0, "last_time_ps": 10000.0},
                        "warnings": [],
                        "failure_reasons": [],
                    },
                    "replicate_2": {
                        "status": "not_ready",
                        "metrics": {"n_frames": 100, "first_time_ps": 4.0, "last_time_ps": 400.0},
                        "warnings": [],
                        "failure_reasons": ["bad trajectory"],
                    },
                },
            },
        }

        rendered = VERIFIER._render_markdown(manifest)

        self.assertIn("| Case | Replicate | Status |", rendered)
        self.assertIn("evaluation_ready", rendered)
        self.assertIn("bad trajectory", rendered)

    def test_inspect_dcd_does_not_full_scan_trajectory(self) -> None:
        class _Frame:
            def __init__(self, time: float) -> None:
                self.time = time

        class _Trajectory:
            def __len__(self) -> int:
                return 2500

            def __getitem__(self, idx: int) -> _Frame:
                if idx == 0:
                    return _Frame(4.0)
                if idx == -1:
                    return _Frame(10000.0)
                raise AssertionError(f"unexpected frame access: {idx}")

            def __iter__(self):
                raise AssertionError("full trajectory iteration should not happen")

        class _Universe:
            def __init__(self) -> None:
                self.trajectory = _Trajectory()

        original_robust_universe = VERIFIER.robust_universe
        VERIFIER.robust_universe = lambda *_args, **_kwargs: _Universe()
        try:
            metrics = VERIFIER._inspect_dcd(Path("/tmp/topology.pdb"), Path("/tmp/production.dcd"))
        finally:
            VERIFIER.robust_universe = original_robust_universe

        self.assertEqual(metrics["n_frames"], 2500)
        self.assertEqual(metrics["first_time_ps"], 4.0)
        self.assertEqual(metrics["last_time_ps"], 10000.0)
        self.assertEqual(metrics["frames_scanned"], 2)

    def _write_replicate(
        self,
        results_root: Path,
        case_id: str,
        replicate_id: int,
        *,
        include_dcd: bool = True,
        include_log: bool = True,
    ) -> None:
        rep_dir = results_root / case_id / f"replicate_{replicate_id}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        if include_dcd:
            (rep_dir / "production.dcd").write_text("dcd")
        if include_log:
            (rep_dir / "production.log").write_text("log")
        (rep_dir / "production_final.pdb").write_text("pdb")
        (rep_dir / "production.chk").write_text("chk")
        (rep_dir / "TASK_COMPLETE").write_text(
            json.dumps({"case_id": case_id, "replicate": replicate_id, "final_step": VERIFIER.TOTAL_STEPS}),
        )
        (rep_dir / "_case_manifest.json").write_text(
            json.dumps({"case_id": case_id, "replicate_id": replicate_id, "tier": VERIFIER.TIER_LABEL}),
        )


if __name__ == "__main__":
    unittest.main()
