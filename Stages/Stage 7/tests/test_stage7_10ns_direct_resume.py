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
MODULE_PATH = STAGE7_DIR / "run_production_10ns_direct_v2.py"


def _load_runner_module():
    fake_openmm = types.ModuleType("openmm")
    fake_openmm.app = types.SimpleNamespace()
    fake_openmm.unit = types.SimpleNamespace(
        kelvin=1,
        bar=1,
        picoseconds=1,
        picosecond=1,
    )
    sys.modules["openmm"] = fake_openmm

    spec = importlib.util.spec_from_file_location("stage7_run_production_10ns_direct_v2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner_module()


class Stage710nsDirectResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mm = RUNNER.mm
        self.original_app = RUNNER.app
        self.original_validate = RUNNER._validate

    def tearDown(self) -> None:
        RUNNER.mm = self.original_mm
        RUNNER.app = self.original_app
        RUNNER._validate = self.original_validate

    def test_completed_sentinel_is_ignored_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = Path(tmpdir) / "TASK_COMPLETE"
            sentinel.write_text(json.dumps({"final_step": RUNNER.TOTAL_STEPS}))

            self.assertIsNone(RUNNER._completed_step_from_sentinel(sentinel, resume=False))
            self.assertEqual(
                RUNNER._completed_step_from_sentinel(sentinel, resume=True),
                RUNNER.TOTAL_STEPS,
            )

    def test_checkpoint_is_ignored_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "production.chk"
            checkpoint.write_bytes(b"checkpoint-bytes")

            self.assertFalse(RUNNER._should_resume_from_checkpoint(checkpoint, resume=False))
            self.assertTrue(RUNNER._should_resume_from_checkpoint(checkpoint, resume=True))

    def test_corrupt_sentinel_does_not_crash_resume_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = Path(tmpdir) / "TASK_COMPLETE"
            sentinel.write_text("{not-json")

            self.assertIsNone(RUNNER._completed_step_from_sentinel(sentinel, resume=True))

    def test_completed_checkpoint_is_revalidated_before_success(self) -> None:
        calls = []
        self._install_fake_openmm(validate_side_effect=lambda *_args: calls.append("validated"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stage6 = tmp / "stage6"
            results = tmp / "results"
            case_dir = results / "A_ERDRP_WT" / "replicate_1"
            case_dir.mkdir(parents=True)
            self._write_stage6_inputs(stage6)
            (case_dir / "production.chk").write_bytes(b"checkpoint")
            (case_dir / "production.dcd").write_bytes(b"dcd")
            (case_dir / "TASK_COMPLETE").write_text(json.dumps({"final_step": RUNNER.TOTAL_STEPS}))

            outputs = RUNNER.run_production(
                "A_ERDRP_WT",
                self._case_data(stage6),
                1,
                results,
                stage6,
                resume=True,
            )

            self.assertEqual(calls, ["validated"])
            self.assertEqual(outputs["production_checkpoint"], str(case_dir / "production.chk"))

    def test_invalid_completed_checkpoint_is_replayed_fresh(self) -> None:
        calls = []

        def fake_validate(*_args):
            calls.append("validate")
            if len(calls) == 1:
                raise RuntimeError("invalid finished trajectory")

        self._install_fake_openmm(validate_side_effect=fake_validate)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stage6 = tmp / "stage6"
            results = tmp / "results"
            case_dir = results / "A_ERDRP_WT" / "replicate_1"
            case_dir.mkdir(parents=True)
            self._write_stage6_inputs(stage6)
            (case_dir / "production.chk").write_bytes(b"stale-checkpoint")
            (case_dir / "production.dcd").write_text("stale-dcd")
            (case_dir / "production.log").write_text("stale-log")
            (case_dir / "production_final.pdb").write_text("stale-pdb")
            (case_dir / "TASK_COMPLETE").write_text(json.dumps({"final_step": RUNNER.TOTAL_STEPS}))

            RUNNER.run_production(
                "A_ERDRP_WT",
                self._case_data(stage6),
                1,
                results,
                stage6,
                resume=True,
            )

            self.assertEqual(calls, ["validate", "validate"])
            self.assertEqual((case_dir / "production.chk").read_text(), "step=5000000")
            self.assertIn("current-step=5000000", (case_dir / "production.dcd").read_text())
            self.assertNotEqual((case_dir / "production_final.pdb").read_text(), "stale-pdb")
            sentinel_payload = json.loads((case_dir / "TASK_COMPLETE").read_text())
            self.assertEqual(sentinel_payload["final_step"], RUNNER.TOTAL_STEPS)

    def test_invalid_loaded_completed_checkpoint_recreates_simulation_before_replay(self) -> None:
        calls = []

        def fake_validate(*_args):
            calls.append("validate")
            if len(calls) == 1:
                raise RuntimeError("invalid completed checkpoint")

        self._install_fake_openmm(validate_side_effect=fake_validate)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            stage6 = tmp / "stage6"
            results = tmp / "results"
            case_dir = results / "A_ERDRP_WT" / "replicate_4"
            case_dir.mkdir(parents=True)
            self._write_stage6_inputs(stage6)
            (case_dir / "production.chk").write_bytes(b"completed-checkpoint")
            (case_dir / "production.dcd").write_text("bad-dcd")
            (case_dir / "production.log").write_text("bad-log")
            (case_dir / "production_final.pdb").write_text("bad-pdb")

            RUNNER.run_production(
                "A_ERDRP_WT",
                self._case_data(stage6),
                4,
                results,
                stage6,
                resume=True,
            )

            self.assertEqual(calls, ["validate", "validate"])
            self.assertEqual((case_dir / "production.chk").read_text(), "step=5000000")
            self.assertIn("current-step=5000000", (case_dir / "production.dcd").read_text())
            self.assertNotIn("10000000", (case_dir / "production.dcd").read_text())
            sentinel_payload = json.loads((case_dir / "TASK_COMPLETE").read_text())
            self.assertEqual(sentinel_payload["final_step"], RUNNER.TOTAL_STEPS)

    def _write_stage6_inputs(self, stage6_dir: Path) -> None:
        stage6_dir.mkdir(parents=True, exist_ok=True)
        (stage6_dir / "system.xml").write_text("fake-system")
        (stage6_dir / "topology.pdb").write_text("fake-topology")
        (stage6_dir / "state.xml").write_text("fake-state")

    def _case_data(self, stage6_dir: Path) -> dict[str, str]:
        return {
            "system_xml": str(stage6_dir / "system.xml"),
            "topology_pdb": str(stage6_dir / "topology.pdb"),
            "equilibrated_state_xml": str(stage6_dir / "state.xml"),
        }

    def _install_fake_openmm(self, validate_side_effect) -> None:
        class FakeSystem:
            def __init__(self) -> None:
                self._forces = []

            def getForces(self):
                return list(self._forces)

            def removeForce(self, _idx: int) -> None:
                pass

            def addForce(self, force) -> None:
                self._forces.append(force)

        class FakeState:
            def getPositions(self):
                return "positions"

            def getVelocities(self):
                return "velocities"

            def getPeriodicBoxVectors(self):
                return ("a", "b", "c")

        class FakeContext:
            def __init__(self) -> None:
                self.positions = None

            def setPositions(self, positions) -> None:
                self.positions = positions

            def setVelocities(self, _velocities) -> None:
                pass

            def setPeriodicBoxVectors(self, *_vectors) -> None:
                pass

            def getState(self, getPositions=False):
                class _ContextState:
                    def getPositions(self_inner):
                        return ["final-positions"]

                return _ContextState()

        class FakeIntegrator:
            def setRandomNumberSeed(self, _seed: int) -> None:
                pass

        class FakeReporter:
            def __init__(self, path: str, append: bool = False, kind: str = "generic") -> None:
                self.path = Path(path)
                self.append = append
                self.kind = kind

        class FakeSimulation:
            def __init__(self, *_args, **_kwargs) -> None:
                self.currentStep = 0
                self.context = FakeContext()
                self.reporters = []
                self.topology = "fake-topology"

            def loadCheckpoint(self, _path: str) -> None:
                self.currentStep = RUNNER.TOTAL_STEPS

            def step(self, steps: int) -> None:
                self.currentStep += steps
                for reporter in self.reporters:
                    reporter.path.parent.mkdir(parents=True, exist_ok=True)
                    mode = "a" if reporter.append and reporter.path.exists() else "w"
                    if reporter.kind == "dcd":
                        with reporter.path.open(mode) as fh:
                            fh.write(f"current-step={self.currentStep}\n")
                    elif reporter.kind == "log":
                        with reporter.path.open(mode) as fh:
                            fh.write(f"step={self.currentStep}\n")

            def saveCheckpoint(self, path: str) -> None:
                Path(path).write_text(f"step={self.currentStep}")

        class FakePDBFile:
            def __init__(self, _path: str) -> None:
                self.topology = "fake-topology"

            @staticmethod
            def writeFile(_topology, _positions, handle) -> None:
                handle.write("fake-final-pdb")

        class FakeXmlSerializer:
            @staticmethod
            def deserialize(payload: str):
                if payload == "fake-system":
                    return FakeSystem()
                return FakeState()

        class FakeMonteCarloBarostat:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

        RUNNER.mm = types.SimpleNamespace(
            XmlSerializer=FakeXmlSerializer,
            CustomExternalForce=type("FakeCustomExternalForce", (), {}),
            MonteCarloBarostat=FakeMonteCarloBarostat,
            LangevinMiddleIntegrator=lambda *_args, **_kwargs: FakeIntegrator(),
            Platform=types.SimpleNamespace(getPlatformByName=lambda name: name),
        )
        RUNNER.app = types.SimpleNamespace(
            PDBFile=FakePDBFile,
            Simulation=FakeSimulation,
            StateDataReporter=lambda path, *_args, append=False, **_kwargs: FakeReporter(path, append=append, kind="log"),
            DCDReporter=lambda path, *_args, append=False, **_kwargs: FakeReporter(path, append=append, kind="dcd"),
            CheckpointReporter=lambda *_args, **_kwargs: FakeReporter("checkpoint"),
        )
        RUNNER._validate = validate_side_effect


if __name__ == "__main__":
    unittest.main()
