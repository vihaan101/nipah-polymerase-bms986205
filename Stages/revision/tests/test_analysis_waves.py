"""CPU and GPU analysis waves assign slots without starting a device."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_revision  # noqa: E402


class AnalysisWaveTests(unittest.TestCase):
    def test_cpu_wave_is_one_process_per_case_and_caps_prolif(self) -> None:
        plan = run_revision.plan_cpu_wave(run_revision.CASES, cpu_count=16, parallel=None)
        self.assertEqual(plan["concurrent_cases"], 4)
        self.assertEqual(plan["prolif_n_jobs"], 4)
        self.assertEqual(plan["env"]["STAGE7_PLIF_N_JOBS"], "4")
        self.assertIn("md_eval.analyze_plif", plan["modules"])
        self.assertIn("md_eval.analyze_backbone_rmsd", plan["modules"])
        self.assertNotIn("md_eval.analyze_mmgbsa", plan["modules"])

    def test_prolif_cap_stays_at_least_one(self) -> None:
        plan = run_revision.plan_cpu_wave(run_revision.CASES, cpu_count=3, parallel=None)
        self.assertEqual(plan["prolif_n_jobs"], 1)

    def test_parallel_limits_cpu_cases(self) -> None:
        inflight = []
        peaks = []
        seen = []

        def worker(task):
            inflight.append(task["case"])
            seen.append(task["case"])
            peaks.append(len(inflight))

            class _Handle:
                def wait(self):
                    inflight.remove(task["case"])
                    return 0, ""

            return _Handle()

        run_revision.run_cpu_wave(run_revision.CASES, cpu_count=16, parallel=2, worker=worker)
        self.assertEqual(max(peaks), 2)
        self.assertEqual([peak for peak in peaks if peak == 2], [2, 2])
        self.assertEqual(sorted(seen), sorted(run_revision.CASES))

    def test_gpu_wave_uses_two_slots_per_device(self) -> None:
        inflight = []
        snapshots = []
        contexts = []
        original = run_revision.multiprocessing.get_context

        def get_context(method):
            contexts.append(method)

            class _Ctx:
                def Process(self, *args, **kwargs):
                    raise AssertionError("injected worker should run the batch")

            return _Ctx()

        def worker(task):
            inflight.append(task)
            snapshots.append([dict(item) for item in inflight])

            class _Handle:
                def wait(self):
                    inflight.remove(task)
                    return 0, ""

            return _Handle()

        run_revision.multiprocessing.get_context = get_context
        try:
            run_revision.run_gpu_wave(run_revision.CASES, n_gpus=2, parallel=None, worker=worker)
        finally:
            run_revision.multiprocessing.get_context = original

        full = [snap for snap in snapshots if len(snap) == 4]
        self.assertEqual(len(full), 10)
        for snap in full:
            counts = {}
            for task in snap:
                counts[task["cuda_visible_devices"]] = counts.get(task["cuda_visible_devices"], 0) + 1
                self.assertEqual(task["start_method"], "spawn")
            self.assertEqual(counts, {"0": 2, "1": 2})
        self.assertIn("spawn", contexts)
        self.assertNotIn("fork", contexts)

    def test_parallel_overrides_gpu_slot_cap(self) -> None:
        inflight = []
        peaks = []

        def worker(task):
            inflight.append(task["replicate"])
            peaks.append(len(inflight))

            class _Handle:
                def wait(self):
                    inflight.pop()
                    return 0, ""

            return _Handle()

        run_revision.run_gpu_wave(run_revision.CASES, n_gpus=1, parallel=6, worker=worker)
        self.assertEqual(max(peaks), 6)

    def test_oom_retries_only_that_replicate(self) -> None:
        calls = []

        def worker(task):
            calls.append(dict(task))

            class _Handle:
                def wait(self):
                    oom = (
                        task.get("attempt", 1) == 1
                        and task["case"] == "D_BMS_MUT"
                        and task["replicate"] == 4
                        and task["module"] == "md_eval.analyze_mmgbsa"
                    )
                    if oom:
                        return 1, "CUDA out of memory"
                    return 0, ""

            return _Handle()

        run_revision.run_gpu_wave(run_revision.CASES, n_gpus=2, parallel=None, worker=worker)
        retries = [task for task in calls if task.get("attempt") == 2]
        self.assertEqual(len(retries), 1)
        retry = retries[0]
        self.assertEqual((retry["case"], retry["replicate"], retry["module"]), (
            "D_BMS_MUT", 4, "md_eval.analyze_mmgbsa",
        ))
        command = " ".join(retry["command"])
        self.assertIn("D_BMS_MUT", command)
        self.assertIn("4", command)
        for other in ("A_ERDRP_WT", "B_ERDRP_MUT", "C_BMS_WT"):
            self.assertNotIn(other, command)

    def test_entropy_runs_after_mmgbsa(self) -> None:
        order = []

        def worker(task):
            order.append(task.get("module", "cpu"))

            class _Handle:
                def wait(self):
                    return 0, ""

            return _Handle()

        run_revision.run_eval(cpu_count=8, n_gpus=1, parallel=4, worker=worker)
        mmgbsa = [index for index, name in enumerate(order) if name == "md_eval.analyze_mmgbsa"]
        entropy = [index for index, name in enumerate(order) if name == "md_eval.analyze_interaction_entropy"]
        self.assertTrue(mmgbsa)
        self.assertTrue(entropy)
        self.assertLess(max(mmgbsa), min(entropy))

    def test_eval_environment_points_at_revision_work_dir(self) -> None:
        root = Path("/repo")
        env = run_revision.eval_environment(root)
        self.assertEqual(env["STAGE7_DIRECT_RESULTS_ROOT"], "/repo/stage7_50ns_direct/results")
        self.assertEqual(env["STAGE7_EVAL_WORK_DIR"], "/repo/Stages/revision/results/eval")


if __name__ == "__main__":
    unittest.main()
