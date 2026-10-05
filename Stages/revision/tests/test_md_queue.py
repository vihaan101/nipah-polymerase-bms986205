"""MD queue plans 20 fake tasks and does not start OpenMM."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


REVISION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_DIR))

import run_revision  # noqa: E402


class MdQueueTests(unittest.TestCase):
    def test_queue_has_twenty_case_replicate_pairs(self) -> None:
        tasks = run_revision.build_md_tasks()
        pairs = [(task["case"], task["replicate"]) for task in tasks]
        self.assertEqual(len(pairs), 20)
        self.assertEqual(len(set(pairs)), 20)
        self.assertEqual(pairs.count(("C_BMS_WT", 5)), 1)

    def test_complete_sentinel_is_skipped_and_partial_is_kept(self) -> None:
        tasks = run_revision.build_md_tasks()
        done = {("A_ERDRP_WT", 1): {"final_step": 25_000_000}}
        plan = run_revision.plan_md_queue(tasks, done, n_gpus=1, parallel=20)
        pending = [(task["case"], task["replicate"]) for task in plan["pending"]]
        self.assertEqual(len(pending), 19)
        self.assertNotIn(("A_ERDRP_WT", 1), pending)

        partial = {("A_ERDRP_WT", 1): {"final_step": 1_000}}
        kept = run_revision.plan_md_queue(tasks, partial, n_gpus=1, parallel=20)
        kept_pairs = [(task["case"], task["replicate"]) for task in kept["pending"]]
        self.assertIn(("A_ERDRP_WT", 1), kept_pairs)
        self.assertEqual(len(kept_pairs), 20)

    def test_parallel_cap_assigns_cuda_devices(self) -> None:
        tasks = run_revision.build_md_tasks()
        plan = run_revision.plan_md_queue(tasks, {}, n_gpus=2, parallel=4)
        self.assertEqual(plan["concurrency"], 4)
        self.assertEqual([len(batch) for batch in plan["batches"]], [4, 4, 4, 4, 4])
        self.assertTrue(plan["use_mps"])
        for batch in plan["batches"]:
            devices = {task["cuda_visible_devices"] for task in batch}
            self.assertEqual(devices, {"0", "1"})

    def test_default_concurrency_is_four_per_gpu(self) -> None:
        plan = run_revision.plan_md_queue(run_revision.build_md_tasks(), {}, n_gpus=2, parallel=None)
        self.assertEqual(plan["concurrency"], 8)
        self.assertTrue(plan["use_mps"])

    def test_failed_pairs_are_listed_and_later_tasks_still_run(self) -> None:
        calls = []

        def runner(task, command):
            calls.append((task["case"], task["replicate"], "--resume" in command))
            if task == {"case": "B_ERDRP_MUT", "replicate": 2, "cuda_visible_devices": "0"}:
                return 1
            return 0

        result = run_revision.run_md_queue(
            run_revision.build_md_tasks(),
            {},
            n_gpus=1,
            parallel=2,
            runner=runner,
            resume=True,
        )

        self.assertEqual(len(calls), 20)
        self.assertTrue(all(resumed for _case, _rep, resumed in calls))
        self.assertEqual(result["failed"], [("B_ERDRP_MUT", 2)])
        fresh = run_revision.production_command(
            {"case": "A_ERDRP_WT", "replicate": 1},
            resume=False,
        )
        self.assertNotIn("--resume", fresh)
        self.assertIn("run_production_50ns_direct.py", " ".join(fresh))

    def test_each_batch_overlaps_concurrency_tasks(self) -> None:
        inflight = []
        peaks = []

        def runner(task, command, log_path=None):
            inflight.append((task["case"], task["replicate"]))
            peaks.append(len(inflight))

            class _Handle:
                def wait(self):
                    inflight.remove((task["case"], task["replicate"]))
                    return 0

            return _Handle()

        plan = run_revision.plan_md_queue(run_revision.build_md_tasks(), {}, n_gpus=1, parallel=4)
        run_revision.run_md_queue(
            run_revision.build_md_tasks(),
            {},
            n_gpus=1,
            parallel=4,
            runner=runner,
            log_dir=Path("/tmp/revision-md-logs"),
        )
        self.assertEqual(plan["concurrency"], 4)
        self.assertEqual([peak for peak in peaks if peak == 4], [4, 4, 4, 4, 4])

    def test_each_job_has_its_own_log_path(self) -> None:
        logs = []

        def runner(task, command, log_path=None):
            logs.append(log_path)

            class _Handle:
                def wait(self):
                    return 0

            return _Handle()

        run_revision.run_md_queue(
            run_revision.build_md_tasks(),
            {},
            n_gpus=1,
            parallel=2,
            runner=runner,
            log_dir=Path("/tmp/revision-md-logs"),
        )
        names = [Path(path).name for path in logs]
        self.assertEqual(len(names), 20)
        self.assertIn("C_BMS_WT_rep5.log", names)

    def test_mps_starts_only_when_a_batch_shares_a_gpu(self) -> None:
        events = []

        def runner(task, command, log_path=None):
            class _Handle:
                def wait(self):
                    return 0

            return _Handle()

        def start():
            events.append("start")
            return True

        def stop():
            events.append("stop")

        run_revision.run_md_queue(
            run_revision.build_md_tasks(),
            {},
            n_gpus=2,
            parallel=1,
            runner=runner,
            start_mps=start,
            stop_mps=stop,
        )
        self.assertEqual(events, [])

        run_revision.run_md_queue(
            run_revision.build_md_tasks(),
            {},
            n_gpus=1,
            parallel=2,
            runner=runner,
            start_mps=start,
            stop_mps=stop,
        )
        self.assertEqual(events, ["start", "stop"] * 10)

    def test_main_prints_failed_pairs_and_returns_nonzero(self) -> None:
        import io
        from contextlib import redirect_stdout

        def runner(task, command, log_path=None):
            if (task["case"], task["replicate"]) == ("B_ERDRP_MUT", 2):
                return 1
            return 0

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_revision.main(
                ["--production", "--parallel", "2"],
                production_runner=runner,
                n_gpus=1,
                sentinels={},
            )
        text = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("B_ERDRP_MUT", text)
        self.assertIn("2", text)


if __name__ == "__main__":
    unittest.main()
