#!/usr/bin/env python3
"""Revision analyses for the 50 ns direct campaign. One entry point."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_COMMON = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from docking_utils import (  # noqa: E402
    build_vina_command,
    load_docking_box,
    openmm_subprocess_env,
)
from md_eval.stage7_success_criteria import block_average, split_window_drift  # noqa: E402


CASES = ("A_ERDRP_WT", "B_ERDRP_MUT", "C_BMS_WT", "D_BMS_MUT")
REPLICATES = (1, 2, 3, 4, 5)
TOTAL_STEPS = 25_000_000
PRODUCTION_SCRIPT = (
    Path(__file__).resolve().parents[1] / "Stage 7" / "run_production_50ns_direct.py"
)
CPU_MODULES = (
    "md_eval.analyze_backbone_rmsd",
    "md_eval.analyze_ligand_rmsd",
    "md_eval.analyze_rmsf",
    "md_eval.analyze_pca",
    "md_eval.analyze_hbond",
    "md_eval.analyze_contacts",
    "md_eval.analyze_plif",
    "md_eval.analyze_pocket_volume",
)
GPU_MODULES = (
    "md_eval.analyze_mmgbsa",
    "md_eval.analyze_decomp",
)
POCKET_BACKBONE_SELECTION = "backbone and (around 10.0 resname UNK)"
COMPARISON_PAIRS = {
    "wt_mutant": (("A_ERDRP_WT", "B_ERDRP_MUT"), ("C_BMS_WT", "D_BMS_MUT")),
    "ligand": (("A_ERDRP_WT", "C_BMS_WT"), ("B_ERDRP_MUT", "D_BMS_MUT")),
}
ENTROPY_SPACING_NS = 0.1
ENTROPY_SUBSAMPLE = 100
_KB = 0.001987204
_BETA = 1.0 / (_KB * 300.0)
BMS_WT_CASE = "C_BMS_WT"
ENTROPY_STRIDE = 25
CONTACT_CUTOFF = 4.5
VINA_SEEDS = (42, 0, 1)
BACKBONE_NAMES = {"N", "CA", "C", "O"}
WORDING = "These summaries are descriptive and are not an affinity claim.\n"
_FRAME_CACHE: dict = {}


def build_md_tasks() -> list[dict]:
    return [{"case": case, "replicate": replicate} for case in CASES for replicate in REPLICATES]


def _is_complete(payload: dict | None) -> bool:
    if not payload:
        return False
    return int(payload.get("final_step", -1)) == TOTAL_STEPS


def plan_md_queue(tasks: list[dict], sentinels: dict, n_gpus: int, parallel: int | None) -> dict:
    pending = [
        task
        for task in tasks
        if not _is_complete(sentinels.get((task["case"], task["replicate"])))
    ]
    concurrency = 4 * n_gpus if parallel is None else parallel
    concurrency = max(1, concurrency)
    batches = []
    for start in range(0, len(pending), concurrency):
        batch = []
        for offset, task in enumerate(pending[start:start + concurrency]):
            assigned = dict(task)
            assigned["cuda_visible_devices"] = str(offset % n_gpus)
            batch.append(assigned)
        batches.append(batch)
    devices_shared = False
    for batch in batches:
        counts: dict[str, int] = {}
        for task in batch:
            device = task["cuda_visible_devices"]
            counts[device] = counts.get(device, 0) + 1
        if any(count > 1 for count in counts.values()):
            devices_shared = True
    return {
        "concurrency": concurrency,
        "batches": batches,
        "pending": [task for batch in batches for task in batch],
        "use_mps": devices_shared,
    }


def production_command(
    task: dict,
    resume: bool,
    results_root: Path | None = None,
    stage6_results: Path | None = None,
    s3_options: dict | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(PRODUCTION_SCRIPT),
        "--case",
        task["case"],
        "--replicate",
        str(task["replicate"]),
    ]
    if results_root is not None:
        command.extend(["--results-root", str(results_root)])
    if stage6_results is not None:
        command.extend(["--stage6-results", str(stage6_results)])
    if resume:
        command.append("--resume")
    opts = s3_options or {}
    if opts.get("archive_s3"):
        command.append("--archive-s3")
    if opts.get("s3_bucket"):
        command.extend(["--s3-bucket", str(opts["s3_bucket"])])
    if opts.get("s3_prefix"):
        command.extend(["--s3-prefix", str(opts["s3_prefix"])])
    if opts.get("s3_profile"):
        command.extend(["--s3-profile", str(opts["s3_profile"])])
    if opts.get("s3_region"):
        command.extend(["--s3-region", str(opts["s3_region"])])
    if opts.get("s3_delete_chk_above_mb") is not None:
        command.extend(["--s3-delete-chk-above-mb", str(opts["s3_delete_chk_above_mb"])])
    if opts.get("max_steps") is not None:
        command.extend(["--max-steps", str(opts["max_steps"])])
    return command


def _s3_options_from_args(args: argparse.Namespace) -> dict:
    return {
        "archive_s3": getattr(args, "archive_s3", False),
        "s3_bucket": getattr(args, "s3_bucket", None),
        "s3_prefix": getattr(args, "s3_prefix", None),
        "s3_profile": getattr(args, "s3_profile", None),
        "s3_region": getattr(args, "s3_region", None),
        "s3_delete_chk_above_mb": getattr(args, "s3_delete_chk_above_mb", None),
        "max_steps": getattr(args, "max_steps", None),
    }


def plan_cpu_wave(cases, cpu_count: int, parallel: int | None) -> dict:
    n_cases = len(tuple(cases))
    prolif_n_jobs = max(1, cpu_count // n_cases)
    concurrent = n_cases if parallel is None else min(n_cases, parallel)
    return {
        "modules": list(CPU_MODULES),
        "concurrent_cases": concurrent,
        "prolif_n_jobs": prolif_n_jobs,
        "env": {"STAGE7_PLIF_N_JOBS": str(prolif_n_jobs)},
    }


def plan_gpu_wave(cases, n_gpus: int, parallel: int | None) -> dict:
    slots_per_device = 2
    concurrency = slots_per_device * n_gpus if parallel is None else parallel
    tasks = [
        {"case": case, "replicate": replicate}
        for case in cases
        for replicate in REPLICATES
    ]
    batches = [tasks[start:start + concurrency] for start in range(0, len(tasks), concurrency)]
    return {
        "modules": list(GPU_MODULES),
        "slots_per_device": slots_per_device,
        "concurrency": concurrency,
        "start_method": "spawn",
        "tasks": tasks,
        "batches": batches,
    }


def oom_retry(task: dict, message: str) -> list[dict]:
    if "out of memory" in message.lower():
        return [task]
    return []


def eval_environment(project_root: Path) -> dict[str, str]:
    root = Path(project_root)
    return {
        "STAGE7_DIRECT_RESULTS_ROOT": str(root / "stage7_50ns_direct" / "results"),
        "STAGE7_EVAL_WORK_DIR": str(root / "Stages" / "revision" / "results" / "eval"),
    }


def _batch_shares_gpu(batch: list[dict]) -> bool:
    counts: dict[str, int] = {}
    for task in batch:
        device = str(task["cuda_visible_devices"])
        counts[device] = counts.get(device, 0) + 1
    return any(count > 1 for count in counts.values())


def _call_runner(runner, task, command, log_path):
    try:
        return runner(task, command, log_path)
    except TypeError:
        return runner(task, command)


def _wait_result(handle):
    if not hasattr(handle, "wait"):
        if isinstance(handle, tuple):
            return handle
        return handle, ""
    result = handle.wait()
    if isinstance(result, tuple):
        return result
    return result, ""


class _PopenHandle:
    def __init__(self, process, log_handle):
        self.process = process
        self.log_handle = log_handle

    def wait(self):
        code = self.process.wait()
        self.log_handle.close()
        return code


def _popen_md(task, command, log_path):
    env = openmm_subprocess_env(1, base=os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = str(task["cuda_visible_devices"])
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    return _PopenHandle(process, log_handle)


def run_md_queue(
    tasks,
    sentinels,
    n_gpus,
    parallel,
    runner,
    resume: bool = False,
    results_root: Path | None = None,
    stage6_results: Path | None = None,
    log_dir: Path | None = None,
    start_mps=None,
    stop_mps=None,
    s3_options: dict | None = None,
) -> dict:
    plan = plan_md_queue(tasks, sentinels, n_gpus, parallel)
    failed = []
    for batch in plan["batches"]:
        mps_on = False
        if _batch_shares_gpu(batch) and start_mps is not None:
            mps_on = bool(start_mps())
        waiting = []
        for task in batch:
            command = production_command(
                task, resume, results_root, stage6_results, s3_options=s3_options
            )
            log_path = None
            if log_dir is not None:
                log_path = Path(log_dir) / f"{task['case']}_rep{task['replicate']}.log"
            started = _call_runner(runner, task, command, log_path)
            if hasattr(started, "wait"):
                waiting.append((task, started))
            elif started != 0:
                failed.append((task["case"], task["replicate"]))
        for task, handle in waiting:
            code, _log = _wait_result(handle)
            if code != 0:
                failed.append((task["case"], task["replicate"]))
        if mps_on and stop_mps is not None:
            stop_mps()
    return {"failed": failed, "plan": plan}


def pocket_rmsd_diagnostic(csv_path: Path) -> dict:
    series = []
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            series.append(float(row["pocket_backbone_rmsd_A"]))
    return {
        "selection": POCKET_BACKBONE_SELECTION,
        "series": series,
        "drift": split_window_drift(series),
    }


def block_sign_check(left: list[float], right: list[float], block_size: int) -> dict:
    diff = [a - b for a, b in zip(left, right)]
    blocks = block_average(diff, block_size)
    mid = len(blocks) // 2
    first = sum(block["mean"] for block in blocks[:mid]) / mid
    second = sum(block["mean"] for block in blocks[mid:]) / (len(blocks) - mid)
    return {"same_sign": (first >= 0) == (second >= 0), "first_mean": first, "second_mean": second}


def _endpoint_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _pooled_within_sd(left: list[float], right: list[float]) -> float:
    groups = [np.asarray(values, dtype=float) for values in (left, right)]
    total = sum(group.size for group in groups)
    variance = sum(group.size * float(group.var()) for group in groups) / total
    return float(np.sqrt(variance))


def replicate_spread(endpoints: dict[str, list[float]]) -> dict:
    cases = {case: _endpoint_stats(values) for case, values in endpoints.items()}
    gaps = {}
    for pairs in COMPARISON_PAIRS.values():
        for left, right in pairs:
            if left not in endpoints or right not in endpoints:
                continue
            gaps[(left, right)] = {
                "gap": abs(cases[left]["mean"] - cases[right]["mean"]),
                "pooled_sd": _pooled_within_sd(endpoints[left], endpoints[right]),
            }
    return {"cases": cases, "gaps": gaps, "callout": BMS_WT_CASE}


def _extreme_frame_share(values) -> float:
    arr = np.asarray(values, dtype=float)
    centered = _BETA * (arr - arr.mean())
    weights = np.exp(centered - centered.max())
    return float(weights.max() / weights.sum())


def entropy_diagnostic(delta_e: list[float]) -> dict:
    arr = np.asarray(delta_e, dtype=float)
    count = min(ENTROPY_SUBSAMPLE, arr.size)
    indices = np.linspace(0, arr.size - 1, count).astype(int)
    return {
        "n_frames": int(arr.size),
        "spacing_ns": ENTROPY_SPACING_NS,
        "sd_before_logsumexp": float(arr.std()),
        "delta_e": [float(value) for value in arr],
        "extreme_frame_share": _extreme_frame_share(arr),
        "stride": ENTROPY_STRIDE,
        "subsample_n": int(count),
        "subsample_share": _extreme_frame_share(arr[indices]),
    }


def _read_dg(path: Path) -> list[float]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [float(row["dG_bind_kcal_mol"]) for row in csv.DictReader(handle)]


def _mmgbsa_file(directory: Path, case: str, replicate: int) -> Path:
    root = Path(directory)
    nested = root / "mmgbsa_50ns_direct" / f"{case}_mmgbsa_rep{replicate}.csv"
    flat = root / f"{case}_mmgbsa_rep{replicate}.csv"
    if nested.is_file():
        return nested
    return flat


def diagnostics_from_directory(directory: Path) -> dict:
    root = Path(directory)
    endpoints: dict[str, list[float]] = {}
    entropy = {}
    for case in CASES:
        means = []
        for replicate in REPLICATES:
            path = _mmgbsa_file(root, case, replicate)
            if not path.is_file():
                continue
            series = _read_dg(path)
            means.append(float(np.mean(series)))
            entropy[f"{case}_rep{replicate}"] = entropy_diagnostic(series)
        if means:
            endpoints[case] = means
    return {"spread": replicate_spread(endpoints), "entropy": entropy}


HIGHLIGHT_RESNAMES = {
    875: frozenset({"HIS"}),
    806: frozenset({"TRP"}),
    730: frozenset({"TRP", "ALA"}),
}


def contact_occupancy(residues: list[dict], frames: list[dict]) -> dict[int, list[int]]:
    present = {int(row["resid"]): row["resname"] for row in residues}
    missing = [resid for resid in HIGHLIGHT_RESNAMES if resid not in present]
    if missing:
        raise ValueError(f"topology missing residue IDs: {missing}")
    for resid, allowed in HIGHLIGHT_RESNAMES.items():
        if present[resid] not in allowed:
            raise ValueError(f"residue {resid} resname {present[resid]} is not in {sorted(allowed)}")
    return {
        resid: [int(frame[resid]) for frame in frames]
        for resid in HIGHLIGHT_RESNAMES
    }


WEIGHT_LEVELS = (0.0, 0.5, 1.0, 2.0)
SEVERE_PENALTIES = (0.0, 1.0, 2.0, 4.0)
MODERATE_PENALTIES = (0.0, 0.5, 1.0)
SEVERE_CUTOFFS = (2.0, 2.5, 3.0)
MODERATE_CUTOFFS = (3.0, 3.2, 3.5)
BASE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
BASE_SEVERE_PENALTY = 2.0
BASE_MODERATE_PENALTY = 0.5
BASE_SEVERE_CUTOFF = 2.5
BASE_MODERATE_CUTOFF = 3.2
NINE_VXV_LIGANDS = ("ERDRP-0519", "BMS-986205")


def _ingredients(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["wt_affinity"],
                row["mut_affinity"],
                max(float(row["delta_affinity"]), 0.0),
                abs(float(row["delta_dist"])),
                row["pose_center_shift"],
            ]
            for row in rows
        ],
        dtype=float,
    )


def frozen_scale(rows: list[dict]) -> dict[str, np.ndarray]:
    matrix = _ingredients(rows)
    return {"mean": matrix.mean(axis=0), "sd": matrix.std(axis=0)}


def _ghost_penalty(distance: float, severe_cutoff: float, moderate_cutoff: float, severe_penalty: float, moderate_penalty: float) -> float:
    if distance < severe_cutoff:
        return severe_penalty
    if distance < moderate_cutoff:
        return moderate_penalty
    return 0.0


def _score_with_scale(rows, scale, weights, severe_cutoff, moderate_cutoff, severe_penalty, moderate_penalty):
    matrix = _ingredients(rows)
    standardized = np.zeros_like(matrix)
    for column in range(matrix.shape[1]):
        if scale["sd"][column] == 0:
            continue
        standardized[:, column] = -(matrix[:, column] - scale["mean"][column]) / scale["sd"][column]
    scores = standardized @ np.asarray(weights, dtype=float)
    for index, row in enumerate(rows):
        scores[index] -= _ghost_penalty(
            float(row["ghost_clash_dist"]),
            severe_cutoff,
            moderate_cutoff,
            severe_penalty,
            moderate_penalty,
        )
    order = sorted(range(len(rows)), key=lambda index: (-scores[index], rows[index]["name"]))
    ranks = [0] * len(rows)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return scores, ranks


def project_on_frozen_scale(library_rows: list[dict], ligand_rows: list[dict]) -> list[dict]:
    scale = frozen_scale(library_rows)
    scores, _ranks = _score_with_scale(
        ligand_rows,
        scale,
        BASE_WEIGHTS,
        BASE_SEVERE_CUTOFF,
        BASE_MODERATE_CUTOFF,
        BASE_SEVERE_PENALTY,
        BASE_MODERATE_PENALTY,
    )
    projected = []
    for row, score in zip(ligand_rows, scores):
        item = dict(row)
        item["mutation_composite_score"] = float(score)
        projected.append(item)
    return projected


def nine_vxv_jobs() -> list[dict]:
    return [
        {"ligand": ligand, "state": state}
        for ligand in NINE_VXV_LIGANDS
        for state in ("WT", "W730A")
    ]


def require_trp_before_mutation(resname: str) -> str:
    if resname != "TRP":
        raise ValueError(f"9VXV W730 site is {resname}; mutagenesis to ALA requires TRP")
    return resname


def _spearman(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0:
        return 1.0
    return float((a * b).sum() / denom)


def _setting_row(rows, scale, grid, weights, severe_penalty, moderate_penalty, severe_cutoff, moderate_cutoff):
    _scores, ranks = _score_with_scale(
        rows, scale, weights, severe_cutoff, moderate_cutoff, severe_penalty, moderate_penalty,
    )
    bms_index = next(index for index, row in enumerate(rows) if row["name"] == "BMS-986205")
    above = [index for index, rank in enumerate(ranks) if rank < ranks[bms_index]]
    positive = all(float(rows[index]["delta_affinity"]) > 0 for index in above)
    return {
        "grid": grid,
        "weights": tuple(weights),
        "severe_penalty": severe_penalty,
        "moderate_penalty": moderate_penalty,
        "severe_cutoff": severe_cutoff,
        "moderate_cutoff": moderate_cutoff,
        "bms_rank": ranks[bms_index],
        "spearman": _spearman([row["rank_mutation_aware"] for row in rows], ranks),
        "delta_affinity_positive_above_bms": positive,
    }


def sweep_composite(rows: list[dict]) -> list[dict]:
    scale = frozen_scale(rows)
    settings = []
    for weights in itertools.product(WEIGHT_LEVELS, repeat=5):
        settings.append(_setting_row(
            rows, scale, "weights", weights,
            BASE_SEVERE_PENALTY, BASE_MODERATE_PENALTY, BASE_SEVERE_CUTOFF, BASE_MODERATE_CUTOFF,
        ))
    for severe_penalty, moderate_penalty in itertools.product(SEVERE_PENALTIES, MODERATE_PENALTIES):
        settings.append(_setting_row(
            rows, scale, "penalties", BASE_WEIGHTS,
            severe_penalty, moderate_penalty, BASE_SEVERE_CUTOFF, BASE_MODERATE_CUTOFF,
        ))
    for severe_cutoff, moderate_cutoff in itertools.product(SEVERE_CUTOFFS, MODERATE_CUTOFFS):
        settings.append(_setting_row(
            rows, scale, "cutoffs", BASE_WEIGHTS,
            BASE_SEVERE_PENALTY, BASE_MODERATE_PENALTY, severe_cutoff, moderate_cutoff,
        ))
    return settings


FIGURE_GROUPS = ("convergence", "replicate_spread", "entropy", "9vxv", "sweep")


def methods_note() -> str:
    return "\n".join([
        "# 50 ns revision methods",
        "",
        "Production is 50 ns: TOTAL_STEPS = 25_000_000 at 2 fs, tier 50ns_direct.",
        'Langevin seed is sha256("{case}_rep{N}").',
        "Bootstrap is Stage 6 equilibrated_state_xml only.",
        "MD queue default concurrency is 4 jobs per GPU under NVIDIA MPS.",
        "GPU energy workers use 2 slots per device and the spawn start method.",
        "Interaction entropy uses 0.1 ns spacing (MM-GBSA stride 25), plus a 100-frame even subsample.",
        "9VXV docks ERDRP-0519 and BMS-986205 only (WT and W730A). No library redock and no MD.",
        "The site must be TRP before mutagenesis to ALA.",
        "Composite sweep, each grid separate:",
        "- descriptor weights, each in {0, 0.5, 1, 2}, baseline all 1s, with severe penalty 2, moderate penalty 0.5, and cutoffs 2.5 Å / 3.2 Å",
        "- severe ghost penalties in {0, 1, 2, 4} and moderate ghost penalties in {0, 0.5, 1}, at cutoffs 2.5 Å / 3.2 Å",
        "- threshold check: severe cutoff in {2.0, 2.5, 3.0} Å and moderate cutoff in {3.0, 3.2, 3.5} Å",
        "",
    ])


def _series_svg(values) -> str:
    tokens = [str(value) for value in values]
    points = " ".join(f"{index},{token}" for index, token in enumerate(tokens))
    return (
        "<svg xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline points='{points}'/><text>{' '.join(tokens)}</text></svg>\n"
    )


def write_deliverables(out_dir: Path, sources: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    series = sources or {}
    (out / "figure_groups.csv").write_text(
        "group\n" + "\n".join(FIGURE_GROUPS) + "\n",
        encoding="utf-8",
    )
    for group in FIGURE_GROUPS:
        values = series.get(group, [])
        (out / f"{group}.svg").write_text(_series_svg(values), encoding="utf-8")
        (out / f"{group}.csv").write_text(
            "value\n" + "\n".join(str(value) for value in values) + "\n",
            encoding="utf-8",
        )
    (out / "methods_note.md").write_text(methods_note(), encoding="utf-8")


def _csv_column(path: Path, column: str) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[column] for row in csv.DictReader(handle) if row.get(column) not in (None, "")]


def _walk_numbers(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _walk_numbers(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_numbers(value)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        yield str(payload)


def load_deliverable_series(project_root: Path) -> dict[str, list]:
    root = Path(project_root)
    results = root / "Stages" / "revision" / "results"
    diag = results / "diagnostics"
    spread_means = []
    spread_path = diag / "spread.json"
    if spread_path.is_file():
        spread = json.loads(spread_path.read_text(encoding="utf-8"))
        for stats in spread.get("cases", {}).values():
            if "mean" in stats:
                spread_means.append(str(stats["mean"]))
    entropy_values = []
    entropy_path = diag / "entropy.json"
    if entropy_path.is_file():
        entropy = json.loads(entropy_path.read_text(encoding="utf-8"))
        for payload in entropy.values():
            if isinstance(payload, dict) and payload.get("delta_e"):
                entropy_values = [str(value) for value in payload["delta_e"]]
                break
    vxv_values = []
    comparison_path = results / "9vxv" / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        vxv_values = list(_walk_numbers(comparison.get("ligand_rmsd", {})))
    return {
        "convergence": _csv_column(diag / "pocket_rmsd.csv", "pocket_backbone_rmsd_A"),
        "replicate_spread": spread_means,
        "entropy": entropy_values,
        "9vxv": vxv_values,
        "sweep": _csv_column(results / "sweep" / "sweep.csv", "bms_rank"),
    }


_STEP_ATTRS = (
    ("eval", "eval"),
    ("diagnostics", "diagnostics"),
    ("vxv", "9vxv"),
    ("sweep", "sweep"),
    ("deliverables", "deliverables"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="50 ns revision analyses")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--9vxv", action="store_true", dest="vxv")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--deliverables", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--archive-s3",
        action="store_true",
        help="Upload production.dcd to S3 after each completed replicate.",
    )
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--s3-profile", default=None)
    parser.add_argument("--s3-region", default=None)
    parser.add_argument("--s3-delete-chk-above-mb", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def selected_steps(args: argparse.Namespace) -> list[str]:
    steps = ["production"] if args.production else []
    labels = [label for _attr, label in _STEP_ATTRS]
    if args.all:
        return steps + labels
    for attr, label in _STEP_ATTRS:
        if getattr(args, attr):
            steps.append(label)
    return steps


def _chunks(items, size: int):
    items = list(items)
    step = max(1, size)
    return [items[start:start + step] for start in range(0, len(items), step)]


def _launch_and_wait(tasks, worker):
    waiting = [(task, worker(task)) for task in tasks]
    return [(task, _wait_result(handle)) for task, handle in waiting]


def run_cpu_wave(cases, cpu_count: int, parallel: int | None, worker, env: dict | None = None) -> dict:
    plan = plan_cpu_wave(cases, cpu_count, parallel)
    task_env = {**(env or {}), **plan["env"]}
    for batch in _chunks(cases, plan["concurrent_cases"]):
        tasks = []
        for case in batch:
            script = ["import runpy, sys"]
            for module in plan["modules"]:
                script.append(f"sys.argv = ['{module}', '--case', '{case}']")
                script.append(f"runpy.run_module('{module}', run_name='__main__')")
            tasks.append({
                "case": case,
                "wave": "cpu",
                "module": "cpu",
                "modules": list(plan["modules"]),
                "env": task_env,
                "command": [sys.executable, "-c", "\n".join(script)],
            })
        _launch_and_wait(tasks, worker)
    return plan


def _gpu_task(module: str, task: dict, device: str, env: dict, attempt: int = 1) -> dict:
    item = dict(task)
    item.update({
        "module": module,
        "cuda_visible_devices": device,
        "start_method": "spawn",
        "attempt": attempt,
        "env": dict(env),
        "command": [
            sys.executable, "-m", module, "--case", task["case"], "--replicate", str(task["replicate"]),
        ],
    })
    return item


def run_gpu_wave(cases, n_gpus: int, parallel: int | None, worker, env: dict | None = None) -> dict:
    plan = plan_gpu_wave(cases, n_gpus, parallel)
    multiprocessing.get_context(plan["start_method"])
    task_env = dict(env or {})
    for module in plan["modules"]:
        for batch in plan["batches"]:
            tasks = [
                _gpu_task(module, task, str(offset % n_gpus), task_env)
                for offset, task in enumerate(batch)
            ]
            for task, (code, log) in _launch_and_wait(tasks, worker):
                if code != 0 and oom_retry(task, log):
                    _launch_and_wait([
                        _gpu_task(module, task, task["cuda_visible_devices"], task_env, attempt=2)
                    ], worker)
    return plan


def run_eval(cpu_count: int, n_gpus: int, parallel: int | None, worker, env: dict | None = None) -> None:
    run_cpu_wave(CASES, cpu_count, parallel, worker, env)
    run_gpu_wave(CASES, n_gpus, parallel, worker, env)
    plan = plan_cpu_wave(CASES, cpu_count, parallel)
    entropy_env = dict(env or {})
    for batch in _chunks(CASES, plan["concurrent_cases"]):
        tasks = []
        for case in batch:
            tasks.append({
                "case": case,
                "wave": "cpu",
                "module": "md_eval.analyze_interaction_entropy",
                "env": {**entropy_env, **plan["env"]},
                "command": [sys.executable, "-m", "md_eval.analyze_interaction_entropy", "--case", case],
            })
        _launch_and_wait(tasks, worker)


def _read_time_dg(path: Path):
    times = []
    values = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time_ns"]))
            values.append(float(row["dG_bind_kcal_mol"]))
    return times, values


def _pocket_indices(frame: list[dict]) -> list[int]:
    ligand = [np.asarray(atom["xyz"], dtype=float) for atom in frame if atom["resname"] == "UNK"]
    chosen = []
    for index, atom in enumerate(frame):
        if atom["name"] not in BACKBONE_NAMES or atom["resname"] == "UNK":
            continue
        xyz = np.asarray(atom["xyz"], dtype=float)
        if ligand and any(float(np.linalg.norm(xyz - other)) <= 10.0 for other in ligand):
            chosen.append(index)
    return chosen


def pocket_series_from_frames(frames: list) -> list[float]:
    if not frames:
        return []
    indices = _pocket_indices(frames[0])
    if not indices:
        return [0.0 for _frame in frames]
    reference = np.vstack([np.asarray(frames[0][index]["xyz"], dtype=float) for index in indices])
    series = []
    for frame in frames:
        xyz = np.vstack([np.asarray(frame[index]["xyz"], dtype=float) for index in indices])
        series.append(float(np.sqrt(((xyz - reference) ** 2).sum() / len(indices))))
    return series


def occupancy_from_frames(frames: list) -> dict[int, list[int]]:
    residues = []
    seen = set()
    for atom in frames[0]:
        if atom["resname"] == "UNK":
            continue
        resid = int(atom["resid"])
        if resid in seen:
            continue
        seen.add(resid)
        residues.append({"resid": resid, "resname": atom["resname"]})
    rows = []
    for frame in frames:
        ligand = [
            np.asarray(atom["xyz"], dtype=float)
            for atom in frame
            if atom["resname"] == "UNK" and not str(atom["name"]).startswith("H")
        ]
        flags = {}
        for resid in HIGHLIGHT_RESNAMES:
            heavy = [
                np.asarray(atom["xyz"], dtype=float)
                for atom in frame
                if int(atom["resid"]) == resid and not str(atom["name"]).startswith("H")
            ]
            hit = 0
            for xyz in heavy:
                if any(float(np.linalg.norm(xyz - other)) <= CONTACT_CUTOFF for other in ligand):
                    hit = 1
                    break
            flags[resid] = hit
        rows.append(flags)
    return contact_occupancy(residues, rows)


def occupancy_from_plif(path: Path) -> dict[int, list[int]]:
    residues = {}
    grouped: dict[int, dict[int, int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            resid = int(row["resid"])
            residues[resid] = row["resname"]
            grouped.setdefault(int(row["frame"]), {})[resid] = int(row["contact"])
    residue_rows = [{"resid": resid, "resname": name} for resid, name in sorted(residues.items())]
    frames = [grouped[frame] for frame in sorted(grouped)]
    return contact_occupancy(residue_rows, frames)


def _block_size_for_times(times: list[float]) -> int:
    if len(times) < 2:
        return 1
    spacing = float(times[1] - times[0])
    if spacing <= 0:
        return 1
    return max(1, int(round(1.0 / spacing)))


def _mean_columns(columns: list[list[float]]) -> list[float]:
    width = min(len(column) for column in columns)
    stacked = np.vstack([np.asarray(column[:width], dtype=float) for column in columns])
    return [float(value) for value in stacked.mean(axis=0)]


def _summarize_case(case: str) -> dict:
    eval_dir = Path(os.environ["REVISION_DIAG_EVAL"])
    cached = _FRAME_CACHE.get(case, {})
    pocket = {}
    occupancy = {}
    entropy = {}
    series = {}
    endpoints = []
    error = None
    try:
        for replicate in REPLICATES:
            frames = cached.get(replicate)
            if frames:
                series_values = pocket_series_from_frames(frames)
                pocket[str(replicate)] = {
                    "selection": POCKET_BACKBONE_SELECTION,
                    "series": series_values,
                    **split_window_drift(series_values),
                }
                occupancy[str(replicate)] = {
                    str(resid): flags for resid, flags in occupancy_from_frames(frames).items()
                }
            else:
                plif = eval_dir / "plif_50ns_direct" / f"{case}_plif_frames_rep{replicate}.csv"
                if plif.is_file():
                    occupancy[str(replicate)] = {
                        str(resid): flags for resid, flags in occupancy_from_plif(plif).items()
                    }
            path = _mmgbsa_file(eval_dir, case, replicate)
            if not path.is_file():
                continue
            times, values = _read_time_dg(path)
            series[str(replicate)] = {"time_ns": times, "dg": values}
            endpoints.append(float(np.mean(values)))
            report = entropy_diagnostic(values)
            entropy[f"{case}_rep{replicate}"] = report
        if not occupancy:
            raise ValueError(f"topology missing residue IDs: {list(HIGHLIGHT_RESNAMES)}")
    except ValueError as exc:
        error = str(exc)
    decomp_path = eval_dir / "decomp_50ns_direct" / f"{case}_decomp_timeseries.csv"
    decomp = decomp_path.read_text(encoding="utf-8") if decomp_path.is_file() else ""
    return {
        "case": case,
        "pocket": pocket,
        "occupancy": occupancy,
        "entropy": entropy,
        "series": series,
        "endpoints": endpoints,
        "decomp": decomp,
        "error": error,
    }


def write_diagnostics(project_root: Path, frame_source=None, pool_factory=None, parallel: int | None = None) -> int:
    global _FRAME_CACHE
    repo = Path(project_root)
    eval_dir = Path(eval_environment(repo)["STAGE7_EVAL_WORK_DIR"])
    os.environ["REVISION_DIAG_EVAL"] = str(eval_dir)
    _FRAME_CACHE = {}
    if frame_source is not None:
        for case in CASES:
            _FRAME_CACHE[case] = {replicate: frame_source(case, replicate) for replicate in REPLICATES}
    workers = len(CASES) if parallel is None else min(len(CASES), parallel)
    factory = pool_factory or ProcessPoolExecutor
    with factory(max_workers=workers) as pool:
        parts = list(pool.map(_summarize_case, list(CASES)))
    errors = [part["error"] for part in parts if part["error"]]
    out = repo / "Stages" / "revision" / "results" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    if errors:
        (out / "errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        return 1
    pocket_lines = ["case,replicate,time_index,pocket_backbone_rmsd_A,selection"]
    drift = {}
    occupancy = {}
    entropy = {}
    endpoints = {}
    series = {}
    overlay = ["case,decomp"]
    for part in parts:
        case = part["case"]
        drift[case] = part["pocket"]
        occupancy[case] = part["occupancy"]
        entropy.update(part["entropy"])
        if part["endpoints"]:
            endpoints[case] = part["endpoints"]
        series[case] = part["series"]
        for replicate, payload in part["pocket"].items():
            for index, value in enumerate(payload["series"]):
                pocket_lines.append(
                    f"{case},{replicate},{index},{value:.4f},{payload['selection']}"
                )
        if part["decomp"]:
            overlay.append(f"{case},{part['decomp'].replace(chr(10), ' | ')}")
    spread = replicate_spread(endpoints)
    blocks = {}
    for pairs in COMPARISON_PAIRS.values():
        for left, right in pairs:
            if left not in series or right not in series:
                continue
            left_cols = [payload["dg"] for payload in series[left].values()]
            right_cols = [payload["dg"] for payload in series[right].values()]
            if not left_cols or not right_cols:
                continue
            left_mean = _mean_columns(left_cols)
            right_mean = _mean_columns(right_cols)
            times = next(iter(series[left].values()))["time_ns"]
            blocks[f"{left}|{right}"] = block_sign_check(
                left_mean, right_mean, _block_size_for_times(times),
            )
    (out / "pocket_rmsd.csv").write_text("\n".join(pocket_lines) + "\n", encoding="utf-8")
    (out / "pocket_drift.json").write_text(json.dumps(drift) + "\n", encoding="utf-8")
    (out / "spread.json").write_text(json.dumps({
        "callout": spread["callout"],
        "cases": spread["cases"],
        "gaps": {f"{left}|{right}": gap for (left, right), gap in spread["gaps"].items()},
        "wording": WORDING.strip(),
    }, indent=2) + "\n", encoding="utf-8")
    (out / "block_sign.json").write_text(json.dumps(blocks) + "\n", encoding="utf-8")
    (out / "entropy.json").write_text(json.dumps(entropy) + "\n", encoding="utf-8")
    (out / "occupancy.json").write_text(json.dumps(occupancy) + "\n", encoding="utf-8")
    (out / "decomp_overlay.csv").write_text("\n".join(overlay) + "\n", encoding="utf-8")
    (out / "wording.txt").write_text(WORDING, encoding="utf-8")
    return 0


def clean_structure_records(records: list[dict]) -> list[dict]:
    if not records:
        return []
    chains = [record["chain"] for record in records]
    chain = "A" if "A" in chains else chains[0]
    return [
        record for record in records
        if record.get("chain") == chain and record.get("het", " ") == " "
    ]


def map_w730(reference: list[dict], query: list[dict]) -> dict:
    ref = clean_structure_records(reference)
    cleaned = clean_structure_records(query)
    ref_index = next(index for index, record in enumerate(ref) if int(record["resid"]) == 730)
    start = max(0, ref_index - 1)
    motif = [record["resname"] for record in ref[start:ref_index + 2]]
    names = [record["resname"] for record in cleaned]
    best = None
    for offset in range(0, len(names) - len(motif) + 1):
        window = names[offset:offset + len(motif)]
        score = sum(left == right for left, right in zip(window, motif))
        if best is None or score > best[0]:
            best = (score, offset)
    if best is None:
        raise ValueError("9VXV sequence has no window for 9KNZ Trp730")
    site = cleaned[best[1] + (ref_index - start)]
    require_trp_before_mutation(site["resname"])
    return site


def _kabsch(mobile, target):
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _singular, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def ligand_rmsd_after_pocket_fit(reference_ligand, mobile_ligand, reference_ca, mobile_ca) -> float:
    rotation, translation = _kabsch(mobile_ca, reference_ca)
    moved = np.asarray(mobile_ligand, dtype=float) @ rotation + translation
    delta = moved - np.asarray(reference_ligand, dtype=float)
    return float(np.sqrt((delta * delta).sum(axis=1).mean()))


def nine_vxv_seed_jobs(box: dict) -> list[dict]:
    jobs = []
    for ligand in NINE_VXV_LIGANDS:
        for state in ("WT", "W730A"):
            for seed in VINA_SEEDS:
                command = build_vina_command(
                    Path("vina"),
                    Path(f"9VXV_{state}.pdbqt"),
                    Path(f"{ligand}.pdbqt"),
                    Path(f"{ligand}_{state}_seed{seed}.pdbqt"),
                    box,
                    seed=seed,
                )
                jobs.append({"ligand": ligand, "state": state, "seed": seed, "command": command})
    return jobs


def _library_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("wt_affinity", "mut_affinity", "delta_affinity", "delta_dist", "pose_center_shift", "ghost_clash_dist", "rank_mutation_aware"):
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def run_nine_vxv(project_root: Path, structure_source, dock_runner, mutate) -> int:
    repo = Path(project_root)
    library_path = repo / "Stages" / "Stage 3" / "results" / "library_100_mutation_ranked.csv"
    try:
        source = structure_source() if structure_source is not None else {}
        cleaned = clean_structure_records(source.get("vxv_records", []))
        site = map_w730(source.get("knz_records", []), source.get("vxv_records", []))
        if mutate is not None:
            mutate(site)
        box = load_docking_box()
        jobs = nine_vxv_seed_jobs(box)
        if dock_runner is None:
            raise RuntimeError("9VXV docking runner is not available")
        waiting = [(job, dock_runner(job)) for job in jobs]
        poses = []
        for job, handle in waiting:
            if not hasattr(handle, "wait"):
                raise RuntimeError("9VXV dock runner did not return a pose")
            poses.append((job, handle.wait()))
    except (ValueError, RuntimeError, StopIteration) as exc:
        print(exc)
        return 1
    by_complex: dict[tuple, dict] = {}
    for job, pose in poses:
        key = (job["ligand"], job["state"])
        current = by_complex.get(key)
        if current is None or pose["affinity"] < current["affinity"]:
            by_complex[key] = pose
    references = source.get("reference_ligand", {})
    ligand_rmsd = {}
    vxv_rows = []
    ingredients = []
    library = _library_rows(library_path) if library_path.is_file() else []
    library_by_name = {row["name"]: row for row in library}
    for ligand in NINE_VXV_LIGANDS:
        ligand_rmsd[ligand] = {}
        wt = by_complex[(ligand, "WT")]
        mut = by_complex[(ligand, "W730A")]
        for state, pose in (("WT", wt), ("W730A", mut)):
            reference = references.get(ligand)
            if reference is None:
                continue
            ligand_rmsd[ligand][state] = ligand_rmsd_after_pocket_fit(
                reference, pose["ligand_xyz"], pose["pocket_ca"], pose["pocket_ca"],
            )
        wt_center = np.asarray(wt["ligand_xyz"], dtype=float).mean(axis=0)
        mut_center = np.asarray(mut["ligand_xyz"], dtype=float).mean(axis=0)
        row = {
            "name": ligand,
            "wt_affinity": float(wt["affinity"]),
            "mut_affinity": float(mut["affinity"]),
            "delta_affinity": float(mut["affinity"] - wt["affinity"]),
            "delta_dist": float(mut["site_dist"] - wt["site_dist"]),
            "pose_center_shift": float(np.linalg.norm(wt_center - mut_center)),
            "ghost_clash_dist": float(wt["ghost_clash_dist"]),
        }
        vxv_rows.append(row)
        ingredients.append({"ligand": ligand, "source": "9VXV", **{key: row[key] for key in row if key != "name"}})
        published = library_by_name.get(ligand)
        if published is not None:
            ingredients.append({
                "ligand": ligand,
                "source": "9KNZ",
                "wt_affinity": published["wt_affinity"],
                "mut_affinity": published["mut_affinity"],
                "delta_affinity": published["delta_affinity"],
                "delta_dist": published["delta_dist"],
                "pose_center_shift": published["pose_center_shift"],
                "ghost_clash_dist": published["ghost_clash_dist"],
            })
    projected = project_on_frozen_scale(library, vxv_rows) if library else []
    out = repo / "Stages" / "revision" / "results" / "9vxv"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cleaned_resnames": [record["resname"] for record in cleaned],
        "site_resname": site["resname"],
        "ingredients": ingredients,
        "ligand_rmsd": ligand_rmsd,
        "projected": [
            {"name": row["name"], "mutation_composite_score": row["mutation_composite_score"]}
            for row in projected
        ],
    }
    (out / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _write_sweep(ranked: Path, out_dir: Path) -> None:
    rows = _library_rows(ranked)
    settings = sweep_composite(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(settings[0].keys()))
        writer.writeheader()
        for setting in settings:
            payload = dict(setting)
            payload["weights"] = " ".join(str(value) for value in setting["weights"])
            writer.writerow(payload)


def _eval_popen(task):
    env = os.environ.copy()
    env.update(task.get("env") or {})
    if task.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(task["cuda_visible_devices"])
    process = subprocess.Popen(
        task["command"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    class _Handle:
        def wait(self):
            output, _unused = process.communicate()
            return process.returncode, output or ""

    return _Handle()


def eval_jobs(cpu_count: int, n_gpus: int, parallel: int | None, project_root: Path) -> list[dict]:
    env = eval_environment(project_root)
    cpu = plan_cpu_wave(CASES, cpu_count, parallel)
    gpu = plan_gpu_wave(CASES, n_gpus, parallel)
    jobs = []
    for module in cpu["modules"]:
        jobs.append({
            "module": module,
            "wave": "cpu",
            "env": {**env, **cpu["env"]},
            "start_method": None,
        })
    for module in gpu["modules"]:
        jobs.append({
            "module": module,
            "wave": "gpu",
            "env": dict(env),
            "start_method": gpu["start_method"],
        })
    return jobs


def load_sentinels(results_root: Path) -> dict:
    found = {}
    root = Path(results_root)
    for task in build_md_tasks():
        path = root / task["case"] / f"replicate_{task['replicate']}" / "TASK_COMPLETE"
        if not path.is_file():
            continue
        try:
            found[(task["case"], task["replicate"])] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            found[(task["case"], task["replicate"])] = {}
    return found


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _gpu_count() -> int:
    try:
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return 1
    if probe.returncode != 0:
        return 1
    count = sum(1 for line in probe.stdout.splitlines() if line.startswith("GPU"))
    return max(1, count)


def _start_mps() -> bool:
    try:
        probe = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if probe.returncode != 0:
        return False
    subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True)
    return True


def _stop_mps() -> None:
    subprocess.run(["bash", "-c", "echo quit | nvidia-cuda-mps-control"], capture_output=True)


def main(
    argv: list[str] | None = None,
    production_runner=None,
    n_gpus: int | None = None,
    sentinels: dict | None = None,
    project_root: Path | None = None,
    eval_worker=None,
    frame_source=None,
    pool_factory=None,
    structure_source=None,
    dock_runner=None,
    mutate=None,
) -> int:
    args = parse_args(argv)
    repo = Path(project_root) if project_root is not None else _repo_root()
    results_root = repo / "stage7_50ns_direct" / "results"
    stage6_results = repo / "Stages" / "Stage 6" / "results"
    exit_code = 0
    for step in selected_steps(args):
        if step == "production":
            gpus = _gpu_count() if n_gpus is None else n_gpus
            payloads = load_sentinels(results_root) if sentinels is None else sentinels
            result = run_md_queue(
                build_md_tasks(),
                payloads,
                gpus,
                args.parallel,
                production_runner or _popen_md,
                resume=args.resume,
                results_root=results_root,
                stage6_results=stage6_results,
                log_dir=repo / "Stages" / "revision" / "results" / "production_logs",
                start_mps=_start_mps if production_runner is None else None,
                stop_mps=_stop_mps if production_runner is None else None,
                s3_options=_s3_options_from_args(args),
            )
            for case, replicate in result["failed"]:
                print(f"failed: {case} replicate {replicate}")
            if result["failed"]:
                exit_code = 1
        elif step == "eval":
            gpu_count = _gpu_count() if n_gpus is None else n_gpus
            run_eval(
                os.cpu_count() or 1,
                gpu_count,
                args.parallel,
                eval_worker or _eval_popen,
                eval_environment(repo),
            )
        elif step == "sweep":
            ranked = repo / "Stages" / "Stage 3" / "results" / "library_100_mutation_ranked.csv"
            if not ranked.is_file():
                print(f"missing ranked library: {ranked}")
                exit_code = 1
                continue
            _write_sweep(ranked, repo / "Stages" / "revision" / "results" / "sweep")
        elif step == "diagnostics":
            status = write_diagnostics(repo, frame_source, pool_factory, args.parallel)
            if status:
                exit_code = status
        elif step == "9vxv":
            status = run_nine_vxv(repo, structure_source, dock_runner, mutate)
            if status:
                exit_code = status
        elif step == "deliverables":
            write_deliverables(
                repo / "Stages" / "revision" / "results" / "deliverables",
                load_deliverable_series(repo),
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
