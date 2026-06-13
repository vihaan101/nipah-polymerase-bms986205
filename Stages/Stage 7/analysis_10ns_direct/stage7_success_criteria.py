"""
Stage 7 statistical helpers and metric output utilities.
Success criteria / rubric gating removed — all analysis runs descriptively.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STATUS_QC_PASS = "qc_pass"
STATUS_QC_ISSUE = "qc_issue"
STATUS_SUPPORTS = "descriptive"
STATUS_CONCERN = "descriptive"
STATUS_DESCRIPTIVE = "descriptive"
STATUS_NOT_ASSESSED = "descriptive"


def criterion_result(
    criterion_id: str,
    status: str,
    detail: str,
    *,
    metric_value: Any | None = None,
    threshold: Any | None = None,
    threshold_note: str | None = None,
    extra: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "status": STATUS_DESCRIPTIVE,
        "detail": detail,
        "metric_value": metric_value,
        "threshold": None,
        "threshold_note": None,
        "extra": extra or {},
        "limitations": limitations or [],
    }


def make_metric_summary(
    metric_name: str,
    case_id: str,
    summary: dict[str, Any],
    *,
    derived_metrics: dict[str, Any] | None = None,
    interpretation: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "case_id": case_id,
        "summary": summary,
        "derived_metrics": derived_metrics or {},
        "interpretation": interpretation or [],
        "notes": notes or [],
        "limitations": limitations or [],
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def series_summary(values: Any, *, times: Any | None = None) -> dict[str, Any]:
    arr = finite_array(values)
    summary = {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "final": float(arr[-1]) if arr.size else None,
    }
    if times is not None:
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        if times_arr.size:
            summary["time_start"] = float(times_arr[0])
            summary["time_end"] = float(times_arr[-1])
    return summary


def late_window_indices(length: int, fraction: float = 0.25) -> slice:
    if length <= 0:
        return slice(0, 0)
    window = max(2, int(round(length * fraction)))
    return slice(max(0, length - window), length)


def linear_slope(x: Any, y: Any) -> float | None:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return None
    x_fit = x_arr[mask]
    y_fit = y_arr[mask]
    if np.allclose(x_fit, x_fit[0]):
        return None
    slope, _ = np.polyfit(x_fit, y_fit, 1)
    return float(slope)


def rolling_window(values: Any, window_size: int) -> np.ndarray:
    arr = finite_array(values)
    if arr.size == 0:
        return np.array([], dtype=float)
    window_size = max(1, min(int(window_size), arr.size))
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(arr, kernel, mode="valid")


def rolling_stability(values: Any, *, window_fraction: float = 0.1) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size < 3:
        return {"window_size": None, "rolling_mean_range": None, "rolling_std_mean": None}
    window = max(3, int(round(arr.size * window_fraction)))
    mean_trace = rolling_window(arr, window)
    std_trace = np.array([
        arr[i:i + window].std() for i in range(0, arr.size - window + 1)
    ])
    return {
        "window_size": int(window),
        "rolling_mean_range": float(mean_trace.max() - mean_trace.min()) if mean_trace.size else None,
        "rolling_std_mean": float(std_trace.mean()) if std_trace.size else None,
    }


def late_window_metrics(times: Any, values: Any, *, fraction: float = 0.25) -> dict[str, Any]:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if values_arr.size == 0 or times_arr.size == 0:
        return {"n_points": 0}
    idx = late_window_indices(min(values_arr.size, times_arr.size), fraction=fraction)
    late_values = values_arr[idx]
    late_times = times_arr[idx]
    return {
        "n_points": int(late_values.size),
        "mean": float(np.nanmean(late_values)) if late_values.size else None,
        "std": float(np.nanstd(late_values)) if late_values.size else None,
        "slope_per_ns": linear_slope(late_times, late_values),
        "range": float(np.nanmax(late_values) - np.nanmin(late_values)) if late_values.size else None,
    }


def split_window_drift(values: Any) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size < 4:
        return {"first_mean": None, "second_mean": None, "mean_shift": None}
    half = arr.size // 2
    first = arr[:half]
    second = arr[half:]
    return {
        "first_mean": float(first.mean()),
        "second_mean": float(second.mean()),
        "mean_shift": float(second.mean() - first.mean()),
        "pooled_std": float(np.sqrt((first.var() + second.var()) / 2.0)),
    }


def block_average(values: Any, block_size: int) -> list[dict[str, Any]]:
    arr = finite_array(values)
    block_size = max(1, int(block_size))
    blocks: list[dict[str, Any]] = []
    for start in range(0, arr.size, block_size):
        block = arr[start:start + block_size]
        if block.size == 0:
            continue
        blocks.append(
            {
                "start_index": int(start),
                "end_index": int(start + block.size - 1),
                "n": int(block.size),
                "mean": float(block.mean()),
                "std": float(block.std()),
            }
        )
    return blocks


def bootstrap_mean_ci(
    values: Any,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 7,
) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if arr.size == 1:
        val = float(arr[0])
        return {"n": 1, "mean": val, "ci_low": val, "ci_high": val}
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
    }


def longest_run(mask: Any) -> int:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    best = current = 0
    for val in arr:
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def excursion_metrics(
    values: Any,
    threshold: float,
    *,
    times: Any | None = None,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr) & (arr > threshold)
    fraction = float(mask.mean()) if mask.size else None
    longest = longest_run(mask)
    duration = None
    if times is not None and len(np.asarray(times).reshape(-1)) >= 2 and longest > 0:
        times_arr = np.asarray(times, dtype=float).reshape(-1)
        dt = float(np.nanmedian(np.diff(times_arr[: min(times_arr.size, arr.size)])))
        duration = dt * longest if math.isfinite(dt) else None
    return {
        "threshold": float(threshold),
        "fraction_above": fraction,
        "longest_run_frames": int(longest),
        "longest_run_time_ns": duration,
    }


def matrix_window_similarity(matrix_a: Any, matrix_b: Any) -> dict[str, Any]:
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 2:
        return {"pearson_r": None, "frobenius_diff": None}
    tri = np.triu_indices(a.shape[0], k=1)
    va = a[tri]
    vb = b[tri]
    if va.size < 2:
        return {"pearson_r": None, "frobenius_diff": float(np.linalg.norm(a - b))}
    if np.std(va) == 0 or np.std(vb) == 0:
        corr = None
    else:
        corr = float(np.corrcoef(va, vb)[0, 1])
    return {
        "pearson_r": corr,
        "frobenius_diff": float(np.linalg.norm(a - b)),
    }


def rmsip(eigenvectors_a: Any, eigenvectors_b: Any, *, n_components: int = 3) -> float | None:
    a = np.asarray(eigenvectors_a, dtype=float)
    b = np.asarray(eigenvectors_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        return None
    k = min(n_components, a.shape[1], b.shape[1])
    if k <= 0:
        return None
    qa, _ = np.linalg.qr(a[:, :k])
    qb, _ = np.linalg.qr(b[:, :k])
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.sqrt(np.sum(sv ** 2) / k))