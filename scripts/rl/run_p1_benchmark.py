#!/usr/bin/env python3
"""Run one independent P1 B0, B1, B2, B3, or S0 benchmark process.

B0 measures the direct robot state/action core without Task-v0 height-map or
contact observations. It runs both frozen step_core and rollout_with_reset
workloads. B1 measures only the complete Task-v0 / Observation-v0
rollout_with_reset workload. This script does not start ROS, Yamcs, SDG
runtime, or a trainer. S0 reuses the B1 workload for the frozen 100,000-step
stability run and streams transition records to avoid benchmark-side growth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PERIODIC_ACTIONS = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.0),
    (0.0, -0.5),
    (-0.25, 0.0),
)
ACTION_HOLD_POLICY_STEPS = 25
GPU_IDLE_MEMORY_LIMIT_BYTES = 1024**3
GPU_IDLE_UTILIZATION_LIMIT_PERCENT = 5
GPU_PREFLIGHT_DURATION_S = 10
SHARED_GPU_MIN_FREE_BYTES = 8 * 1024**3


class ResourceContentionError(RuntimeError):
    """Raised when the monitor detects a new external process on the selected GPU."""


def _raise_if_resource_contention(contention_file: Path | None) -> None:
    if contention_file is None or not contention_file.is_file():
        return
    details = json.loads(contention_file.read_text(encoding="utf-8"))
    raise ResourceContentionError(
        f"new external GPU process detected: {details.get('unexpected_gpu_pids', [])}"
    )


def _gpu_argument(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError("GPU index must be non-negative or 'auto'")
    return index


def _gpu_sample_is_idle(sample: dict[str, Any]) -> bool:
    return bool(
        int(sample["memory_used_bytes"]) < GPU_IDLE_MEMORY_LIMIT_BYTES
        and int(sample["gpu_utilization_percent"]) < GPU_IDLE_UTILIZATION_LIMIT_PERCENT
        and not sample["compute_processes"]
        and not sample["graphics_processes"]
    )


def _select_idle_gpu(samples: list[dict[str, Any]]) -> int | None:
    eligible = [sample for sample in samples if _gpu_sample_is_idle(sample)]
    if not eligible:
        return None
    selected = min(eligible, key=lambda sample: (int(sample["memory_used_bytes"]), int(sample["gpu_index"])))
    return int(selected["gpu_index"])


def _select_shared_gpu(samples: list[dict[str, Any]]) -> int | None:
    eligible = [
        sample
        for sample in samples
        if int(sample["memory_total_bytes"]) - int(sample["memory_used_bytes"]) >= SHARED_GPU_MIN_FREE_BYTES
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda sample: (int(sample["memory_used_bytes"]), int(sample["gpu_index"])))
    return int(selected["gpu_index"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("b0", "b1", "b2", "b3", "s0"), default="b0")
    parser.add_argument("--gpu", type=_gpu_argument, default="auto")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="defaults to 100 for B2/B3 and 500 for B0/B1/S0",
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=None,
        help="defaults to 1000 for B2/B3, 100000 for S0, and 10000 for B0/B1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="allow shared conservative performance measurement; card-level metrics remain non-formal",
    )
    return parser.parse_args()


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays in resolved configs to JSON primitives."""

    return json.loads(json.dumps(value, default=_json_default))


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latencies(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50_s": None, "p95_s": None, "p99_s": None, "max_s": None}
    return {
        "p50_s": _percentile(values, 50.0),
        "p95_s": _percentile(values, 95.0),
        "p99_s": _percentile(values, 99.0),
        "max_s": max(values),
    }


def _throughput(steps: int, wall_s: float) -> dict[str, float]:
    env_steps_s = steps / wall_s
    return {
        "measured_policy_steps": steps,
        "measured_wall_s": wall_s,
        "env_steps_per_s": env_steps_s,
        "physics_steps_per_s": 6.0 * env_steps_s,
        "real_time_factor": 0.2 * env_steps_s,
        "t_1m_hours": 1_000_000.0 / env_steps_s / 3600.0,
        "t_100m_days": 100_000_000.0 / env_steps_s / 86400.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _periodic_action(np, step: int):
    action_index = (step // ACTION_HOLD_POLICY_STEPS) % len(PERIODIC_ACTIONS)
    return np.asarray(PERIODIC_ACTIONS[action_index], dtype=np.float32)


def _record_transition(mode: str, index: int, latency_s: float, action, observation, reward, terminated, truncated, info):
    np = __import__("numpy")
    applied = info["applied_vehicle_command"]
    root_position = np.asarray(info.get("root_position_m", observation[:3]), dtype=np.float64)
    root_orientation = np.asarray(info.get("root_orientation_xyzw", observation[3:7]), dtype=np.float64)
    return {
        "mode": mode,
        "policy_step": index,
        "step_wall_s": latency_s,
        "action_linear": float(action[0]),
        "action_yaw": float(action[1]),
        "action_finite": bool(np.isfinite(action).all()),
        "applied_v_m_s": float(applied[0]),
        "applied_omega_rad_s": float(applied[1]),
        "root_x_m": float(root_position[0]),
        "root_y_m": float(root_position[1]),
        "root_z_m": float(root_position[2]),
        "quat_x": float(root_orientation[0]),
        "quat_y": float(root_orientation[1]),
        "quat_z": float(root_orientation[2]),
        "quat_w": float(root_orientation[3]),
        "actual_v_m_s": float(info.get("actual_v_m_s", observation[7])),
        "actual_omega_rad_s": float(info.get("actual_omega_rad_s", observation[8])),
        "observation_size": int(observation.size),
        "observation_version": info["observation_version"],
        "drive_targets_rad_s": json.dumps(info["drive_targets_rad_s"], sort_keys=True, separators=(",", ":")),
        "measured_wheel_rad_s": json.dumps(
            info["measured_wheel_rad_s"], sort_keys=True, separators=(",", ":")
        ),
        "reward": float(reward),
        "reward_finite": bool(np.isfinite(reward)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason"),
        "episode_id": info["episode_id"],
        "episode_step": info["episode_steps"],
        "physics_steps_this_action": info["physics_steps_this_action"],
        "physics_steps_total": info["physics_steps_total"],
        "articulation_ready": info["articulation_ready"],
        "observation_finite": bool(np.isfinite(observation).all()),
    }


def _run_step_core(
    env,
    np,
    seed: int,
    warmup: int,
    measured: int,
    reset_options: dict[str, Any],
    contention_file: Path | None = None,
):
    observation, reset_info = env.reset(seed=seed, options=reset_options)
    first_observation_at = time.perf_counter()
    del observation
    for index in range(warmup):
        _raise_if_resource_contention(contention_file)
        _, _, terminated, truncated, _ = env.step(_periodic_action(np, index))
        _raise_if_resource_contention(contention_file)
        if terminated or truncated:
            raise RuntimeError(f"B0 step_core ended during warmup at step {index + 1}")

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    invalid_articulation = invalid_action = invalid_observation = invalid_reward = bad_decimation = 0
    start = time.perf_counter()
    for index in range(measured):
        _raise_if_resource_contention(contention_file)
        before = time.perf_counter()
        action = _periodic_action(np, warmup + index)
        observation, reward, terminated, truncated, info = env.step(action)
        latency = time.perf_counter() - before
        _raise_if_resource_contention(contention_file)
        latencies.append(latency)
        rows.append(
            _record_transition("step_core", index + 1, latency, action, observation, reward, terminated, truncated, info)
        )
        invalid_articulation += int(not info["articulation_ready"])
        invalid_action += int(not np.isfinite(action).all())
        invalid_observation += int(not np.isfinite(observation).all())
        invalid_reward += int(not np.isfinite(reward))
        bad_decimation += int(info["physics_steps_this_action"] != 6)
        if terminated or truncated:
            raise RuntimeError(f"B0 step_core ended unexpectedly at measured step {index + 1}")
    wall_s = time.perf_counter() - start
    return {
        **_throughput(measured, wall_s),
        "step_latency": _latencies(latencies),
        "initial_reset": reset_info,
        "articulation_invalid_count": invalid_articulation,
        "action_nonfinite_count": invalid_action,
        "observation_nonfinite_count": invalid_observation,
        "reward_nonfinite_count": invalid_reward,
        "action_decimation_error_count": bad_decimation,
        "unexpected_terminal_count": 0,
        "_first_observation_perf_counter": first_observation_at,
    }, rows


def _run_rollout(
    env,
    np,
    seed: int,
    warmup: int,
    measured: int,
    reset_options: dict[str, Any],
    contention_file: Path | None = None,
    expected_observation_size: int | None = None,
    expected_observation_version: str | None = None,
    stream_output_dir: Path | None = None,
    policy_step_hook: Any | None = None,
):
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed, options=reset_options)
    first_observation_at = time.perf_counter()
    del observation
    for _ in range(warmup):
        _raise_if_resource_contention(contention_file)
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if policy_step_hook is not None:
            policy_step_hook()
        _raise_if_resource_contention(contention_file)
        if terminated or truncated:
            env.reset(options=reset_options)

    # The measured interval starts at a clean episode boundary; this reset is
    # excluded from throughput, as is the initial reset in step_core.
    _, measured_reset = env.reset(options=reset_options)
    if policy_step_hook is not None and hasattr(policy_step_hook, "begin_measurement"):
        policy_step_hook.begin_measurement()
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    latencies: list[float] = []
    reset_latencies: list[float] = []
    invalid_articulation = invalid_action = invalid_observation = invalid_reward = bad_decimation = 0
    observation_contract_errors = 0
    local_resets = 0
    episode_return = 0.0
    episode_length = 0
    transition_stream = episode_stream = None
    transition_writer = None
    if stream_output_dir is not None:
        transition_stream = (stream_output_dir / "transitions.csv").open("w", newline="", encoding="utf-8")
        episode_stream = (stream_output_dir / "episodes.jsonl").open("w", encoding="utf-8")
    measured_start_monotonic = time.monotonic()
    start = time.perf_counter()
    try:
        for index in range(measured):
            _raise_if_resource_contention(contention_file)
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            before = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            if policy_step_hook is not None:
                policy_step_hook()
            latency = time.perf_counter() - before
            _raise_if_resource_contention(contention_file)
            latencies.append(latency)
            row = _record_transition(
                "rollout_with_reset", index + 1, latency, action, observation, reward, terminated, truncated, info
            )
            if transition_stream is None:
                rows.append(row)
            else:
                if transition_writer is None:
                    transition_writer = csv.DictWriter(transition_stream, fieldnames=list(row))
                    transition_writer.writeheader()
                transition_writer.writerow(row)
            invalid_articulation += int(not info["articulation_ready"])
            invalid_action += int(not np.isfinite(action).all())
            invalid_observation += int(not np.isfinite(observation).all())
            invalid_reward += int(not np.isfinite(reward))
            bad_decimation += int(info["physics_steps_this_action"] != 6)
            observation_contract_errors += int(
                (expected_observation_size is not None and observation.size != expected_observation_size)
                or (
                    expected_observation_version is not None
                    and info["observation_version"] != expected_observation_version
                )
            )
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                episode = {
                    "episode_id": info["episode_id"],
                    "length": episode_length,
                    "return": episode_return,
                    "reason": info["termination_reason"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                episodes.append(episode)
                if episode_stream is not None:
                    episode_stream.write(json.dumps(episode, sort_keys=True) + "\n")
                reset_start = time.perf_counter()
                env.reset(options=reset_options)
                reset_latencies.append(time.perf_counter() - reset_start)
                local_resets += 1
                episode_return = 0.0
                episode_length = 0
    finally:
        wall_s = time.perf_counter() - start
        measured_end_monotonic = time.monotonic()
        if transition_stream is not None:
            transition_stream.close()
        if episode_stream is not None:
            episode_stream.close()
    if episode_length:
        final_episode = {
                "episode_id": info["episode_id"],
                "length": episode_length,
                "return": episode_return,
                "reason": "measurement_end",
                "terminated": False,
                "truncated": False,
            }
        episodes.append(final_episode)
        if stream_output_dir is not None:
            with (stream_output_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(final_episode, sort_keys=True) + "\n")
    return {
        **_throughput(measured, wall_s),
        "step_latency": _latencies(latencies),
        "reset_latency": _latencies(reset_latencies),
        "measured_interval_local_resets": local_resets,
        "initial_measured_reset": measured_reset,
        "articulation_invalid_count": invalid_articulation,
        "action_nonfinite_count": invalid_action,
        "observation_nonfinite_count": invalid_observation,
        "reward_nonfinite_count": invalid_reward,
        "action_decimation_error_count": bad_decimation,
        "observation_contract_error_count": observation_contract_errors,
        "termination_reason_counts": {
            reason: sum(episode["reason"] == reason for episode in episodes)
            for reason in sorted({episode["reason"] for episode in episodes})
        },
        "_first_observation_perf_counter": first_observation_at,
        "_measured_start_monotonic_s": measured_start_monotonic,
        "_measured_end_monotonic_s": measured_end_monotonic,
    }, rows, episodes


def _memory_trend(
    rows: list[dict[str, str]],
    value_key: str,
    measured_start_monotonic_s: float,
    measured_end_monotonic_s: float,
    measured_steps: int,
) -> dict[str, float | int | None]:
    """Compute frozen first/last-decile growth and a least-squares step slope."""

    selected = [
        (float(row["monotonic_s"]), int(row[value_key]))
        for row in rows
        if measured_start_monotonic_s <= float(row["monotonic_s"]) <= measured_end_monotonic_s
    ]
    if not selected:
        return {
            "sample_count": 0,
            "first_10_percent_median_bytes": None,
            "last_10_percent_median_bytes": None,
            "growth_bytes": None,
            "slope_bytes_per_1000_env_steps": None,
        }
    edge_count = max(1, math.ceil(len(selected) * 0.1))
    first_median = float(statistics.median(value for _, value in selected[:edge_count]))
    last_median = float(statistics.median(value for _, value in selected[-edge_count:]))
    duration_s = measured_end_monotonic_s - measured_start_monotonic_s
    slope_per_1000: float | None = None
    if len(selected) >= 2 and duration_s > 0.0:
        points = [
            ((timestamp - measured_start_monotonic_s) / duration_s * measured_steps, float(value))
            for timestamp, value in selected
        ]
        mean_x = statistics.fmean(point[0] for point in points)
        mean_y = statistics.fmean(point[1] for point in points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator > 0.0:
            slope_per_step = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
            slope_per_1000 = slope_per_step * 1000.0
    return {
        "sample_count": len(selected),
        "first_10_percent_median_bytes": first_median,
        "last_10_percent_median_bytes": last_median,
        "growth_bytes": last_median - first_median,
        "slope_bytes_per_1000_env_steps": slope_per_1000,
    }


def _resource_summary(
    output_dir: Path,
    measured_start_monotonic_s: float | None = None,
    measured_end_monotonic_s: float | None = None,
    measured_steps: int | None = None,
) -> dict[str, Any]:
    def read(name: str):
        with (output_dir / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    process_rows = read("process_samples.csv")
    gpu_rows = read("gpu_samples.csv")
    project_gpu_memory_peak = max((int(row["target_gpu_memory_bytes"]) for row in gpu_rows), default=0)
    result = {
        "sample_count": min(len(process_rows), len(gpu_rows)),
        "process_rss_peak_bytes": max((int(row["rss_bytes"]) for row in process_rows), default=0),
        "process_vms_peak_bytes": max((int(row["vms_bytes"]) for row in process_rows), default=0),
        "gpu_memory_peak_bytes": project_gpu_memory_peak,
        "project_gpu_memory_peak_bytes": project_gpu_memory_peak,
        "card_gpu_memory_peak_bytes": max((int(row["memory_used_bytes"]) for row in gpu_rows), default=0),
        "other_gpu_memory_peak_bytes": max((int(row["other_gpu_memory_bytes"]) for row in gpu_rows), default=0),
        "unattributed_gpu_memory_peak_bytes": max(
            (int(row["unattributed_gpu_memory_bytes"]) for row in gpu_rows), default=0
        ),
        "gpu_memory_total_bytes": max((int(row["memory_total_bytes"]) for row in gpu_rows), default=0),
        "gpu_utilization_peak_percent": max((int(row["gpu_utilization_percent"]) for row in gpu_rows), default=0),
        "other_gpu_pids": sorted(
            {
                pid
                for row in gpu_rows
                for pid in json.loads(row["other_gpu_pids"])
            }
        ),
        "target_gpu_pids": sorted(
            {
                pid
                for row in gpu_rows
                for pid in json.loads(row["target_gpu_pids"])
            }
        ),
    }
    if (
        measured_start_monotonic_s is not None
        and measured_end_monotonic_s is not None
        and measured_steps is not None
    ):
        result["measured_interval_memory_trend"] = {
            "process_rss": _memory_trend(
                process_rows,
                "rss_bytes",
                measured_start_monotonic_s,
                measured_end_monotonic_s,
                measured_steps,
            ),
            "project_gpu_memory": _memory_trend(
                gpu_rows,
                "target_gpu_memory_bytes",
                measured_start_monotonic_s,
                measured_end_monotonic_s,
                measured_steps,
            ),
        }
    return result


def _s0_stability_gate(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen S0 correctness and measured-interval memory limits."""

    growth_limit = 512 * 1024**2
    slope_limit = 1 * 1024**2
    trends = result["resources"].get("measured_interval_memory_trend", {})
    rss = trends.get("process_rss", {})
    gpu = trends.get("project_gpu_memory", {})

    def metric_pass(metric: dict[str, Any]) -> bool:
        return bool(
            metric.get("sample_count", 0) >= 2
            and metric.get("growth_bytes") is not None
            and metric["growth_bytes"] <= growth_limit
            and metric.get("slope_bytes_per_1000_env_steps") is not None
            and metric["slope_bytes_per_1000_env_steps"] <= slope_limit
        )

    return {
        "expected_measured_policy_steps": 100_000,
        "measured_policy_steps": result["measured_policy_steps"],
        "completion_pass": result["measured_policy_steps"] == 100_000,
        "correctness_pass": all(
            result[key] == 0
            for key in (
                "articulation_invalid_count",
                "action_nonfinite_count",
                "observation_nonfinite_count",
                "reward_nonfinite_count",
                "action_decimation_error_count",
                "observation_contract_error_count",
                "full_world_resets_during_benchmark",
            )
        ),
        "rss_growth_max_bytes": growth_limit,
        "gpu_growth_max_bytes": growth_limit,
        "slope_max_bytes_per_1000_env_steps": slope_limit,
        "rss_memory_pass": metric_pass(rss),
        "gpu_memory_pass": metric_pass(gpu),
        "rss_peak_pass": result["resources"]["process_rss_peak_bytes"] <= 24 * 1024**3,
        "gpu_peak_pass": result["resources"]["project_gpu_memory_peak_bytes"]
        < 0.9 * result["resources"]["gpu_memory_total_bytes"],
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# P1 {summary['case']} independent run",
        "",
        f"- pass: `{str(summary['pass']).lower()}`",
        f"- seed: `{summary['seed']}`",
        f"- GPU: index `{summary['gpu']}`, `{summary['gpu_name']}`, `{summary['gpu_uuid']}`",
        f"- formal eligible: `{str(summary.get('formal_eligible', False)).lower()}`",
    ]
    if "step_core" in summary:
        lines.append(f"- step_core: `{summary['step_core']['env_steps_per_s']:.6f}` env-step/s")
    lines.extend(
        [
        f"- rollout_with_reset: `{summary['rollout_with_reset']['env_steps_per_s']:.6f}` env-step/s",
        f"- articulation invalid: `{summary['articulation_invalid_count']}`",
        f"- action non-finite: `{summary['action_nonfinite_count']}`",
        f"- action decimation errors: `{summary['action_decimation_error_count']}`",
        f"- observation contract errors: `{summary['observation_contract_error_count']}`",
        f"- observation non-finite: `{summary['observation_nonfinite_count']}`",
        f"- reward non-finite: `{summary['reward_nonfinite_count']}`",
        f"- resource samples: `{summary['resources']['sample_count']}`",
        f"- project PID GPU memory peak: `{summary['resources']['project_gpu_memory_peak_bytes'] / 1024**2:.3f}` MiB",
        f"- other PID GPU memory peak: `{summary['resources']['other_gpu_memory_peak_bytes'] / 1024**2:.3f}` MiB",
        f"- card GPU memory peak: `{summary['resources']['card_gpu_memory_peak_bytes'] / 1024**2:.3f}` MiB",
        "",
        ]
    )
    if "stability_gate" in summary:
        trend = summary["resources"]["measured_interval_memory_trend"]
        lines.extend(
            [
                f"- S0 completion: `{summary['measured_policy_steps']}/100000`",
                f"- RSS measured growth: `{trend['process_rss']['growth_bytes']}` bytes",
                "- RSS slope: "
                f"`{trend['process_rss']['slope_bytes_per_1000_env_steps']}` bytes/1000 env-step",
                f"- project GPU measured growth: `{trend['project_gpu_memory']['growth_bytes']}` bytes",
                "- project GPU slope: "
                f"`{trend['project_gpu_memory']['slope_bytes_per_1000_env_steps']}` bytes/1000 env-step",
                "",
            ]
        )
    if "camera_cost_probe" in summary:
        camera = summary["camera_cost_probe"]
        lines.extend(
            [
                f"- camera frames: `{camera['valid_capture_count']}/{camera['capture_count']}` valid",
                f"- camera resolution: `{camera['resolution']}`",
                f"- camera invalid frames: `{camera['invalid_capture_count']}`",
                "",
            ]
        )
    if "terrain_deformation_probe" in summary:
        deformation = summary["terrain_deformation_probe"]
        lines.extend(
            [
                f"- terrain deformation updates: `{deformation['update_count']}/{deformation['attempt_count']}`",
                f"- terrain deformation errors: `{deformation['error_count']}`",
                f"- changed DEM samples: `{deformation['changed_dem_sample_count']}`",
                f"- maximum absolute DEM change: `{deformation['maximum_abs_dem_change_m']:.9f}` m",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    case_name = args.case.upper()
    if args.warmup_steps is None:
        args.warmup_steps = 100 if args.case in ("b2", "b3") else 500
    if args.measured_steps is None:
        args.measured_steps = 1_000 if args.case in ("b2", "b3") else 100_000 if args.case == "s0" else 10_000
    if args.case == "s0" and args.seed != 42:
        raise ValueError("S0 frozen workload requires seed 42")
    if args.case in ("b2", "b3") and args.seed != 42:
        raise ValueError(f"{args.case.upper()} frozen workload requires seed 42")
    requested_gpu = args.gpu
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        raise ValueError("warmup must be non-negative and measured steps must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc)
    process_start = time.perf_counter()
    command = "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python " + " ".join(
        shlex.quote(part) for part in sys.argv
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    status = _git_output("status", "--short", "--branch")
    diff = _git_output("diff", "--binary")
    (args.output_dir / "git_status.txt").write_text(status, encoding="utf-8")
    (args.output_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    diff_sha256 = hashlib.sha256(diff.encode()).hexdigest()
    from scripts.rl.resource_monitor import NvmlDevice, nvml_device_count

    discovery_samples: list[dict[str, Any]] = []
    for gpu_index in range(nvml_device_count()):
        discovery_device = NvmlDevice(gpu_index)
        try:
            discovery_sample = discovery_device.sample()
        finally:
            discovery_device.close()
        discovery_sample["gpu_index"] = gpu_index
        discovery_samples.append(discovery_sample)
    _write_json(args.output_dir / "gpu_discovery.json", discovery_samples)

    if requested_gpu == "auto":
        selected_gpu = _select_shared_gpu(discovery_samples) if args.allow_shared_gpu else _select_idle_gpu(discovery_samples)
    else:
        selected_gpu = int(requested_gpu)
    if selected_gpu is None:
        failure = (
            "no GPU retains the required shared-run memory headroom"
            if args.allow_shared_gpu
            else "no GPU satisfies the frozen idle/exclusive preflight"
        )
        (args.output_dir / "failure.txt").write_text(failure + "\n", encoding="utf-8")
        _write_json(
            args.output_dir / "manifest.json",
            {
                "protocol": "P1 frozen 2026-10-14; GPU selection revision 2026-10-15",
                "case": case_name,
                "seed": args.seed,
                "gpu_requested": requested_gpu,
                "gpu": None,
                "command": command,
                "commit": _git_output("rev-parse", "HEAD").strip(),
                "dirty": bool(_git_output("status", "--porcelain").strip()),
                "diff_sha256": diff_sha256,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "start_utc": start_utc.isoformat(),
                "end_utc_before_shutdown": datetime.now(timezone.utc).isoformat(),
                "exit_code": 3,
                "validity": "invalid",
                "pass": False,
                "invalid_reason": "insufficient_shared_gpu_headroom"
                if args.allow_shared_gpu
                else "no_idle_exclusive_gpu",
            },
        )
        (args.output_dir / "exit_code.txt").write_text("3\n", encoding="utf-8")
        print(failure, flush=True)
        return 3
    args.gpu = selected_gpu

    preflight_samples: list[dict[str, Any]] = []
    preflight_device = NvmlDevice(args.gpu)
    try:
        # Include both endpoints so samples span a full ten monotonic seconds.
        sample_count = 1 if args.allow_shared_gpu else GPU_PREFLIGHT_DURATION_S + 1
        for sample_index in range(sample_count):
            sample = preflight_device.sample()
            sample["gpu_index"] = args.gpu
            sample["sample_index"] = sample_index
            sample["monotonic_s"] = time.monotonic()
            preflight_samples.append(sample)
            if sample_index + 1 < sample_count:
                time.sleep(1.0)
    finally:
        preflight_device.close()
    _write_json(args.output_dir / "gpu_preflight_samples.json", preflight_samples)
    preflight_gpu = preflight_samples[-1]
    preflight_gpu_pids = sorted(
        {
            item["pid"]
            for sample in preflight_samples
            for process_kind in ("compute_processes", "graphics_processes")
            for item in sample[process_kind]
        }
    )
    preflight_exclusive = all(_gpu_sample_is_idle(sample) for sample in preflight_samples)

    host_preflight = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    (args.output_dir / "host_preflight.txt").write_text(host_preflight, encoding="utf-8")
    if not preflight_exclusive and not args.allow_shared_gpu:
        failure = f"GPU {args.gpu} did not remain idle/exclusive for {len(preflight_samples)} preflight samples"
        (args.output_dir / "failure.txt").write_text(failure + "\n", encoding="utf-8")
        _write_json(
            args.output_dir / "manifest.json",
            {
                "protocol": "P1 frozen 2026-10-14; GPU selection revision 2026-10-15",
                "case": case_name,
                "seed": args.seed,
                "gpu_requested": requested_gpu,
                "gpu": args.gpu,
                "gpu_uuid": preflight_gpu["gpu_uuid"],
                "gpu_name": preflight_gpu["gpu_name"],
                "command": command,
                "commit": _git_output("rev-parse", "HEAD").strip(),
                "dirty": bool(_git_output("status", "--porcelain").strip()),
                "diff_sha256": diff_sha256,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
                "start_utc": start_utc.isoformat(),
                "end_utc_before_shutdown": datetime.now(timezone.utc).isoformat(),
                "exit_code": 3,
                "validity": "invalid",
                "pass": False,
                "invalid_reason": "gpu_not_exclusive",
                "preflight_gpu_pids": preflight_gpu_pids,
                "preflight_exclusive": preflight_exclusive,
            },
        )
        (args.output_dir / "exit_code.txt").write_text("3\n", encoding="utf-8")
        print(failure, flush=True)
        return 3

    stop_file = args.output_dir / ".stop_resource_monitor"
    contention_file = args.output_dir / "resource_contention.json"
    monitor_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/rl/resource_monitor.py"),
        "--target-pid",
        str(os.getpid()),
        "--gpu",
        str(args.gpu),
        "--output-dir",
        str(args.output_dir),
        "--stop-file",
        str(stop_file),
        "--contention-file",
        str(contention_file),
    ]
    for allowed_pid in preflight_gpu_pids:
        monitor_command.extend(("--allowed-other-gpu-pid", str(allowed_pid)))
    monitor = subprocess.Popen(monitor_command, cwd=REPO_ROOT)
    abort_contention_file = None if args.allow_shared_gpu else contention_file

    from isaacsim import SimulationApp

    launch_config = {
        "headless": True,
        "renderer": "RayTracedLighting",
        "active_gpu": None,
        "active_cuda_gpus": [args.gpu],
        "physics_gpu": args.gpu,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": ["--portable-root", str(REPO_ROOT / ".cache" / "isaacsim")],
    }
    simulation_app = SimulationApp(launch_config)
    app_ready = time.perf_counter()
    scene = env = sensor_probe = None
    result: dict[str, Any] | None = None
    invalid_reason: str | None = None
    exit_code = 1
    try:
        import numpy as np
        from omegaconf import OmegaConf

        from scripts.rl.smoke_env import _scene_and_robot, _wait_for_articulation
        from src.environments_wrappers.rl.env import OmniLRSEnv
        from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, RobotAdapterCfg
        from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter

        robot_values = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "cfg/rl/husky_v0.yaml"), resolve=True)
        adapter_cfg = RobotAdapterCfg.from_mapping(robot_values)
        scene = _scene_and_robot(args, simulation_app, adapter_cfg)
        scene_ready = time.perf_counter()
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        articulation_ready = time.perf_counter()
        common_env_cfg = {
            "robot": adapter_cfg,
            "default_start_position": tuple(float(value) for value in scene["start_position"]),
            "default_start_orientation_xyzw": tuple(float(value) for value in scene["start_orientation_xyzw"]),
            "render": args.case == "b2",
            "render_interval_physics_steps": 6 if args.case == "b2" else 1,
            "record_trace": False,
        }
        step_core = None
        core_rows: list[dict[str, Any]] = []
        if args.case == "b0":
            reset_options = {
                "start_position": scene["start_position"],
                "start_orientation_xyzw": scene["start_orientation_xyzw"],
            }
            core_cfg = OmniLRSEnvCfg(
                episode_policy_steps=args.warmup_steps + args.measured_steps + 1,
                **common_env_cfg,
            )
            env = OmniLRSEnv(scene["world"], adapter, core_cfg)
            step_core, core_rows = _run_step_core(
                env,
                np,
                args.seed,
                args.warmup_steps,
                args.measured_steps,
                reset_options,
                abort_contention_file,
            )
            first_observation = step_core.pop("_first_observation_perf_counter")
            env.close()
            rollout_cfg = OmniLRSEnvCfg(episode_policy_steps=300, **common_env_cfg)
            env = OmniLRSEnv(scene["world"], adapter, rollout_cfg)
            expected_observation_size = 11
            expected_observation_version = "DiagnosticObservation-P1"
            resolved_case_cfg = {
                "periodic_actions": PERIODIC_ACTIONS,
                "action_hold_policy_steps": ACTION_HOLD_POLICY_STEPS,
                "core_environment": asdict(core_cfg),
                "rollout_environment": asdict(rollout_cfg),
            }
        else:
            from src.environments_wrappers.rl.isaac_providers import DemTerrainProvider, NamedRigidContactProvider
            from src.environments_wrappers.rl.task_runtime import TaskV0Runtime

            terrain = DemTerrainProvider(scene["lab"].dem, resolution_m=0.01)
            contacts = NamedRigidContactProvider(
                robot_prim_path=adapter_cfg.prim_path,
                contact_links=adapter_cfg.contact_links,
                physics_dt_s=1.0 / 30.0,
                name="omnilrs_b1_contact_view",
            )
            adapter.update()
            runtime = TaskV0Runtime(
                terrain_height_query=terrain.height,
                footprint_is_clear=lambda xy, yaw: terrain.footprint_is_finite(
                    xy, yaw, length_m=0.990, width_m=0.670
                ),
                classified_contact_forces=contacts,
                robot_cfg=adapter_cfg,
                terrain_id="Terrain_lidar01",
            )
            reset_options = {}
            rollout_cfg = OmniLRSEnvCfg(episode_policy_steps=300, **common_env_cfg)
            env = OmniLRSEnv(scene["world"], adapter, rollout_cfg, runtime)
            expected_observation_size = runtime.observation_cfg.observation_size
            expected_observation_version = "Observation-v0"
            resolved_case_cfg = {
                "rollout_environment": asdict(rollout_cfg),
                "task": asdict(runtime.task_cfg),
                "observation": asdict(runtime.observation_cfg),
                "reward": asdict(runtime.reward_cfg),
                "termination": asdict(runtime.termination_cfg),
                "contact_links": list(adapter_cfg.contact_links),
            }
            if args.case == "b2":
                import omni

                from src.environments_wrappers.rl.benchmark_sensors import D455RgbDepthCostProbe

                camera_prim_path = f"{adapter_cfg.prim_path.rstrip('/')}/d455/RSD455/Camera_OmniVision_OV9782_Color"
                camera_prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_prim_path)
                if not camera_prim.IsValid():
                    raise RuntimeError(f"D455 color camera prim does not exist: {camera_prim_path}")
                sensor_probe = D455RgbDepthCostProbe(camera_prim_path, resolution=(320, 240))
                resolved_case_cfg["camera_cost_probe"] = {
                    "prim_path": camera_prim_path,
                    "resolution": [320, 240],
                    "frequency_hz": 5.0,
                    "modalities": ["rgba", "distance_to_image_plane"],
                    "observation_includes_camera": False,
                }
            elif args.case == "b3":
                from src.environments_wrappers.rl.benchmark_deformation import TerrainDeformationCostProbe

                if not scene["lab"].deformation_conf.enable:
                    raise RuntimeError("B3 requires the deformable Lunalab configuration")
                deformation_engine = scene["lab"].T._G.DE
                link_count = len(adapter_cfg.allowed_ground_contact_links)
                if deformation_engine.num_links != link_count:
                    deformation_engine.num_links = link_count
                    deformation_engine.instantiate_np_buffer(link_count, deformation_engine.profile.shape[0])
                sensor_probe = TerrainDeformationCostProbe(
                    terrain_manager=scene["lab"].T,
                    terrain_provider=terrain,
                    link_provider=contacts,
                    deformation_links=adapter_cfg.allowed_ground_contact_links,
                )
                resolved_case_cfg["terrain_deformation_cost_probe"] = {
                    "enabled": bool(scene["lab"].deformation_conf.enable),
                    "cadence": "once_per_policy_step_after_physics",
                    "orientation_order": "wxyz",
                    "links": list(adapter_cfg.allowed_ground_contact_links),
                    "observation_uses_updated_dem": True,
                    "collider_updated": False,
                }

        rollout, rollout_rows, episodes = _run_rollout(
            env,
            np,
            args.seed,
            args.warmup_steps,
            args.measured_steps,
            reset_options,
            abort_contention_file,
            expected_observation_size,
            expected_observation_version,
            args.output_dir if args.case == "s0" else None,
            sensor_probe,
        )
        rollout_first_observation = rollout.pop("_first_observation_perf_counter")
        measured_start_monotonic = rollout.pop("_measured_start_monotonic_s")
        measured_end_monotonic = rollout.pop("_measured_end_monotonic_s")
        if args.case != "b0":
            first_observation = rollout_first_observation
        core_observation_contract_error_count = sum(
            row["observation_size"] != expected_observation_size
            or row["observation_version"] != expected_observation_version
            for row in core_rows
        )
        observation_contract_error_count = (
            core_observation_contract_error_count + rollout["observation_contract_error_count"]
        )
        if args.case != "s0":
            _write_csv(args.output_dir / "transitions.csv", [*core_rows, *rollout_rows])
            with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                for episode in episodes:
                    stream.write(json.dumps(episode, sort_keys=True) + "\n")
        result = {
            "case": case_name,
            "seed": args.seed,
            "gpu_requested": requested_gpu,
            "gpu": args.gpu,
            "gpu_uuid": preflight_gpu["gpu_uuid"],
            "gpu_name": preflight_gpu["gpu_name"],
            "preflight_gpu_pids": preflight_gpu_pids,
            "preflight_exclusive": preflight_exclusive,
            "warmup_policy_steps": args.warmup_steps,
            "measured_policy_steps": args.measured_steps,
            "rollout_with_reset": rollout,
            "articulation_invalid_count": (step_core["articulation_invalid_count"] if step_core else 0)
            + rollout["articulation_invalid_count"],
            "action_nonfinite_count": (step_core["action_nonfinite_count"] if step_core else 0)
            + rollout["action_nonfinite_count"],
            "observation_nonfinite_count": (step_core["observation_nonfinite_count"] if step_core else 0)
            + rollout["observation_nonfinite_count"],
            "reward_nonfinite_count": (step_core["reward_nonfinite_count"] if step_core else 0)
            + rollout["reward_nonfinite_count"],
            "action_decimation_error_count": (step_core["action_decimation_error_count"] if step_core else 0)
            + rollout["action_decimation_error_count"],
            "observation_contract_error_count": observation_contract_error_count,
            "full_world_resets_during_benchmark": 0,
            "measurement_window_monotonic_s": {
                "start": measured_start_monotonic,
                "end": measured_end_monotonic,
            },
            "startup": {
                "process_to_app_ready_s": app_ready - process_start,
                "app_ready_to_scene_ready_s": scene_ready - app_ready,
                "scene_ready_to_articulation_ready_s": articulation_ready - scene_ready,
                "process_to_first_observation_s": first_observation - process_start,
                "articulation_ready_physics_step": ready_step,
            },
            "robot_usd": adapter_cfg.usd_path,
            "terrain_id": "Terrain_lidar01",
            "physics_dt_s": rollout_cfg.action.physics_dt_s,
            "action_decimation": rollout_cfg.action.action_decimation,
            "policy_dt_s": rollout_cfg.action.policy_dt_s,
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "isaacsim_version": importlib.metadata.version("isaacsim"),
        }
        if step_core is not None:
            result["step_core"] = step_core
        if args.case == "b2" and sensor_probe is not None:
            result["camera_cost_probe"] = sensor_probe.summary()
        if args.case == "b3" and sensor_probe is not None:
            result["terrain_deformation_probe"] = sensor_probe.summary()
        result["pass"] = bool(
            result["articulation_invalid_count"] == 0
            and result["action_nonfinite_count"] == 0
            and result["observation_nonfinite_count"] == 0
            and result["reward_nonfinite_count"] == 0
            and result["action_decimation_error_count"] == 0
            and result["observation_contract_error_count"] == 0
            and (
                sensor_probe is None
                or args.case == "b3"
                and result["terrain_deformation_probe"]["attempt_count"] == args.measured_steps
                and result["terrain_deformation_probe"]["update_count"] == args.measured_steps
                and result["terrain_deformation_probe"]["error_count"] == 0
                and result["terrain_deformation_probe"]["changed_dem_sample_count"] > 0
                or (
                    result["camera_cost_probe"]["capture_count"] == args.measured_steps
                    and result["camera_cost_probe"]["invalid_capture_count"] == 0
                )
            )
            and (step_core is None or step_core["unexpected_terminal_count"] == 0)
        )
        OmegaConf.save(
            OmegaConf.create(
                _json_safe({
                    "case": args.case,
                    "seed": args.seed,
                    "warmup_policy_steps": args.warmup_steps,
                    "measured_policy_steps": args.measured_steps,
                    "gpu_requested": requested_gpu,
                    "gpu_selected": args.gpu,
                    "gpu_uuid": preflight_gpu["gpu_uuid"],
                    "gpu_name": preflight_gpu["gpu_name"],
                    "allow_shared_gpu": args.allow_shared_gpu,
                    **resolved_case_cfg,
                    "launch": launch_config,
                })
            ),
            args.output_dir / "resolved_config.yaml",
        )
        exit_code = 0 if result["pass"] else 2
    except ResourceContentionError:
        invalid_reason = "runtime_resource_contention"
        exit_code = 3
        failure = traceback.format_exc()
        (args.output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        print(failure, flush=True)
    except BaseException:
        invalid_reason = "execution_failure"
        failure = traceback.format_exc()
        (args.output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        print(failure, flush=True)
    finally:
        if env is not None:
            env.close()
        if scene is not None:
            scene["timeline"].stop()
            scene["world"].stop()
        stop_file.touch()
        try:
            monitor.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=5.0)
        if monitor.returncode != 0:
            exit_code = 1
            (args.output_dir / "resource_monitor_failure.txt").write_text(
                f"resource monitor exit code {monitor.returncode}\n", encoding="utf-8"
            )
        if result is not None and monitor.returncode == 0:
            window = result["measurement_window_monotonic_s"]
            result["resources"] = _resource_summary(
                args.output_dir,
                window["start"],
                window["end"],
                result["measured_policy_steps"],
            )
            result["execution_complete"] = invalid_reason is None
            if invalid_reason is not None:
                result["pass"] = False
            result["gpu_exclusive_throughout"] = not bool(result["resources"]["other_gpu_pids"])
            result["shared_gpu_diagnostic"] = bool(args.allow_shared_gpu)
            result["runtime_resource_contention"] = (
                json.loads(contention_file.read_text(encoding="utf-8")) if contention_file.is_file() else None
            )
            frozen_measured_steps = 1_000 if args.case in ("b2", "b3") else 100_000 if args.case == "s0" else 10_000
            frozen_warmup_steps = 100 if args.case in ("b2", "b3") else 500
            frozen_workload = bool(
                preflight_gpu["gpu_name"] == "NVIDIA GeForce RTX 4090"
                and args.warmup_steps == frozen_warmup_steps
                and args.measured_steps == frozen_measured_steps
            )
            if args.case == "s0":
                result["stability_gate"] = _s0_stability_gate(result)
                if frozen_workload:
                    result["pass"] = bool(
                        result["pass"]
                        and all(
                            value
                            for key, value in result["stability_gate"].items()
                            if key.endswith("_pass")
                        )
                    )
            result["formal_eligible"] = bool(
                result["pass"]
                and not args.allow_shared_gpu
                and preflight_exclusive
                and frozen_workload
                and result["gpu_exclusive_throughout"]
            )
            result["shared_conservative_eligible"] = bool(
                result["pass"]
                and args.allow_shared_gpu
                and frozen_workload
                and args.case not in ("b2", "b3", "s0")
            )
            result["shared_stability_eligible"] = bool(
                result["pass"] and args.allow_shared_gpu and frozen_workload and args.case == "s0"
            )
            result["diagnostic_eligible"] = bool(
                result["pass"] and frozen_workload and args.case in ("b2", "b3")
            )
            result["gate_eligible"] = bool(
                args.case not in ("b2", "b3")
                and (
                    result["formal_eligible"]
                    or result["shared_conservative_eligible"]
                    or result["shared_stability_eligible"]
                )
            )
            result["performance_validity"] = (
                "not_applicable"
                if args.case == "s0"
                else "shared_approximate"
                if args.case in ("b2", "b3") and args.allow_shared_gpu and result["diagnostic_eligible"]
                else "isolated"
                if result["formal_eligible"]
                else "shared_conservative"
                if result["shared_conservative_eligible"]
                else "invalid"
            )
            if not args.allow_shared_gpu:
                result["pass"] = bool(result["pass"] and result["gpu_exclusive_throughout"])
            if not result["pass"] and invalid_reason is None:
                exit_code = 2
            _write_json(args.output_dir / "summary.json", result)
            (args.output_dir / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
            print("P1_BENCHMARK_SUMMARY=" + json.dumps(result, sort_keys=True, default=_json_default), flush=True)

        end_utc = datetime.now(timezone.utc)
        manifest = {
            "protocol": "P1 frozen 2026-10-14; GPU selection revision 2026-10-15",
            "case": case_name,
            "seed": args.seed,
            "gpu_requested": requested_gpu,
            "gpu": args.gpu,
            "gpu_uuid": preflight_gpu["gpu_uuid"],
            "gpu_name": preflight_gpu["gpu_name"],
            "command": command,
            "commit": _git_output("rev-parse", "HEAD").strip(),
            "dirty": bool(_git_output("status", "--porcelain").strip()),
            "diff_sha256": diff_sha256,
            "physics_dt_s": 1.0 / 30.0,
            "action_decimation": 6,
            "policy_dt_s": 0.2,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "terrain_id": "Terrain_lidar01",
            "robot_usd": "assets/USD_Assets/robots/nograph_husky.usd",
            "sensor_flags": {
                "heightmap": args.case in ("b1", "b2", "b3", "s0"),
                "camera": args.case == "b2",
                "terrain_deformation": args.case == "b3",
            },
            "preflight_gpu_pids": preflight_gpu_pids,
            "preflight_exclusive": preflight_exclusive,
            "allow_shared_gpu": args.allow_shared_gpu,
            "start_utc": start_utc.isoformat(),
            "end_utc_before_shutdown": end_utc.isoformat(),
            "exit_code": exit_code,
            "validity": (
                "invalid_resource_contention"
                if invalid_reason == "runtime_resource_contention"
                else "invalid"
                if invalid_reason is not None
                else "valid_shared_conservative"
                if result is not None
                and monitor.returncode == 0
                and args.allow_shared_gpu
                and result.get("shared_conservative_eligible", False)
                else "valid_shared_stability"
                if result is not None
                and monitor.returncode == 0
                and args.allow_shared_gpu
                and result.get("shared_stability_eligible", False)
                else "valid_shared_diagnostic"
                if result is not None
                and monitor.returncode == 0
                and args.allow_shared_gpu
                and result.get("diagnostic_eligible", False)
                else "diagnostic_shared"
                if result is not None and monitor.returncode == 0 and args.allow_shared_gpu
                else "invalid_resource_contention"
                if result is not None
                and monitor.returncode == 0
                and not result.get("gpu_exclusive_throughout", False)
                else "valid"
                if result is not None and monitor.returncode == 0
                else "invalid"
            ),
            "pass": bool(exit_code == 0 and result and result.get("pass")),
            "formal_eligible": bool(exit_code == 0 and result and result.get("formal_eligible")),
            "gate_eligible": bool(exit_code == 0 and result and result.get("gate_eligible")),
            "performance_validity": result.get("performance_validity") if result else "invalid",
            "invalid_reason": invalid_reason,
        }
        _write_json(args.output_dir / "manifest.json", manifest)
        (args.output_dir / "environment_versions.txt").write_text(
            f"python={platform.python_version()}\n"
            f"isaacsim={importlib.metadata.version('isaacsim')}\n"
            f"gymnasium={importlib.metadata.version('gymnasium')}\n"
            f"psutil={importlib.metadata.version('psutil')}\n",
            encoding="utf-8",
        )
        (args.output_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        stop_file.unlink(missing_ok=True)
        print(f"P1_BENCHMARK_SHUTDOWN_REQUEST={time.monotonic():.9f}", flush=True)
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
