#!/usr/bin/env python3
"""Aggregate three valid independent P1 B0 or B1 benchmark runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_WARMUP_STEPS = 500
EXPECTED_MEASURED_STEPS = 10_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("b0", "b1"), default="b0")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-shared-runs",
        action="store_true",
        help="accept shared conservative runs; below-threshold results are inconclusive, not No-Go",
    )
    return parser.parse_args()


def aggregate_scalar(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty value list")
    mean = statistics.fmean(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "coefficient_of_variation_percent": 0.0 if mean == 0.0 else statistics.pstdev(values) / mean * 100.0,
    }


def _load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = path / "summary.json"
    manifest_path = path / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"missing summary.json or manifest.json in {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return summary, manifest


def aggregate_runs(
    run_dirs: list[Path], *, case: str = "b0", allow_shared_runs: bool = False
) -> dict[str, Any]:
    case_lower = case.lower()
    if case_lower not in {"b0", "b1"}:
        raise ValueError(f"unsupported benchmark case: {case}")
    case_name = case_lower.upper()
    if len(run_dirs) != 3:
        raise ValueError(f"{case_name} aggregation requires exactly 3 runs, got {len(run_dirs)}")
    loaded = [(*_load_run(path), path.resolve()) for path in run_dirs]
    seeds = sorted(int(summary["seed"]) for summary, _, _ in loaded)
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, got {seeds}")
    eligibility_modes: list[str] = []
    for summary, manifest, path in loaded:
        if summary.get("case") != case_name or manifest.get("case") != case_name:
            raise ValueError(f"non-{case_name} run supplied: {path}")
        isolated = bool(
            manifest.get("validity") == "valid"
            and summary.get("formal_eligible")
            and manifest.get("formal_eligible")
            and summary.get("gpu_exclusive_throughout")
            and not summary["resources"].get("other_gpu_pids")
        )
        shared_conservative = bool(
            allow_shared_runs
            and manifest.get("validity") == "valid_shared_conservative"
            and summary.get("pass")
            and summary.get("gate_eligible")
            and summary.get("shared_conservative_eligible")
            and manifest.get("gate_eligible")
        )
        if manifest.get("exit_code") != 0 or not summary.get("pass") or not (isolated or shared_conservative):
            raise ValueError(f"run is not eligible/pass/exit-0: {path}")
        eligibility_modes.append("isolated" if isolated else "shared_conservative")
        if summary.get("warmup_policy_steps") != EXPECTED_WARMUP_STEPS:
            raise ValueError(f"wrong warmup length in {path}")
        if summary.get("measured_policy_steps") != EXPECTED_MEASURED_STEPS:
            raise ValueError(f"wrong measured length in {path}")
        if summary.get("gpu_name") != "NVIDIA GeForce RTX 4090" or not summary.get("gpu_uuid"):
            raise ValueError(f"{case_name} run did not record the frozen GPU model/UUID: {path}")

    if len(set(eligibility_modes)) != 1:
        raise ValueError(f"cannot mix benchmark validity classes: {eligibility_modes}")
    all_isolated = eligibility_modes[0] == "isolated"

    modes: dict[str, Any] = {}
    mode_names = ("step_core", "rollout_with_reset") if case_lower == "b0" else ("rollout_with_reset",)
    for mode in mode_names:
        modes[mode] = {
            metric: aggregate_scalar([float(summary[mode][metric]) for summary, _, _ in loaded])
            for metric in (
                "env_steps_per_s",
                "physics_steps_per_s",
                "real_time_factor",
                "t_1m_hours",
                "t_100m_days",
            )
        }
        for percentile in ("p50_s", "p95_s", "p99_s", "max_s"):
            modes[mode].setdefault("step_latency", {})[percentile] = aggregate_scalar(
                [float(summary[mode]["step_latency"][percentile]) for summary, _, _ in loaded]
            )

    correctness = {
        "articulation_invalid_count": sum(summary["articulation_invalid_count"] for summary, _, _ in loaded),
        "action_nonfinite_count": sum(summary["action_nonfinite_count"] for summary, _, _ in loaded),
        "observation_nonfinite_count": sum(summary["observation_nonfinite_count"] for summary, _, _ in loaded),
        "reward_nonfinite_count": sum(summary["reward_nonfinite_count"] for summary, _, _ in loaded),
        "action_decimation_error_count": sum(summary["action_decimation_error_count"] for summary, _, _ in loaded),
        "full_world_resets_during_benchmark": sum(
            summary["full_world_resets_during_benchmark"] for summary, _, _ in loaded
        ),
        "observation_contract_error_count": sum(
            summary.get("observation_contract_error_count", 0) for summary, _, _ in loaded
        ),
    }
    correctness_pass = bool(
        correctness["articulation_invalid_count"] == 0
        and correctness["action_nonfinite_count"] == 0
        and correctness["observation_nonfinite_count"] == 0
        and correctness["reward_nonfinite_count"] == 0
        and correctness["action_decimation_error_count"] == 0
        and correctness["full_world_resets_during_benchmark"] == 0
        and correctness["observation_contract_error_count"] == 0
    )
    resource_peaks = {
        "process_rss_bytes": max(summary["resources"]["process_rss_peak_bytes"] for summary, _, _ in loaded),
        "gpu_memory_bytes": max(summary["resources"]["gpu_memory_peak_bytes"] for summary, _, _ in loaded),
        "project_gpu_memory_bytes": max(
            summary["resources"].get(
                "project_gpu_memory_peak_bytes", summary["resources"]["gpu_memory_peak_bytes"]
            )
            for summary, _, _ in loaded
        ),
        "card_gpu_memory_bytes": max(
            summary["resources"].get(
                "card_gpu_memory_peak_bytes", summary["resources"]["gpu_memory_peak_bytes"]
            )
            for summary, _, _ in loaded
        ),
        "gpu_memory_total_bytes": max(summary["resources"]["gpu_memory_total_bytes"] for summary, _, _ in loaded),
    }
    if case_lower == "b0":
        measured = modes["step_core"]["env_steps_per_s"]["median"]
        performance_pass = measured >= 20.0
        resource_pass = True
        gate = {
            "step_core_median_min_env_steps_per_s": 20.0,
            "measured": measured,
            "threshold_pass": performance_pass,
            "shared_failure_is_inconclusive": not all_isolated,
        }
    else:
        rollout_metrics = modes["rollout_with_reset"]
        measured = rollout_metrics["env_steps_per_s"]["median"]
        p99_max_s = rollout_metrics["step_latency"]["p99_s"]["max"]
        cv_percent = rollout_metrics["env_steps_per_s"]["coefficient_of_variation_percent"]
        throughput_pass = measured >= 5.0
        latency_pass = p99_max_s <= 1.0
        noise_pass = cv_percent <= 15.0
        performance_pass = throughput_pass and latency_pass and noise_pass
        rss_pass = resource_peaks["process_rss_bytes"] <= 24 * 1024**3
        project_gpu_memory_pass = (
            resource_peaks["project_gpu_memory_bytes"] < 0.9 * resource_peaks["gpu_memory_total_bytes"]
        )
        resource_pass = rss_pass and project_gpu_memory_pass
        gate = {
            "rollout_median_min_env_steps_per_s": 5.0,
            "rollout_median_measured": measured,
            "throughput_pass": throughput_pass,
            "step_latency_p99_max_s": p99_max_s,
            "step_latency_p99_max_allowed_s": 1.0,
            "latency_pass": latency_pass,
            "throughput_cv_percent": cv_percent,
            "throughput_cv_max_percent": 15.0,
            "noise_pass": noise_pass,
            "process_rss_max_bytes": 24 * 1024**3,
            "rss_pass": rss_pass,
            "project_gpu_memory_max_fraction": 0.9,
            "project_gpu_memory_pass": project_gpu_memory_pass,
            "shared_performance_failure_is_inconclusive": not all_isolated,
        }
    passed = bool(performance_pass and correctness_pass and resource_pass)
    if passed:
        decision = "pass"
    elif not correctness_pass or not resource_pass or all_isolated:
        decision = "fail"
    else:
        decision = "inconclusive_shared"
    return {
        "case": f"{case_name} aggregate",
        "pass": passed,
        "decision": decision,
        "performance_validity": "isolated" if all_isolated else "shared_conservative",
        "card_resource_metrics_eligible": all_isolated,
        "gate": gate,
        "seeds": seeds,
        "run_dirs": [str(path) for _, _, path in loaded],
        "gpus": [
            {
                "seed": summary["seed"],
                "index": summary["gpu"],
                "uuid": summary["gpu_uuid"],
                "name": summary["gpu_name"],
            }
            for summary, _, _ in loaded
        ],
        "modes": modes,
        "correctness": correctness,
        "resource_peaks": resource_peaks,
    }


def _markdown(summary: dict[str, Any]) -> str:
    rollout = summary["modes"]["rollout_with_reset"]["env_steps_per_s"]
    rollout_p99 = summary["modes"]["rollout_with_reset"]["step_latency"]["p99_s"]
    lines = [
            f"# P1 {summary['case']}",
            "",
            f"- pass: `{str(summary['pass']).lower()}`",
            f"- decision: `{summary['decision']}`",
            f"- performance validity: `{summary['performance_validity']}`",
            f"- card resource metrics eligible: `{str(summary['card_resource_metrics_eligible']).lower()}`",
            f"- seeds: `{summary['seeds']}`",
    ]
    if "step_core" in summary["modes"]:
        core = summary["modes"]["step_core"]["env_steps_per_s"]
        core_p99 = summary["modes"]["step_core"]["step_latency"]["p99_s"]
        lines.extend(
            [
            f"- step_core env-step/s median/min/max: `{core['median']:.6f}` / `{core['min']:.6f}` / `{core['max']:.6f}`",
            f"- step_core CV: `{core['coefficient_of_variation_percent']:.3f}%`",
            f"- step_core p99 latency median: `{core_p99['median']:.6f} s`",
            ]
        )
    lines.extend(
        [
            f"- rollout_with_reset env-step/s median/min/max: `{rollout['median']:.6f}` / `{rollout['min']:.6f}` / `{rollout['max']:.6f}`",
            f"- rollout_with_reset CV: `{rollout['coefficient_of_variation_percent']:.3f}%`",
            f"- rollout_with_reset p99 latency median: `{rollout_p99['median']:.6f} s`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        summary = aggregate_runs(args.run_dir, case=args.case, allow_shared_runs=args.allow_shared_runs)
    except Exception as error:
        (args.output_dir / "failure.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        (args.output_dir / "exit_code.txt").write_text("2\n", encoding="utf-8")
        raise
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case": summary["case"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "validity": summary["performance_validity"],
                "pass": summary["pass"],
                "decision": summary["decision"],
                "run_dirs": summary["run_dirs"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = 0 if summary["pass"] else 3 if summary["decision"] == "inconclusive_shared" else 2
    (args.output_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
