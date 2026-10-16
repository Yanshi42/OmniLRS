#!/usr/bin/env python3
"""Compare three independent P1 D0 trajectories against the frozen Gate."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RUNS = 3
EXPECTED_SEED = 42
EXPECTED_POLICY_STEPS = 1_000
POSITION_MAX_ERROR_M = 1.0e-3
YAW_MAX_ERROR_RAD = 1.0e-3
REWARD_MAX_ERROR = 1.0e-4
DISCRETE_FIELDS = (
    "policy_index",
    "episode_id",
    "episode_step",
    "terminated",
    "truncated",
    "termination_reason",
    "physics_steps_this_action",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-shared-runs",
        action="store_true",
        help="accept legacy shared-run manifests; D0 correctness remains Gate-eligible",
    )
    return parser.parse_args()


def _read_run(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    with (path / "transitions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return summary, manifest, rows


def _angle_error(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def compare_runs(run_dirs: list[Path], *, allow_shared_runs: bool = False) -> dict[str, Any]:
    if len(run_dirs) != EXPECTED_RUNS:
        raise ValueError(f"D0 requires exactly {EXPECTED_RUNS} runs, got {len(run_dirs)}")
    loaded = [(*_read_run(path), path.resolve()) for path in run_dirs]
    action_hashes = {summary["action_sequence_sha256"] for summary, _, _, _ in loaded}
    for summary, manifest, rows, path in loaded:
        if summary.get("case") != "D0 independent run" or manifest.get("case") != "D0":
            raise ValueError(f"non-D0 run supplied: {path}")
        if not summary.get("pass") or manifest.get("exit_code") != 0:
            raise ValueError(f"D0 run did not pass its local correctness checks: {path}")
        if summary.get("seed") != EXPECTED_SEED or summary.get("policy_steps") != EXPECTED_POLICY_STEPS:
            raise ValueError(f"D0 run does not use the frozen seed/length: {path}")
        if len(rows) != EXPECTED_POLICY_STEPS:
            raise ValueError(f"D0 run has {len(rows)} transition rows: {path}")
        if not allow_shared_runs and (
            manifest.get("validity") != "valid"
            and manifest.get("validity") != "valid_shared_correctness"
        ):
            raise ValueError(f"D0 run has an unsupported validity: {path}")
    if len(action_hashes) != 1:
        raise ValueError("D0 runs used different action sequences")

    discrete_mismatches = 0
    max_position_error_m = 0.0
    max_yaw_error_rad = 0.0
    max_reward_error = 0.0
    pair_results: list[dict[str, Any]] = []
    for (left_summary, _, left_rows, left_path), (right_summary, _, right_rows, right_path) in itertools.combinations(
        loaded, 2
    ):
        del left_summary, right_summary
        pair_discrete = 0
        pair_position = pair_yaw = pair_reward = 0.0
        for left, right in zip(left_rows, right_rows, strict=True):
            pair_discrete += sum(left[field] != right[field] for field in DISCRETE_FIELDS)
            pair_discrete += int(left["action_linear"] != right["action_linear"])
            pair_discrete += int(left["action_yaw"] != right["action_yaw"])
            position_error = math.sqrt(
                sum(
                    (float(left[field]) - float(right[field])) ** 2
                    for field in ("root_x_m", "root_y_m", "root_z_m")
                )
            )
            pair_position = max(pair_position, position_error)
            pair_yaw = max(
                pair_yaw,
                abs(_angle_error(float(left["root_yaw_rad"]), float(right["root_yaw_rad"]))),
            )
            pair_reward = max(pair_reward, abs(float(left["reward"]) - float(right["reward"])))
        discrete_mismatches += pair_discrete
        max_position_error_m = max(max_position_error_m, pair_position)
        max_yaw_error_rad = max(max_yaw_error_rad, pair_yaw)
        max_reward_error = max(max_reward_error, pair_reward)
        pair_results.append(
            {
                "left": str(left_path),
                "right": str(right_path),
                "discrete_mismatch_count": pair_discrete,
                "root_position_max_error_m": pair_position,
                "root_yaw_max_error_rad": pair_yaw,
                "reward_max_error": pair_reward,
            }
        )

    numerical_pass = bool(
        discrete_mismatches == 0
        and max_position_error_m <= POSITION_MAX_ERROR_M
        and max_yaw_error_rad <= YAW_MAX_ERROR_RAD
        and max_reward_error <= REWARD_MAX_ERROR
    )
    d0_gate_eligible = all(
        bool(
            summary["d0_gate_eligible"]
            if "d0_gate_eligible" in summary
            else summary.get("pass")
            if allow_shared_runs
            else summary.get("formal_eligible")
        )
        for summary, _, _, _ in loaded
    )
    timing_resource_eligible = all(
        bool(summary.get("timing_resource_eligible", False)) for summary, _, _, _ in loaded
    )
    return {
        "case": "D0 aggregate",
        "comparison_pass": numerical_pass,
        "d0_gate_pass": bool(numerical_pass and d0_gate_eligible),
        "validity": "valid" if timing_resource_eligible else "valid_shared_correctness",
        "d0_gate_eligible": d0_gate_eligible,
        "timing_resource_eligible": timing_resource_eligible,
        "run_dirs": [str(path) for _, _, _, path in loaded],
        "seed": EXPECTED_SEED,
        "policy_steps_per_run": EXPECTED_POLICY_STEPS,
        "action_sequence_sha256": next(iter(action_hashes)),
        "discrete_mismatch_count": discrete_mismatches,
        "root_position_max_error_m": max_position_error_m,
        "root_yaw_max_error_rad": max_yaw_error_rad,
        "reward_max_error": max_reward_error,
        "thresholds": {
            "root_position_max_error_m": POSITION_MAX_ERROR_M,
            "root_yaw_max_error_rad": YAW_MAX_ERROR_RAD,
            "reward_max_error": REWARD_MAX_ERROR,
        },
        "pairs": pair_results,
    }


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        summary = compare_runs(args.run_dir, allow_shared_runs=args.allow_shared_runs)
    except Exception as error:
        (args.output_dir / "failure.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        (args.output_dir / "exit_code.txt").write_text("2\n", encoding="utf-8")
        raise
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "case": "D0 aggregate",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "validity": summary["validity"],
                "comparison_pass": summary["comparison_pass"],
                "d0_gate_pass": summary["d0_gate_pass"],
                "run_dirs": summary["run_dirs"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = 0 if summary["comparison_pass"] else 2
    (args.output_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
