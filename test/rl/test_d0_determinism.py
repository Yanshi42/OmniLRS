from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rl.run_d0_determinism import SHARED_GPU_MIN_FREE_BYTES, generate_actions, select_shared_gpu
from scripts.rl.summarize_d0_determinism import compare_runs


def test_d0_action_sequence_is_reproducible_and_float32():
    first = generate_actions(np, 42, 1_000)
    second = generate_actions(np, 42, 1_000)

    assert first.shape == (1_000, 2)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(first >= -1.0) and np.all(first <= 1.0)


def _write_fake_run(path: Path, *, position_offset: float = 0.0, formal: bool = True) -> None:
    path.mkdir()
    summary = {
        "case": "D0 independent run",
        "pass": True,
        "seed": 42,
        "policy_steps": 1_000,
        "action_sequence_sha256": "same-actions",
        "formal_eligible": formal,
    }
    manifest = {"case": "D0", "exit_code": 0, "validity": "valid" if formal else "diagnostic_shared", "formal_eligible": formal}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = [
        "policy_index",
        "episode_id",
        "episode_step",
        "action_linear",
        "action_yaw",
        "root_x_m",
        "root_y_m",
        "root_z_m",
        "root_yaw_rad",
        "reward",
        "terminated",
        "truncated",
        "termination_reason",
        "physics_steps_this_action",
    ]
    with (path / "transitions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, 1_001):
            writer.writerow(
                {
                    "policy_index": index,
                    "episode_id": 1,
                    "episode_step": index,
                    "action_linear": 0.25,
                    "action_yaw": -0.5,
                    "root_x_m": index * 0.001 + position_offset,
                    "root_y_m": 2.0,
                    "root_z_m": 0.3,
                    "root_yaw_rad": 0.1,
                    "reward": 0.2,
                    "terminated": False,
                    "truncated": False,
                    "termination_reason": "",
                    "physics_steps_this_action": 6,
                }
            )


def test_d0_comparison_applies_frozen_numeric_thresholds(tmp_path):
    paths = [tmp_path / f"run{index}" for index in range(3)]
    _write_fake_run(paths[0])
    _write_fake_run(paths[1], position_offset=5.0e-4)
    _write_fake_run(paths[2], position_offset=9.0e-4)

    result = compare_runs(paths)

    assert result["comparison_pass"]
    assert result["d0_gate_pass"]
    assert result["root_position_max_error_m"] == pytest.approx(9.0e-4)


def test_shared_d0_comparison_is_correctness_gate_eligible_but_not_timing_eligible(tmp_path):
    paths = [tmp_path / f"run{index}" for index in range(3)]
    for path in paths:
        _write_fake_run(path, formal=False)

    result = compare_runs(paths, allow_shared_runs=True)

    assert result["comparison_pass"]
    assert result["d0_gate_eligible"]
    assert result["d0_gate_pass"]
    assert not result["timing_resource_eligible"]


def test_shared_gpu_selection_prefers_lowest_usage_with_headroom():
    samples = [
        {"gpu_index": 0, "memory_total_bytes": 48 * 1024**3, "memory_used_bytes": 45 * 1024**3},
        {"gpu_index": 1, "memory_total_bytes": 48 * 1024**3, "memory_used_bytes": 30 * 1024**3},
        {"gpu_index": 2, "memory_total_bytes": 48 * 1024**3, "memory_used_bytes": 20 * 1024**3},
    ]

    assert SHARED_GPU_MIN_FREE_BYTES == 8 * 1024**3
    assert select_shared_gpu(samples) == 2
