import math
import json

import numpy as np
import pytest

from scripts.rl.run_p1_benchmark import (
    ResourceContentionError,
    _gpu_sample_is_idle,
    _json_safe,
    _latencies,
    _memory_trend,
    _periodic_action,
    _raise_if_resource_contention,
    _resource_summary,
    _select_idle_gpu,
    _select_shared_gpu,
    _s0_stability_gate,
    _throughput,
)
from scripts.rl.resource_monitor import _gpu_memory_by_pid, _unexpected_gpu_pids
from scripts.rl.summarize_p1_benchmark import aggregate_runs, aggregate_scalar


def test_frozen_periodic_action_holds_each_value_for_25_steps():
    expected = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.0),
        (0.0, -0.5),
        (-0.25, 0.0),
    )
    for block, action in enumerate(expected):
        for offset in range(25):
            np.testing.assert_array_equal(_periodic_action(np, block * 25 + offset), action)
    np.testing.assert_array_equal(_periodic_action(np, 150), expected[0])


def test_throughput_uses_frozen_policy_period_and_decimation():
    result = _throughput(100, 4.0)
    assert result["env_steps_per_s"] == 25.0
    assert result["physics_steps_per_s"] == 150.0
    assert result["real_time_factor"] == 5.0
    assert math.isclose(result["t_1m_hours"], 1_000_000 / 25 / 3600)
    assert math.isclose(result["t_100m_days"], 100_000_000 / 25 / 86400)


def test_latency_percentiles_are_linearly_interpolated():
    result = _latencies([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["p50_s"] == 3.0
    assert result["p95_s"] == 4.8
    assert result["p99_s"] == 4.96
    assert result["max_s"] == 5.0


def test_empty_latency_set_serializes_as_json_nulls():
    result = _latencies([])
    assert result == {"p50_s": None, "p95_s": None, "p99_s": None, "max_s": None}
    assert "NaN" not in json.dumps(result)


def test_json_safe_converts_numpy_scalars_and_arrays():
    result = _json_safe({"scalar": np.float64(0.5), "array": np.asarray([1, 2], dtype=np.int64)})
    assert result == {"scalar": 0.5, "array": [1, 2]}


def test_auto_gpu_selection_uses_lowest_memory_idle_card():
    def sample(index, memory, utilization=0, compute=None, graphics=None):
        return {
            "gpu_index": index,
            "memory_used_bytes": memory,
            "gpu_utilization_percent": utilization,
            "compute_processes": compute or [],
            "graphics_processes": graphics or [],
        }

    samples = [
        sample(0, 500_000_000, compute=[{"pid": 10}]),
        sample(1, 10_000_000),
        sample(2, 5_000_000),
        sample(3, 2_000_000, utilization=9),
    ]

    assert not _gpu_sample_is_idle(samples[0])
    assert _gpu_sample_is_idle(samples[1])
    assert _select_idle_gpu(samples) == 2


def test_shared_gpu_selection_uses_lowest_memory_card_with_headroom():
    gib = 1024**3
    samples = [
        {"gpu_index": 0, "memory_total_bytes": 48 * gib, "memory_used_bytes": 45 * gib},
        {"gpu_index": 1, "memory_total_bytes": 48 * gib, "memory_used_bytes": 30 * gib},
        {"gpu_index": 2, "memory_total_bytes": 48 * gib, "memory_used_bytes": 20 * gib},
    ]
    assert _select_shared_gpu(samples) == 2


def test_resource_monitor_ignores_target_and_preflight_pids():
    assert _unexpected_gpu_pids({10, 20, 30}, {10}, {20}) == {30}


def test_gpu_memory_by_pid_deduplicates_compute_and_graphics_entries():
    compute = [
        {"pid": 10, "used_gpu_memory_bytes": 100},
        {"pid": 20, "used_gpu_memory_bytes": 200},
    ]
    graphics = [{"pid": 10, "used_gpu_memory_bytes": 100}]
    assert _gpu_memory_by_pid(compute, graphics) == {10: 100, 20: 200}


def test_resource_summary_uses_target_pid_memory_as_primary_gpu_metric(tmp_path):
    (tmp_path / "process_samples.csv").write_text(
        "rss_bytes,vms_bytes\n100,200\n300,400\n", encoding="utf-8"
    )
    (tmp_path / "gpu_samples.csv").write_text(
        "target_gpu_memory_bytes,memory_used_bytes,other_gpu_memory_bytes,"
        "unattributed_gpu_memory_bytes,memory_total_bytes,gpu_utilization_percent,"
        "other_gpu_pids,target_gpu_pids\n"
        '20,800,700,80,1000,50,"[22]","[11]"\n'
        '30,900,750,120,1000,60,"[22,33]","[11]"\n',
        encoding="utf-8",
    )
    result = _resource_summary(tmp_path)
    assert result["gpu_memory_peak_bytes"] == 30
    assert result["project_gpu_memory_peak_bytes"] == 30
    assert result["card_gpu_memory_peak_bytes"] == 900
    assert result["other_gpu_memory_peak_bytes"] == 750
    assert result["target_gpu_pids"] == [11]


def test_memory_trend_uses_only_measured_interval_and_reports_step_slope():
    rows = [
        {"monotonic_s": str(timestamp), "rss_bytes": str(value)}
        for timestamp, value in ((0, 999), (10, 100), (20, 200), (30, 300), (40, 999))
    ]
    result = _memory_trend(rows, "rss_bytes", 10.0, 30.0, 2_000)
    assert result["sample_count"] == 3
    assert result["first_10_percent_median_bytes"] == 100.0
    assert result["last_10_percent_median_bytes"] == 300.0
    assert result["growth_bytes"] == 200.0
    assert result["slope_bytes_per_1000_env_steps"] == pytest.approx(100.0)


def test_s0_stability_gate_checks_completion_correctness_peaks_and_memory_trends():
    good_trend = {
        "sample_count": 10,
        "growth_bytes": 100,
        "slope_bytes_per_1000_env_steps": 10,
    }
    result = {
        "measured_policy_steps": 100_000,
        "articulation_invalid_count": 0,
        "action_nonfinite_count": 0,
        "observation_nonfinite_count": 0,
        "reward_nonfinite_count": 0,
        "action_decimation_error_count": 0,
        "observation_contract_error_count": 0,
        "full_world_resets_during_benchmark": 0,
        "resources": {
            "process_rss_peak_bytes": 8 * 1024**3,
            "project_gpu_memory_peak_bytes": 4 * 1024**3,
            "gpu_memory_total_bytes": 24 * 1024**3,
            "measured_interval_memory_trend": {
                "process_rss": good_trend,
                "project_gpu_memory": good_trend,
            },
        },
    }
    gate = _s0_stability_gate(result)
    assert all(value for key, value in gate.items() if key.endswith("_pass"))

    result["measured_policy_steps"] = 99_999
    result["resources"]["measured_interval_memory_trend"]["process_rss"] = {
        **good_trend,
        "growth_bytes": 513 * 1024**2,
    }
    gate = _s0_stability_gate(result)
    assert not gate["completion_pass"]
    assert not gate["rss_memory_pass"]


def test_benchmark_raises_when_runtime_contention_sentinel_appears(tmp_path):
    contention_file = tmp_path / "resource_contention.json"
    _raise_if_resource_contention(contention_file)
    contention_file.write_text(json.dumps({"unexpected_gpu_pids": [1234]}), encoding="utf-8")
    with pytest.raises(ResourceContentionError, match="1234"):
        _raise_if_resource_contention(contention_file)


def test_three_run_aggregate_reports_population_cv():
    result = aggregate_scalar([20.0, 25.0, 30.0])
    assert result["median"] == 25.0
    assert result["min"] == 20.0
    assert result["max"] == 30.0
    assert result["mean"] == 25.0
    assert result["coefficient_of_variation_percent"] == pytest.approx(
        np.std([20.0, 25.0, 30.0]) / 25.0 * 100.0
    )


def test_b0_aggregate_requires_three_valid_frozen_runs(tmp_path):
    run_dirs = []
    run_specs = zip((42, 43, 44), (21.0, 25.0, 29.0), (3, 4, 6), strict=True)
    for seed, throughput, gpu_index in run_specs:
        run_dir = tmp_path / str(seed)
        run_dir.mkdir()
        mode = {
            "env_steps_per_s": throughput,
            "physics_steps_per_s": 6.0 * throughput,
            "real_time_factor": 0.2 * throughput,
            "t_1m_hours": 1_000_000.0 / throughput / 3600.0,
            "t_100m_days": 100_000_000.0 / throughput / 86400.0,
            "step_latency": {"p50_s": 0.03, "p95_s": 0.04, "p99_s": 0.05, "max_s": 0.06},
        }
        summary = {
            "case": "B0",
            "pass": True,
            "seed": seed,
            "gpu": gpu_index,
            "gpu_uuid": f"GPU-test-{seed}",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_exclusive_throughout": True,
            "formal_eligible": True,
            "warmup_policy_steps": 500,
            "measured_policy_steps": 10_000,
            "step_core": mode,
            "rollout_with_reset": mode,
            "articulation_invalid_count": 0,
            "action_nonfinite_count": 0,
            "observation_nonfinite_count": 0,
            "reward_nonfinite_count": 0,
            "action_decimation_error_count": 0,
            "full_world_resets_during_benchmark": 0,
            "resources": {
                "process_rss_peak_bytes": 100,
                "gpu_memory_peak_bytes": 200,
                "gpu_memory_total_bytes": 1000,
                "other_gpu_pids": [],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {"case": "B0", "validity": "valid", "exit_code": 0, "formal_eligible": True}
            ),
            encoding="utf-8",
        )
        run_dirs.append(run_dir)

    result = aggregate_runs(run_dirs)

    assert result["pass"]
    assert result["gate"]["measured"] == 25.0
    assert result["modes"]["step_core"]["env_steps_per_s"]["min"] == 21.0
    assert [gpu["index"] for gpu in result["gpus"]] == [3, 4, 6]

    for index, run_dir in enumerate(run_dirs):
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            gpu_exclusive_throughout=False,
            formal_eligible=False,
            gate_eligible=True,
            shared_conservative_eligible=True,
            performance_validity="shared_conservative",
        )
        summary["resources"]["other_gpu_pids"] = [123]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "case": "B0",
                    "validity": "valid_shared_conservative",
                    "exit_code": 0,
                    "formal_eligible": False,
                    "gate_eligible": True,
                }
            ),
            encoding="utf-8",
        )
        if index == 0:
            with pytest.raises(ValueError, match="cannot mix benchmark validity classes"):
                aggregate_runs(run_dirs, allow_shared_runs=True)

    shared_result = aggregate_runs(run_dirs, allow_shared_runs=True)
    assert shared_result["pass"]
    assert shared_result["decision"] == "pass"
    assert shared_result["performance_validity"] == "shared_conservative"
    assert not shared_result["card_resource_metrics_eligible"]


def test_b0_aggregate_rejects_legacy_shared_diagnostic(tmp_path):
    run_dirs = []
    for seed in (42, 43, 44):
        run_dir = tmp_path / str(seed)
        run_dir.mkdir()
        mode = {
            "env_steps_per_s": 25.0,
            "physics_steps_per_s": 150.0,
            "real_time_factor": 5.0,
            "t_1m_hours": 1_000_000.0 / 25.0 / 3600.0,
            "t_100m_days": 100_000_000.0 / 25.0 / 86400.0,
            "step_latency": {"p50_s": 0.03, "p95_s": 0.04, "p99_s": 0.05, "max_s": 0.06},
        }
        summary = {
            "case": "B0",
            "pass": True,
            "seed": seed,
            "gpu": 4,
            "gpu_uuid": f"GPU-test-{seed}",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_exclusive_throughout": False,
            "formal_eligible": False,
            "warmup_policy_steps": 500,
            "measured_policy_steps": 10_000,
            "step_core": mode,
            "rollout_with_reset": mode,
            "articulation_invalid_count": 0,
            "action_nonfinite_count": 0,
            "observation_nonfinite_count": 0,
            "reward_nonfinite_count": 0,
            "action_decimation_error_count": 0,
            "full_world_resets_during_benchmark": 0,
            "resources": {
                "process_rss_peak_bytes": 100,
                "gpu_memory_peak_bytes": 200,
                "gpu_memory_total_bytes": 1000,
                "other_gpu_pids": [123],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps({"case": "B0", "validity": "diagnostic_shared", "exit_code": 0}),
            encoding="utf-8",
        )
        run_dirs.append(run_dir)

    with pytest.raises(ValueError, match="not eligible"):
        aggregate_runs(run_dirs, allow_shared_runs=True)


def test_b1_shared_aggregate_applies_rollout_gate_and_shared_inconclusive(tmp_path):
    run_dirs = []
    for seed, throughput in zip((42, 43, 44), (6.0, 7.0, 8.0), strict=True):
        run_dir = tmp_path / str(seed)
        run_dir.mkdir()
        mode = {
            "env_steps_per_s": throughput,
            "physics_steps_per_s": 6.0 * throughput,
            "real_time_factor": 0.2 * throughput,
            "t_1m_hours": 1_000_000.0 / throughput / 3600.0,
            "t_100m_days": 100_000_000.0 / throughput / 86400.0,
            "step_latency": {"p50_s": 0.1, "p95_s": 0.15, "p99_s": 0.2, "max_s": 0.3},
        }
        summary = {
            "case": "B1",
            "pass": True,
            "seed": seed,
            "gpu": 4,
            "gpu_uuid": f"GPU-test-{seed}",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_exclusive_throughout": False,
            "formal_eligible": False,
            "gate_eligible": True,
            "shared_conservative_eligible": True,
            "warmup_policy_steps": 500,
            "measured_policy_steps": 10_000,
            "rollout_with_reset": mode,
            "articulation_invalid_count": 0,
            "action_nonfinite_count": 0,
            "observation_nonfinite_count": 0,
            "reward_nonfinite_count": 0,
            "action_decimation_error_count": 0,
            "observation_contract_error_count": 0,
            "full_world_resets_during_benchmark": 0,
            "resources": {
                "process_rss_peak_bytes": 8 * 1024**3,
                "gpu_memory_peak_bytes": 4 * 1024**3,
                "project_gpu_memory_peak_bytes": 4 * 1024**3,
                "gpu_memory_total_bytes": 48 * 1024**3,
                "other_gpu_pids": [123],
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "case": "B1",
                    "validity": "valid_shared_conservative",
                    "exit_code": 0,
                    "formal_eligible": False,
                    "gate_eligible": True,
                }
            ),
            encoding="utf-8",
        )
        run_dirs.append(run_dir)

    result = aggregate_runs(run_dirs, case="b1", allow_shared_runs=True)
    assert result["pass"]
    assert result["decision"] == "pass"
    assert result["gate"]["throughput_pass"]
    assert result["gate"]["latency_pass"]
    assert result["gate"]["noise_pass"]
    assert "step_core" not in result["modes"]

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mode = summary["rollout_with_reset"]
        mode.update(
            env_steps_per_s=4.0,
            physics_steps_per_s=24.0,
            real_time_factor=0.8,
            t_1m_hours=1_000_000.0 / 4.0 / 3600.0,
            t_100m_days=100_000_000.0 / 4.0 / 86400.0,
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = aggregate_runs(run_dirs, case="b1", allow_shared_runs=True)
    assert not result["pass"]
    assert result["decision"] == "inconclusive_shared"
