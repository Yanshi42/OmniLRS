#!/usr/bin/env python3
"""Run the frozen P1 R0 local-reset stress workload in one Isaac process."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shlex
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-count", type=int, default=500)
    parser.add_argument("--steps-before-reset", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def _yaw_from_xyzw(quaternion) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_error(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _history_residual_count(robot: Any) -> int:
    telemetry = getattr(robot, "telemetry", None)
    if telemetry is None:
        return 0
    residual = 0
    # ``joints_state`` and ``contacts_state`` are the freshly sampled current
    # state after settle and are expected to be populated. Only accumulated
    # history buffers must be empty after a local reset.
    for attr in ("joints_history", "contacts_history"):
        value = getattr(telemetry, attr, None)
        if value is None:
            continue
        try:
            residual += len(value)
        except TypeError:
            residual += int(bool(value))
    return residual


def main() -> int:
    args = _parse_args()
    if args.reset_count <= 0 or args.steps_before_reset <= 0:
        raise ValueError("reset-count and steps-before-reset must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc)
    command = "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python " + " ".join(
        shlex.quote(part) for part in sys.argv
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "runner_pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_output("status", "--short", "--branch"), encoding="utf-8")
    (args.output_dir / "git_diff.patch").write_text(_git_output("diff", "--binary"), encoding="utf-8")

    from scripts.rl.resource_monitor import NvmlDevice
    from scripts.rl.run_p1_benchmark import _resource_summary

    preflight_device = NvmlDevice(args.gpu)
    try:
        preflight = preflight_device.sample()
    finally:
        preflight_device.close()
    preflight["gpu_index"] = args.gpu
    _write_json(args.output_dir / "gpu_preflight.json", preflight)
    preflight_pids = sorted(
        {
            process["pid"]
            for kind in ("compute_processes", "graphics_processes")
            for process in preflight[kind]
        }
    )
    if preflight_pids and not args.allow_shared_gpu:
        _write_json(
            args.output_dir / "manifest.json",
            {
                "case": "R0",
                "pass": False,
                "exit_code": 3,
                "invalid_reason": "gpu_not_exclusive",
                "gpu": args.gpu,
                "preflight_gpu_pids": preflight_pids,
            },
        )
        (args.output_dir / "exit_code.txt").write_text("3\n", encoding="utf-8")
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
    for pid in preflight_pids:
        monitor_command.extend(("--allowed-other-gpu-pid", str(pid)))
    monitor = subprocess.Popen(monitor_command, cwd=REPO_ROOT)

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
    scene = env = None
    exit_code = 1
    completed_resets = 0
    full_world_reset_calls = 0
    failure: str | None = None
    summary: dict[str, Any] | None = None
    rows_path = args.output_dir / "resets.csv"
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
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        articulation_ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        original_world_reset = scene["world"].reset

        def counted_world_reset(*world_args, **world_kwargs):
            nonlocal full_world_reset_calls
            full_world_reset_calls += 1
            return original_world_reset(*world_args, **world_kwargs)

        scene["world"].reset = counted_world_reset
        env_cfg = OmniLRSEnvCfg(
            robot=adapter_cfg,
            episode_policy_steps=args.steps_before_reset + 1,
            default_start_position=tuple(float(value) for value in scene["start_position"]),
            default_start_orientation_xyzw=tuple(float(value) for value in scene["start_orientation_xyzw"]),
            render=False,
            record_trace=False,
        )
        env = OmniLRSEnv(scene["world"], adapter, env_cfg)
        reset_options = {
            "start_position": scene["start_position"],
            "start_orientation_xyzw": scene["start_orientation_xyzw"],
        }
        env.reset(seed=args.seed, options=reset_options)
        rng = np.random.default_rng(args.seed)
        target_position = np.asarray(scene["start_position"], dtype=np.float64)
        target_yaw = _yaw_from_xyzw(scene["start_orientation_xyzw"])
        reset_latencies: list[float] = []
        residual_failures = 0
        articulation_invalid = 0
        action_decimation_errors = 0
        nonfinite_count = 0

        fieldnames = [
            "reset_index",
            "reset_wall_s",
            "settle_physics_steps",
            "xy_error_m",
            "yaw_error_rad",
            "root_linear_speed_m_s",
            "root_angular_speed_rad_s",
            "max_wheel_speed_rad_s",
            "applied_action_max_abs",
            "drive_target_max_abs_rad_s",
            "history_residual_count",
            "episode_steps",
            "elapsed_sim_time_s",
            "articulation_ready",
            "observation_finite",
            "pass",
        ]
        with rows_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for reset_index in range(1, args.reset_count + 1):
                for _ in range(args.steps_before_reset):
                    action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                    observation, reward, terminated, truncated, info = env.step(action)
                    articulation_invalid += int(not info["articulation_ready"])
                    action_decimation_errors += int(info["physics_steps_this_action"] != 6)
                    nonfinite_count += int(not np.isfinite(observation).all() or not np.isfinite(reward))
                    if terminated or truncated:
                        raise RuntimeError(
                            f"R0 episode ended before reset {reset_index} after {info['episode_steps']} steps"
                        )

                before = time.perf_counter()
                observation, info = env.reset(options=reset_options)
                latency = time.perf_counter() - before
                reset_latencies.append(latency)
                state = adapter.get_state()
                wheel_speeds = np.asarray(list(state.joint_velocities_rad_s.values()), dtype=np.float64)
                xy_error = float(np.linalg.norm(state.position[:2] - target_position[:2]))
                yaw_error = abs(_angle_error(_yaw_from_xyzw(state.orientation_xyzw), target_yaw))
                root_linear_speed = float(np.linalg.norm(state.linear_velocity_world_m_s))
                root_angular_speed = float(np.linalg.norm(state.angular_velocity_world_rad_s))
                max_wheel_speed = float(np.max(np.abs(wheel_speeds)))
                applied_action_max = float(np.max(np.abs(info["applied_vehicle_command"])))
                drive_target_max = max(abs(float(value)) for value in info["drive_targets_rad_s"].values())
                history_residual = _history_residual_count(scene["robot"])
                histories_zero = bool(
                    np.all(env._previous_action == 0.0)
                    and np.all(env._previous_applied_action == 0.0)
                    and not env.get_trace()
                    and history_residual == 0
                )
                row_pass = bool(
                    info["articulation_ready"]
                    and info["episode_steps"] == 0
                    and info["elapsed_sim_time"] == 0.0
                    and applied_action_max == 0.0
                    and drive_target_max == 0.0
                    and histories_zero
                    and root_linear_speed < 0.05
                    and root_angular_speed < 0.05
                    and max_wheel_speed < 0.10
                    and xy_error <= 0.02
                    and yaw_error <= 0.02
                    and info["settle_physics_steps"] <= 60
                    and np.isfinite(observation).all()
                )
                residual_failures += int(not row_pass)
                writer.writerow(
                    {
                        "reset_index": reset_index,
                        "reset_wall_s": latency,
                        "settle_physics_steps": info["settle_physics_steps"],
                        "xy_error_m": xy_error,
                        "yaw_error_rad": yaw_error,
                        "root_linear_speed_m_s": root_linear_speed,
                        "root_angular_speed_rad_s": root_angular_speed,
                        "max_wheel_speed_rad_s": max_wheel_speed,
                        "applied_action_max_abs": applied_action_max,
                        "drive_target_max_abs_rad_s": drive_target_max,
                        "history_residual_count": history_residual,
                        "episode_steps": info["episode_steps"],
                        "elapsed_sim_time_s": info["elapsed_sim_time"],
                        "articulation_ready": info["articulation_ready"],
                        "observation_finite": bool(np.isfinite(observation).all()),
                        "pass": row_pass,
                    }
                )
                stream.flush()
                completed_resets = reset_index
                _write_json(
                    args.output_dir / "progress.json",
                    {
                        "case": "R0",
                        "completed_resets": completed_resets,
                        "target_resets": args.reset_count,
                        "last_reset_pass": row_pass,
                        "residual_failures": residual_failures,
                        "last_reset_wall_s": latency,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

        summary = {
            "case": "R0",
            "pass": bool(
                completed_resets == args.reset_count
                and residual_failures == 0
                and articulation_invalid == 0
                and action_decimation_errors == 0
                and nonfinite_count == 0
                and full_world_reset_calls == 0
                and (_percentile(reset_latencies, 95.0) or math.inf) <= 3.0
            ),
            "seed": args.seed,
            "gpu": args.gpu,
            "gpu_name": preflight["gpu_name"],
            "gpu_uuid": preflight["gpu_uuid"],
            "shared_gpu_diagnostic": bool(args.allow_shared_gpu),
            "completed_resets": completed_resets,
            "target_resets": args.reset_count,
            "steps_before_reset": args.steps_before_reset,
            "policy_transitions": completed_resets * args.steps_before_reset,
            "residual_failure_count": residual_failures,
            "articulation_invalid_count": articulation_invalid,
            "action_decimation_error_count": action_decimation_errors,
            "nonfinite_count": nonfinite_count,
            "full_world_resets_during_stress": full_world_reset_calls,
            "reset_latency": {
                "p50_s": _percentile(reset_latencies, 50.0),
                "p95_s": _percentile(reset_latencies, 95.0),
                "p99_s": _percentile(reset_latencies, 99.0),
                "max_s": max(reset_latencies),
            },
            "articulation_ready_physics_step": articulation_ready_step,
            "runtime_resource_contention": None,
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "isaacsim_version": importlib.metadata.version("isaacsim"),
        }
        exit_code = 0 if summary["pass"] else 2
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "seed": args.seed,
                    "reset_count": args.reset_count,
                    "steps_before_reset": args.steps_before_reset,
                }
            ),
            args.output_dir / "resolved_config.yaml",
        )
    except BaseException:
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
        if summary is not None and monitor.returncode == 0:
            summary["resources"] = _resource_summary(args.output_dir)
            summary["runtime_resource_contention"] = (
                json.loads(contention_file.read_text(encoding="utf-8")) if contention_file.is_file() else None
            )
            _write_json(args.output_dir / "summary.json", summary)
        manifest = {
            "case": "R0",
            "command": command,
            "commit": _git_output("rev-parse", "HEAD").strip(),
            "dirty": bool(_git_output("status", "--porcelain").strip()),
            "start_utc": start_utc.isoformat(),
            "end_utc_before_shutdown": datetime.now(timezone.utc).isoformat(),
            "gpu": args.gpu,
            "gpu_name": preflight["gpu_name"],
            "gpu_uuid": preflight["gpu_uuid"],
            "preflight_gpu_pids": preflight_pids,
            "allow_shared_gpu": args.allow_shared_gpu,
            "completed_resets": completed_resets,
            "target_resets": args.reset_count,
            "exit_code": exit_code,
            "validity": "diagnostic_shared" if args.allow_shared_gpu else "valid",
            "pass": bool(summary and summary.get("pass")),
            "failure": failure,
        }
        _write_json(args.output_dir / "manifest.json", manifest)
        (args.output_dir / "environment_versions.txt").write_text(
            f"python={platform.python_version()}\n"
            f"isaacsim={importlib.metadata.version('isaacsim')}\n"
            f"gymnasium={importlib.metadata.version('gymnasium')}\n",
            encoding="utf-8",
        )
        (args.output_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        stop_file.unlink(missing_ok=True)
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
