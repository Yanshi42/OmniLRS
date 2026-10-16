#!/usr/bin/env python3
"""Run one independent P1 D0 deterministic Task-v0 trajectory."""

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
FIXED_START_POSE = (1.25, 2.25, 0.0)
FIXED_GOAL_POSE = (4.25, 2.25, 0.0)
FROZEN_SEED = 42
FROZEN_POLICY_STEPS = 1_000
SHARED_GPU_MIN_FREE_BYTES = 8 * 1024**3


def _parse_args() -> argparse.Namespace:
    from scripts.rl.run_p1_benchmark import _gpu_argument

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=_gpu_argument, default="auto")
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    parser.add_argument("--policy-steps", type=int, default=FROZEN_POLICY_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="allow a correctness-eligible shared GPU; timing/card metrics remain non-formal",
    )
    return parser.parse_args()


def generate_actions(np: Any, seed: int, policy_steps: int):
    """Return the frozen PCG64 float32 action sequence for one D0 run."""

    if policy_steps <= 0:
        raise ValueError("policy_steps must be positive")
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(policy_steps, 2)).astype(np.float32)


def select_shared_gpu(samples: list[dict[str, Any]]) -> int | None:
    """Select the least-used GPU that retains the frozen correctness headroom."""

    eligible = [
        sample
        for sample in samples
        if int(sample["memory_total_bytes"]) - int(sample["memory_used_bytes"]) >= SHARED_GPU_MIN_FREE_BYTES
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda sample: (int(sample["memory_used_bytes"]), int(sample["gpu_index"])))
    return int(selected["gpu_index"])


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    ).stdout


def _yaw_from_xyzw(quaternion) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    args = _parse_args()
    requested_gpu = args.gpu
    if args.seed != FROZEN_SEED and not args.allow_shared_gpu:
        raise ValueError(f"formal D0 requires seed {FROZEN_SEED}")
    if args.policy_steps != FROZEN_POLICY_STEPS and not args.allow_shared_gpu:
        raise ValueError(f"formal D0 requires {FROZEN_POLICY_STEPS} policy steps")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start_utc = datetime.now(timezone.utc)
    command = "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python " + " ".join(
        shlex.quote(part) for part in sys.argv
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "runner_pid.txt").write_text(f"{os.getpid()}\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_output("status", "--short", "--branch"), encoding="utf-8")
    diff = _git_output("diff", "--binary")
    (args.output_dir / "git_diff.patch").write_text(diff, encoding="utf-8")

    from scripts.rl.resource_monitor import NvmlDevice, nvml_device_count
    from scripts.rl.run_p1_benchmark import (
        GPU_PREFLIGHT_DURATION_S,
        _gpu_sample_is_idle,
        _resource_summary,
        _select_idle_gpu,
    )

    discovery: list[dict[str, Any]] = []
    for gpu_index in range(nvml_device_count()):
        device = NvmlDevice(gpu_index)
        try:
            sample = device.sample()
        finally:
            device.close()
        sample["gpu_index"] = gpu_index
        discovery.append(sample)
    _write_json(args.output_dir / "gpu_discovery.json", discovery)
    if requested_gpu == "auto":
        selected_gpu = select_shared_gpu(discovery) if args.allow_shared_gpu else _select_idle_gpu(discovery)
    else:
        selected_gpu = int(requested_gpu)
    if selected_gpu is None:
        reason = (
            "no GPU retains the required shared-run memory headroom"
            if args.allow_shared_gpu
            else "no GPU satisfies the frozen idle/exclusive preflight"
        )
        (args.output_dir / "failure.txt").write_text(reason + "\n", encoding="utf-8")
        _write_json(
            args.output_dir / "manifest.json",
            {
                "case": "D0",
                "pass": False,
                "exit_code": 3,
                "invalid_reason": "insufficient_shared_gpu_headroom"
                if args.allow_shared_gpu
                else "no_idle_exclusive_gpu",
            },
        )
        (args.output_dir / "exit_code.txt").write_text("3\n", encoding="utf-8")
        print(reason, flush=True)
        return 3
    args.gpu = selected_gpu

    preflight_samples: list[dict[str, Any]] = []
    device = NvmlDevice(args.gpu)
    try:
        sample_count = 1 if args.allow_shared_gpu else GPU_PREFLIGHT_DURATION_S + 1
        for sample_index in range(sample_count):
            sample = device.sample()
            sample.update(gpu_index=args.gpu, sample_index=sample_index, monotonic_s=time.monotonic())
            preflight_samples.append(sample)
            if sample_index + 1 < sample_count:
                time.sleep(1.0)
    finally:
        device.close()
    _write_json(args.output_dir / "gpu_preflight_samples.json", preflight_samples)
    preflight_exclusive = all(_gpu_sample_is_idle(sample) for sample in preflight_samples)
    preflight_pids = sorted(
        {
            process["pid"]
            for sample in preflight_samples
            for kind in ("compute_processes", "graphics_processes")
            for process in sample[kind]
        }
    )
    if not preflight_exclusive and not args.allow_shared_gpu:
        reason = f"GPU {args.gpu} did not remain idle/exclusive during preflight"
        (args.output_dir / "failure.txt").write_text(reason + "\n", encoding="utf-8")
        _write_json(
            args.output_dir / "manifest.json",
            {"case": "D0", "pass": False, "exit_code": 3, "invalid_reason": "gpu_not_exclusive"},
        )
        (args.output_dir / "exit_code.txt").write_text("3\n", encoding="utf-8")
        print(reason, flush=True)
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
    summary: dict[str, Any] | None = None
    failure: str | None = None
    full_world_reset_calls = 0
    completed_steps = 0
    try:
        import numpy as np
        from omegaconf import OmegaConf

        from scripts.rl.smoke_env import _scene_and_robot, _wait_for_articulation
        from src.environments_wrappers.rl.env import OmniLRSEnv
        from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, RobotAdapterCfg
        from src.environments_wrappers.rl.isaac_providers import DemTerrainProvider, NamedRigidContactProvider
        from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter
        from src.environments_wrappers.rl.task_runtime import TaskV0Runtime

        actions = generate_actions(np, args.seed, args.policy_steps)
        actions_sha256 = hashlib.sha256(actions.tobytes(order="C")).hexdigest()
        _write_csv(
            args.output_dir / "actions.csv",
            [
                {"policy_index": index + 1, "action_linear": float(action[0]), "action_yaw": float(action[1])}
                for index, action in enumerate(actions)
            ],
        )

        robot_values = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "cfg/rl/husky_v0.yaml"), resolve=True)
        adapter_cfg = RobotAdapterCfg.from_mapping(robot_values)
        scene = _scene_and_robot(args, simulation_app, adapter_cfg)
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        terrain = DemTerrainProvider(scene["lab"].dem, resolution_m=0.01)
        contacts = NamedRigidContactProvider(
            robot_prim_path=adapter_cfg.prim_path,
            contact_links=adapter_cfg.contact_links,
            physics_dt_s=1.0 / 30.0,
        )
        adapter.update()
        runtime = TaskV0Runtime(
            terrain_height_query=terrain.height,
            footprint_is_clear=lambda xy, yaw: terrain.footprint_is_finite(xy, yaw, length_m=0.990, width_m=0.670),
            classified_contact_forces=contacts,
            robot_cfg=adapter_cfg,
            terrain_id="Terrain_lidar01",
        )
        env_cfg = OmniLRSEnvCfg(robot=adapter_cfg, render=False, record_trace=False)
        env = OmniLRSEnv(scene["world"], adapter, env_cfg, runtime)
        original_world_reset = scene["world"].reset

        def counted_world_reset(*world_args, **world_kwargs):
            nonlocal full_world_reset_calls
            full_world_reset_calls += 1
            return original_world_reset(*world_args, **world_kwargs)

        scene["world"].reset = counted_world_reset
        reset_options = {"start_pose": FIXED_START_POSE, "goal_pose": FIXED_GOAL_POSE}
        observation, _ = env.reset(seed=args.seed, options=reset_options)
        rows: list[dict[str, Any]] = []
        episode_boundaries: list[int] = []
        invalid_articulation = bad_decimation = nonfinite = local_resets = 0
        for index, action in enumerate(actions, start=1):
            if not args.allow_shared_gpu and contention_file.is_file():
                raise RuntimeError("runtime_resource_contention")
            observation, reward, terminated, truncated, info = env.step(action)
            state = adapter.get_state()
            invalid_articulation += int(not info["articulation_ready"])
            bad_decimation += int(info["physics_steps_this_action"] != 6)
            nonfinite += int(
                not np.isfinite(observation).all()
                or not np.isfinite(reward)
                or not np.isfinite(state.position).all()
                or not np.isfinite(state.orientation_xyzw).all()
            )
            rows.append(
                {
                    "policy_index": index,
                    "episode_id": info["episode_id"],
                    "episode_step": info["episode_steps"],
                    "action_linear": float(action[0]),
                    "action_yaw": float(action[1]),
                    "root_x_m": float(state.position[0]),
                    "root_y_m": float(state.position[1]),
                    "root_z_m": float(state.position[2]),
                    "root_yaw_rad": _yaw_from_xyzw(state.orientation_xyzw),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": info["termination_reason"] or "",
                    "physics_steps_this_action": info["physics_steps_this_action"],
                    "articulation_ready": info["articulation_ready"],
                    "observation_finite": bool(np.isfinite(observation).all()),
                }
            )
            completed_steps = index
            if terminated or truncated:
                episode_boundaries.append(index)
                env.reset(options=reset_options)
                local_resets += 1
        _write_csv(args.output_dir / "transitions.csv", rows)
        correctness_pass = bool(
            completed_steps == args.policy_steps
            and invalid_articulation == 0
            and bad_decimation == 0
            and nonfinite == 0
            and full_world_reset_calls == 0
        )
        summary = {
            "case": "D0 independent run",
            "pass": correctness_pass,
            "seed": args.seed,
            "policy_steps": args.policy_steps,
            "completed_policy_steps": completed_steps,
            "action_sequence_sha256": actions_sha256,
            "fixed_start_pose": FIXED_START_POSE,
            "fixed_goal_pose": FIXED_GOAL_POSE,
            "episode_boundaries": episode_boundaries,
            "termination_reasons": [rows[index - 1]["termination_reason"] for index in episode_boundaries],
            "local_reset_count": local_resets,
            "full_world_reset_calls": full_world_reset_calls,
            "articulation_invalid_count": invalid_articulation,
            "action_decimation_error_count": bad_decimation,
            "nonfinite_count": nonfinite,
            "articulation_ready_physics_step": ready_step,
            "gpu": args.gpu,
            "gpu_name": preflight_samples[-1]["gpu_name"],
            "gpu_uuid": preflight_samples[-1]["gpu_uuid"],
            "preflight_exclusive": preflight_exclusive,
            "shared_gpu_diagnostic": bool(args.allow_shared_gpu),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "isaacsim_version": importlib.metadata.version("isaacsim"),
        }
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "seed": args.seed,
                    "policy_steps": args.policy_steps,
                    "fixed_start_pose": FIXED_START_POSE,
                    "fixed_goal_pose": FIXED_GOAL_POSE,
                    "action_generator": "numpy.default_rng(seed).uniform(-1,1).astype(float32)",
                }
            ),
            args.output_dir / "resolved_config.yaml",
        )
        exit_code = 0 if correctness_pass else 2
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
            summary["gpu_exclusive_throughout"] = not bool(summary["resources"]["other_gpu_pids"])
            summary["runtime_resource_contention"] = (
                json.loads(contention_file.read_text(encoding="utf-8")) if contention_file.is_file() else None
            )
            summary["timing_resource_eligible"] = bool(
                summary["pass"]
                and not args.allow_shared_gpu
                and preflight_exclusive
                and summary["gpu_exclusive_throughout"]
            )
            summary["d0_gate_eligible"] = bool(
                summary["pass"]
                and summary["gpu_name"] == "NVIDIA GeForce RTX 4090"
                and args.seed == FROZEN_SEED
                and args.policy_steps == FROZEN_POLICY_STEPS
            )
            # Backward-compatible field: eligibility is case-specific. D0 is a
            # correctness Gate and does not consume shared-card timing metrics.
            summary["formal_eligible"] = summary["d0_gate_eligible"]
            _write_json(args.output_dir / "summary.json", summary)
        manifest = {
            "case": "D0",
            "command": command,
            "commit": _git_output("rev-parse", "HEAD").strip(),
            "dirty": bool(_git_output("status", "--porcelain").strip()),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "start_utc": start_utc.isoformat(),
            "end_utc_before_shutdown": datetime.now(timezone.utc).isoformat(),
            "gpu_requested": requested_gpu,
            "gpu": args.gpu,
            "completed_policy_steps": completed_steps,
            "target_policy_steps": args.policy_steps,
            "exit_code": exit_code,
            "validity": "valid_shared_correctness" if args.allow_shared_gpu else "valid",
            "pass": bool(summary and summary.get("pass")),
            "formal_eligible": bool(summary and summary.get("d0_gate_eligible")),
            "timing_resource_eligible": bool(summary and summary.get("timing_resource_eligible")),
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
