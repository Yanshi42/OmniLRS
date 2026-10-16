#!/usr/bin/env python3
"""Run repeatable Task-v0 termination and contact-classification cases."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--stationary-policy-steps", type=int, default=5)
    parser.add_argument("--wheel-contact-policy-steps", type=int, default=5)
    return parser.parse_args()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _max_consecutive(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.stationary_policy_steps <= 0 or args.wheel_contact_policy_steps <= 0:
        raise ValueError("baseline policy-step counts must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python " + " ".join(
        shlex.quote(part) for part in sys.argv
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_output("status", "--short", "--branch"), encoding="utf-8")
    (args.output_dir / "git_diff.patch").write_text(_git_output("diff", "--binary"), encoding="utf-8")

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
    env = None
    scene = None
    exit_code = 1
    try:
        import numpy as np
        from omegaconf import OmegaConf

        from scripts.rl.smoke_env import _scene_and_robot, _wait_for_articulation
        from src.environments_wrappers.rl.env import OmniLRSEnv
        from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, RobotAdapterCfg
        from src.environments_wrappers.rl.isaac_providers import DemTerrainProvider, NamedRigidContactProvider
        from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter
        from src.environments_wrappers.rl.task_runtime import TaskV0Runtime

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

        current_case = ["reset"]
        current_trial = [-1]
        contact_rows: list[dict[str, Any]] = []
        contact_sample_index = [0]
        allowed = set(adapter_cfg.allowed_ground_contact_links)

        def recorded_contacts(state):
            forces = contacts(state)
            wheel = {name: float(np.linalg.norm(force)) for name, force in forces.items() if name in allowed}
            chassis = {name: float(np.linalg.norm(force)) for name, force in forces.items() if name not in allowed}
            wheel_name = max(wheel, key=wheel.get)
            chassis_name = max(chassis, key=chassis.get)
            contact_rows.append(
                {
                    "case": current_case[0],
                    "trial": current_trial[0],
                    "physics_contact_sample": contact_sample_index[0],
                    "max_wheel_link": wheel_name,
                    "max_wheel_contact_n": wheel[wheel_name],
                    "max_chassis_link": chassis_name,
                    "max_chassis_contact_n": chassis[chassis_name],
                    "chassis_above_20n": chassis[chassis_name] > 20.0,
                }
            )
            contact_sample_index[0] += 1
            return forces

        runtime = TaskV0Runtime(
            terrain_height_query=terrain.height,
            footprint_is_clear=lambda xy, yaw: terrain.footprint_is_finite(
                xy, yaw, length_m=0.990, width_m=0.670
            ),
            classified_contact_forces=recorded_contacts,
            robot_cfg=adapter_cfg,
            terrain_id="Terrain_lidar01",
        )
        env_cfg = OmniLRSEnvCfg(robot=adapter_cfg, render=False)
        env = OmniLRSEnv(scene["world"], adapter, env_cfg, runtime)
        options = {"start_pose": (1.25, 2.25, 0.0), "goal_pose": (4.25, 2.25, 0.0)}
        result_rows: list[dict[str, Any]] = []

        def reset_case(case_name: str, trial: int):
            current_case[0] = "reset"
            current_trial[0] = trial
            observation, info = env.reset(seed=args.seed, options=options)
            current_case[0] = case_name
            state = adapter.get_state()
            return observation, info, state

        def teleport(position, orientation):
            adapter.emergency_stop()
            adapter.zero_joint_velocities()
            adapter.teleport_root(np.asarray(position, dtype=np.float64), np.asarray(orientation, dtype=np.float64))
            adapter.zero_root_velocities()

        def step_until_end(case_name: str, trial: int, max_steps: int, expected_reason: str | None, action_fn=None):
            start_contact_index = len(contact_rows)
            final = None
            articulation_invalid = 0
            exact_decimation = True
            for step_index in range(max_steps):
                action = np.zeros(2, dtype=np.float32) if action_fn is None else action_fn(step_index)
                observation, reward, terminated, truncated, info = env.step(action)
                articulation_invalid += int(not info["articulation_ready"])
                exact_decimation &= info["physics_steps_this_action"] == 6
                final = (observation, reward, terminated, truncated, info)
                if terminated or truncated:
                    break
            assert final is not None
            observation, reward, terminated, truncated, info = final
            samples = contact_rows[start_contact_index:]
            chassis_values = [float(row["max_chassis_contact_n"]) for row in samples]
            above = [bool(row["chassis_above_20n"]) for row in samples]
            expected_terminated = expected_reason in {"success", "collision", "rollover", "out_of_bounds"}
            expected_truncated = expected_reason == "time_limit"
            passed = bool(
                info["termination_reason"] == expected_reason
                and terminated == expected_terminated
                and truncated == expected_truncated
                and articulation_invalid == 0
                and exact_decimation
                and np.all(np.isfinite(observation))
                and observation.shape == (10206,)
                and (expected_reason is None or np.array_equal(info["applied_vehicle_command"], np.zeros(2)))
                and (expected_reason is None or all(value == 0.0 for value in info["drive_targets_rad_s"].values()))
            )
            row = {
                "case": case_name,
                "trial": trial,
                "expected_reason": expected_reason,
                "actual_reason": info["termination_reason"],
                "terminated": terminated,
                "truncated": truncated,
                "success": info["success"],
                "episode_steps": info["episode_steps"],
                "physics_steps_this_action": info["physics_steps_this_action"],
                "articulation_invalid_count": articulation_invalid,
                "contact_samples": len(samples),
                "max_chassis_contact_n": max(chassis_values, default=0.0),
                "max_consecutive_chassis_above_20n": _max_consecutive(above),
                "collision_triggered": runtime.debounce.collision_triggered,
                "rollover_triggered": runtime.debounce.rollover_triggered,
                "reward": reward,
                "terminal_reward": info["reward_components"]["terminal"],
                "max_roll_rad": info["max_roll"],
                "max_pitch_rad": info["max_pitch"],
                "drive_targets_zero": all(value == 0.0 for value in info["drive_targets_rad_s"].values()),
                "pass": passed,
            }
            result_rows.append(row)
            return row

        # Stationary and driven wheel-ground contact false-positive baselines.
        reset_case("stationary", 0)
        stationary = step_until_end("stationary", 0, args.stationary_policy_steps, None)
        stationary["pass"] &= stationary["max_consecutive_chassis_above_20n"] == 0

        reset_case("wheel_contact", 0)

        def wheel_action(step_index):
            direction = 1.0 if (step_index // 4) % 2 == 0 else -1.0
            return np.asarray([0.25 * direction, 0.0], dtype=np.float32)

        wheel_contact = step_until_end(
            "wheel_contact", 0, args.wheel_contact_policy_steps, None, action_fn=wheel_action
        )
        wheel_contact["pass"] &= wheel_contact["max_consecutive_chassis_above_20n"] == 0

        start_height = float(terrain.height(np.asarray([[1.25, 2.25]]))[0])
        goal_height = float(terrain.height(np.asarray([[4.25, 2.25]]))[0])
        oob_height = float(terrain.height(np.asarray([[0.30, 2.25]]))[0])
        roll = math.radians(75.0)
        for trial in range(args.repetitions):
            # Success: inject a valid terminal pose at the fixed goal.
            _, _, state = reset_case("success", trial)
            settled_clearance = float(state.position[2] - start_height)
            teleport([4.25, 2.25, goal_height + settled_clearance], [0.0, 0.0, 0.0, 1.0])
            step_until_end("success", trial, 1, "success")

            # Out-of-bounds: cross Task-v0's root bound while staying clear of
            # terrain.  The first formal run showed that reusing the settled
            # clearance can make this independent reason probe hit a rough DEM
            # patch and correctly lose to collision precedence.
            _, _, state = reset_case("out_of_bounds", trial)
            settled_clearance = float(state.position[2] - start_height)
            teleport(
                [0.30, 2.25, oob_height + settled_clearance + 0.75],
                [0.0, 0.0, 0.0, 1.0],
            )
            step_until_end("out_of_bounds", trial, 1, "out_of_bounds")

            # Rollover: raise the root clear of terrain and inject 75 degrees roll.
            _, _, state = reset_case("rollover", trial)
            teleport(
                state.position + np.asarray([0.0, 0.0, 0.75]),
                [math.sin(roll / 2.0), 0.0, 0.0, math.cos(roll / 2.0)],
            )
            step_until_end("rollover", trial, 1, "rollover")

            # Chassis scrape: penetrate the settled base into terrain while keeping XY valid.
            _, _, state = reset_case("collision", trial)
            teleport(state.position - np.asarray([0.0, 0.0, 0.14]), [0.0, 0.0, 0.0, 1.0])
            collision = step_until_end("collision", trial, 3, "collision")
            collision["pass"] &= collision["max_consecutive_chassis_above_20n"] >= 2

            # Frozen timeout: exactly policy step 300 and no earlier termination.
            reset_case("time_limit", trial)
            timeout = step_until_end("time_limit", trial, 300, "time_limit")
            timeout["pass"] &= timeout["episode_steps"] == 300

        _write_csv(args.output_dir / "termination_results.csv", result_rows)
        _write_csv(args.output_dir / "contact_samples.csv", contact_rows)
        formal_t0 = bool(
            args.repetitions >= 10
            and args.stationary_policy_steps * env_cfg.action.action_decimation >= 1000
            and args.wheel_contact_policy_steps * env_cfg.action.action_decimation >= 1000
        )
        passed = all(bool(row["pass"]) for row in result_rows)
        summary = {
            "case": "Task-v0 dynamic termination coverage",
            "pass": passed,
            "formal_t0": formal_t0,
            "repetitions_per_reason": args.repetitions,
            "stationary_physics_steps": args.stationary_policy_steps * env_cfg.action.action_decimation,
            "wheel_contact_physics_steps": args.wheel_contact_policy_steps * env_cfg.action.action_decimation,
            "seed": args.seed,
            "gpu": args.gpu,
            "isaacsim_version": importlib.metadata.version("isaacsim"),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "robot_usd": adapter_cfg.usd_path,
            "terrain_id": "Terrain_lidar01",
            "articulation_ready_physics_step": ready_step,
            "contact_view_ready": contacts.is_ready,
            "results": result_rows,
        }
        _write_json(args.output_dir / "summary.json", summary)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "fixed_task": options,
                    "repetitions_per_reason": args.repetitions,
                    "stationary_policy_steps": args.stationary_policy_steps,
                    "wheel_contact_policy_steps": args.wheel_contact_policy_steps,
                }
            ),
            args.output_dir / "resolved_config.yaml",
        )
        _write_json(
            args.output_dir / "manifest.json",
            {
                "command": command,
                "commit": _git_output("rev-parse", "HEAD").strip(),
                "dirty": bool(_git_output("status", "--porcelain").strip()),
                "output_dir": args.output_dir,
                "pass": passed,
                "formal_t0": formal_t0,
            },
        )
        print("RL_TERMINATION_PROBE_SUMMARY=" + json.dumps(summary, sort_keys=True, default=_json_default), flush=True)
        exit_code = 0 if passed else 2
    except BaseException:
        failure = traceback.format_exc()
        (args.output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        print(failure, flush=True)
    finally:
        (args.output_dir / "exit_code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
        if env is not None:
            env.close()
        if scene is not None:
            scene["timeline"].stop()
            scene["world"].stop()
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
