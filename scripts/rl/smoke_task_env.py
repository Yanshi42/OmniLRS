#!/usr/bin/env python3
"""Run a short native Task-v0 runtime smoke in Lunalab (not a benchmark)."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
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
    parser.add_argument("--zero-steps", type=int, default=2)
    parser.add_argument("--forward-steps", type=int, default=5)
    return parser.parse_args()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


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


def main() -> int:
    args = parse_args()
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
        # Creating a contact tensor view must not invalidate the articulation.
        articulation_after_contact_view = adapter.update()
        del articulation_after_contact_view

        runtime = TaskV0Runtime(
            terrain_height_query=terrain.height,
            footprint_is_clear=lambda xy, yaw: terrain.footprint_is_finite(
                xy, yaw, length_m=0.990, width_m=0.670
            ),
            classified_contact_forces=contacts,
            robot_cfg=adapter_cfg,
            terrain_id="Terrain_lidar01",
        )
        env_cfg = OmniLRSEnvCfg(robot=adapter_cfg, render=False)
        env = OmniLRSEnv(scene["world"], adapter, env_cfg, runtime)
        options = {"start_pose": (1.25, 2.25, 0.0), "goal_pose": (4.25, 2.25, 0.0)}
        observation, reset_info = env.reset(seed=args.seed, options=options)

        rows: list[dict[str, Any]] = []
        invalid_articulation_count = 0
        exact_decimation = True
        any_termination = False
        for segment, count, action in (
            ("zero", args.zero_steps, np.asarray([0.0, 0.0], dtype=np.float32)),
            ("forward", args.forward_steps, np.asarray([0.5, 0.0], dtype=np.float32)),
        ):
            for _ in range(count):
                observation, reward, terminated, truncated, info = env.step(action)
                state = adapter.get_state()
                forces = contacts(state)
                wheel_max = max(np.linalg.norm(forces[name]) for name in adapter_cfg.allowed_ground_contact_links)
                chassis_names = [name for name in adapter_cfg.contact_links if name not in adapter_cfg.allowed_ground_contact_links]
                chassis_max = max(np.linalg.norm(forces[name]) for name in chassis_names)
                row = {
                    "episode_step": info["episode_steps"],
                    "segment": segment,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": info["termination_reason"],
                    "distance_to_goal_m": info["distance_to_goal"],
                    "path_length_m": info["path_length"],
                    "root_x_m": float(state.position[0]),
                    "root_y_m": float(state.position[1]),
                    "max_allowed_wheel_contact_n": float(wheel_max),
                    "max_chassis_contact_n": float(chassis_max),
                    "collision_debounce_steps": runtime.debounce.collision_physics_steps,
                    "articulation_ready": info["articulation_ready"],
                    "physics_steps_this_action": info["physics_steps_this_action"],
                }
                rows.append(row)
                invalid_articulation_count += int(not info["articulation_ready"])
                exact_decimation &= info["physics_steps_this_action"] == 6
                any_termination |= terminated or truncated
                if terminated or truncated:
                    break
            if any_termination:
                break

        _write_csv(args.output_dir / "transitions.csv", rows)
        initial_distance = float(reset_info["distance_to_goal"])
        final_distance = float(rows[-1]["distance_to_goal_m"])
        passed = bool(
            observation.shape == (10206,)
            and observation.dtype == np.float32
            and observation.flags.c_contiguous
            and reset_info["observation_version"] == "Observation-v0"
            and contacts.is_ready
            and invalid_articulation_count == 0
            and exact_decimation
            and not any_termination
            and final_distance < initial_distance
        )
        summary = {
            "case": "Task-v0 short runtime smoke",
            "pass": passed,
            "seed": args.seed,
            "gpu": args.gpu,
            "robot_usd": adapter_cfg.usd_path,
            "terrain_id": "Terrain_lidar01",
            "articulation_ready_physics_step": ready_step,
            "contact_view_ready": contacts.is_ready,
            "contact_links": list(adapter_cfg.contact_links),
            "allowed_ground_contact_links": list(adapter_cfg.allowed_ground_contact_links),
            "observation_shape": list(observation.shape),
            "observation_dtype": str(observation.dtype),
            "observation_c_contiguous": observation.flags.c_contiguous,
            "action_decimation_exact": exact_decimation,
            "articulation_invalid_count": invalid_articulation_count,
            "unexpected_episode_end": any_termination,
            "initial_distance_to_goal_m": initial_distance,
            "final_distance_to_goal_m": final_distance,
            "distance_progress_m": initial_distance - final_distance,
            "max_allowed_wheel_contact_n": max(float(row["max_allowed_wheel_contact_n"]) for row in rows),
            "max_chassis_contact_n": max(float(row["max_chassis_contact_n"]) for row in rows),
            "reset_info": reset_info,
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "isaacsim_version": importlib.metadata.version("isaacsim"),
        }
        _write_json(args.output_dir / "summary.json", summary)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "fixed_task": options,
                    "zero_steps": args.zero_steps,
                    "forward_steps": args.forward_steps,
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
            },
        )
        print("RL_TASK_SMOKE_SUMMARY=" + json.dumps(summary, sort_keys=True, default=_json_default), flush=True)
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
