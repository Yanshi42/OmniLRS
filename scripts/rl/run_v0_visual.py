#!/usr/bin/env python3
"""Generate the frozen P1 V0 visual bundle from four scripted Task-v0 cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return f"$ {shlex.join(command)}\n{result.stdout}\n"


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


def _plot_visual_bundle(case_data, policy_rows, rock_position, visual_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_path(ax, data):
        trace = data["trace"]
        x = np.asarray([row["root_x_m"] for row in trace])
        y = np.asarray([row["root_y_m"] for row in trace])
        goal = np.asarray(data["goal_pose"][:2], dtype=np.float64)
        goal_yaw = float(data["goal_pose"][2])
        ax.plot(x, y, linewidth=2, label="root trajectory")
        ax.scatter(x[0], y[0], color="tab:green", s=60, label="start")
        ax.scatter(x[-1], y[-1], color="tab:red", marker="x", s=70, label="end")
        ax.scatter(goal[0], goal[1], color="tab:purple", marker="*", s=100, label="goal")
        ax.arrow(
            goal[0],
            goal[1],
            0.28 * math.cos(goal_yaw),
            0.28 * math.sin(goal_yaw),
            width=0.01,
            head_width=0.08,
            color="tab:purple",
            length_includes_head=True,
        )
        if data["case"] == "collision":
            ax.scatter(rock_position[0], rock_position[1], color="0.2", marker="D", s=90, label="rock4")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("world X [m]")
        ax.set_ylabel("world Y [m]")
        ax.set_title(f"{data['case']}: {data['termination_reason'] or 'non-terminal'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, name in zip(axes.flat, ("straight", "turn", "collision", "success"), strict=True):
        plot_path(ax, case_data[name])
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(visual_dir / "trajectory.png", dpi=160)
    plt.close(fig)

    for name in ("straight", "turn", "collision", "success"):
        data = case_data[name]
        image = mpimg.imread(data["raw_scene_path"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.55, 1.0]})
        axes[0].imshow(image)
        axes[0].set_title(f"Lunalab / Terrain_lidar01 / Husky — {name}")
        axes[0].axis("off")
        plot_path(axes[1], data)
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(visual_dir / f"{name}.png", dpi=160)
        plt.close(fig)

    time_s = np.asarray([row["v0_time_s"] for row in policy_rows])
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(time_s, [row["action_linear"] for row in policy_rows], label="action linear")
    axes[0].plot(time_s, [row["action_yaw"] for row in policy_rows], label="action yaw")
    axes[0].set_ylabel("normalized action")
    axes[0].legend(loc="best")
    axes[1].plot(time_s, [row["actual_v_m_s"] for row in policy_rows], label="actual v [m/s]")
    axes[1].plot(time_s, [row["actual_omega_rad_s"] for row in policy_rows], label="actual omega [rad/s]")
    axes[1].set_ylabel("body velocity")
    axes[1].legend(loc="best")
    axes[2].plot(time_s, [row["distance_to_goal_m"] for row in policy_rows], label="goal distance")
    axes[2].set_ylabel("distance [m]")
    axes[2].legend(loc="best")
    for key in ("total", "progress", "heading", "action", "tilt", "time", "terminal"):
        axes[3].plot(time_s, [row[f"reward_{key}"] for row in policy_rows], label=key)
    axes[3].set_ylabel("Reward-v0")
    axes[3].set_xlabel("V0 concatenated simulation time [s]")
    axes[3].legend(loc="best", ncol=4, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    boundaries = []
    previous = None
    for row in policy_rows:
        if previous is not None and row["case"] != previous:
            boundaries.append(row["v0_time_s"] - 0.1)
        previous = row["case"]
    for ax in axes:
        for boundary in boundaries:
            ax.axvline(boundary, color="0.5", linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig(visual_dir / "controls_rewards.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    visual_dir = args.output_dir / "visual"
    raw_dir = visual_dir / "raw"
    raw_dir.mkdir(parents=True)
    command = "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python " + " ".join(
        shlex.quote(part) for part in sys.argv
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (args.output_dir / "git_status.txt").write_text(_git_output("status", "--short", "--branch"), encoding="utf-8")
    (args.output_dir / "git_diff.patch").write_text(_git_output("diff", "--binary"), encoding="utf-8")
    (args.output_dir / "host_preflight.txt").write_text(
        _command_output(["uname", "-a"])
        + _command_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv"]
        ),
        encoding="utf-8",
    )
    start_utc = datetime.now(timezone.utc).isoformat()

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
        import omni
        from isaacsim.core.prims import RigidPrim
        from omegaconf import OmegaConf
        from pxr import Usd, UsdGeom, UsdPhysics

        from scripts.rl.probe_rock_collision import _capture_rock_scene
        from scripts.rl.smoke_env import _scene_and_robot, _unwrap_delta, _wait_for_articulation, _yaw_from_xyzw
        from src.environments.static_assets_manager import StaticAssetsManager
        from src.environments_wrappers.rl.env import OmniLRSEnv
        from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, RobotAdapterCfg
        from src.environments_wrappers.rl.isaac_providers import DemTerrainProvider, NamedRigidContactProvider
        from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter
        from src.environments_wrappers.rl.task_runtime import TaskV0Runtime

        robot_values = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "cfg/rl/husky_v0.yaml"), resolve=True)
        adapter_cfg = RobotAdapterCfg.from_mapping(robot_values)
        scene = _scene_and_robot(args, simulation_app, adapter_cfg)
        terrain = DemTerrainProvider(scene["lab"].dem, resolution_m=0.01)

        rock_position = np.asarray([2.60, 2.25, 0.05], dtype=np.float64)
        rock_root_path = "/Lunalab/RLV0Assets/rock4"
        static_assets = StaticAssetsManager(
            {
                "root_path": "/Lunalab/RLV0Assets",
                "parameters": [
                    {
                        "asset_name": "rock4",
                        "usd_path": "USD_Assets/rocks/lunalab_rocks/rock_4/rock4.usd",
                        "pose": {"position": rock_position.tolist(), "orientation": [0.0, 0.0, 0.0, 1.0]},
                        "collision": True,
                        "simulate_physics": False,
                    }
                ],
            }
        )
        static_assets.spawn(
            get_height_func=lambda xy: float(terrain.height(np.asarray([xy], dtype=np.float64))[0])
        )
        rock_position[2] += float(terrain.height(rock_position[None, :2])[0])
        stage = omni.usd.get_context().get_stage()
        rock_root = stage.GetPrimAtPath(rock_root_path)
        rock_collision_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(rock_root)
            if prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not rock_collision_paths:
            raise RuntimeError("V0 rock has no collision geometry")

        scene["world"].reset()
        scene["timeline"].play()
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        contacts = NamedRigidContactProvider(
            robot_prim_path=adapter_cfg.prim_path,
            contact_links=adapter_cfg.contact_links,
            physics_dt_s=1.0 / 30.0,
        )
        rock_views: dict[str, RigidPrim] = {}
        for index, link in enumerate(adapter_cfg.contact_links):
            view = RigidPrim(
                prim_paths_expr=f"{adapter_cfg.prim_path}/{link}",
                name=f"omnilrs_v0_rock_contact_{index}",
                contact_filter_prim_paths_expr=rock_collision_paths,
                max_contact_count=64,
                prepare_contact_sensors=True,
            )
            view.initialize()
            rock_views[link] = view
        adapter.update()

        allowed = set(adapter_cfg.allowed_ground_contact_links)
        current_case = ["reset"]
        contact_rows: list[dict[str, Any]] = []

        def rock_forces():
            result = {}
            for link, view in rock_views.items():
                matrix = np.asarray(view.get_contact_force_matrix(dt=1.0 / 30.0), dtype=np.float64)
                result[link] = np.sum(matrix.reshape(1, len(rock_collision_paths), 3)[0], axis=0)
            return result

        def recorded_contacts(state):
            forces = contacts(state)
            filtered = rock_forces()
            chassis = {name: float(np.linalg.norm(force)) for name, force in forces.items() if name not in allowed}
            rock_chassis = {
                name: float(np.linalg.norm(force)) for name, force in filtered.items() if name not in allowed
            }
            contact_rows.append(
                {
                    "case": current_case[0],
                    "sample": len(contact_rows),
                    "max_chassis_link": max(chassis, key=chassis.get),
                    "max_chassis_contact_n": max(chassis.values()),
                    "max_rock_chassis_link": max(rock_chassis, key=rock_chassis.get),
                    "max_rock_chassis_contact_n": max(rock_chassis.values()),
                }
            )
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

        specs = {
            "straight": {
                "start_pose": (1.25, 6.50, 0.0),
                "goal_pose": (5.25, 6.50, 0.0),
                "action": (0.75, 0.0),
                "max_steps": 12,
                "expected_reason": None,
            },
            "turn": {
                "start_pose": (1.25, 2.25, 0.0),
                "goal_pose": (5.25, 2.25, 0.0),
                "action": (0.0, 1.0),
                "max_steps": 20,
                "expected_reason": None,
            },
            "collision": {
                "start_pose": (1.25, 2.25, 0.0),
                "goal_pose": (5.25, 2.25, 0.0),
                "action": (0.75, 0.0),
                "max_steps": 30,
                "expected_reason": "collision",
            },
            "success": {
                "start_pose": (1.25, 6.50, 0.0),
                "goal_pose": (3.75, 6.50, 0.0),
                "action": (0.50, 0.0),
                "max_steps": 40,
                "expected_reason": "success",
            },
        }
        case_data: dict[str, dict[str, Any]] = {}
        policy_rows: list[dict[str, Any]] = []
        all_trace_rows: list[dict[str, Any]] = []
        v0_step = 0

        for case_name in ("straight", "turn", "collision", "success"):
            spec = specs[case_name]
            current_case[0] = "reset"
            observation, reset_info = env.reset(
                seed=args.seed,
                options={"start_pose": spec["start_pose"], "goal_pose": spec["goal_pose"]},
            )
            current_case[0] = case_name
            initial_state = adapter.get_state()
            initial_yaw = _yaw_from_xyzw(initial_state.orientation_xyzw)
            terminated = truncated = False
            reward = 0.0
            info = reset_info
            action = np.asarray(spec["action"], dtype=np.float32)
            articulation_invalid = 0
            exact_decimation = True
            for case_step in range(1, spec["max_steps"] + 1):
                observation, reward, terminated, truncated, info = env.step(action)
                articulation_invalid += int(not info["articulation_ready"])
                exact_decimation &= info["physics_steps_this_action"] == 6
                trace = env.get_trace()
                last = trace[-1]
                components = info["reward_components"]
                v0_step += 1
                policy_rows.append(
                    {
                        "case": case_name,
                        "case_policy_step": case_step,
                        "v0_policy_step": v0_step,
                        "v0_time_s": v0_step * env_cfg.action.policy_dt_s,
                        "action_linear": float(action[0]),
                        "action_yaw": float(action[1]),
                        "actual_v_m_s": last["actual_v_m_s"],
                        "actual_omega_rad_s": last["actual_omega_rad_s"],
                        "distance_to_goal_m": info["distance_to_goal"],
                        **{f"reward_{key}": value for key, value in components.items()},
                        "termination_reason": info["termination_reason"],
                    }
                )
                if terminated or truncated:
                    break

            trace = env.get_trace()
            for row in trace:
                all_trace_rows.append({"case": case_name, **row})
            final_state = adapter.get_state()
            final_yaw = _yaw_from_xyzw(final_state.orientation_xyzw)
            yaw_delta = _unwrap_delta(final_yaw, initial_yaw)
            displacement = np.asarray(final_state.position[:2]) - np.asarray(initial_state.position[:2])
            expected_reason = spec["expected_reason"]
            case_pass = bool(
                np.all(np.isfinite(observation))
                and observation.shape == (10206,)
                and articulation_invalid == 0
                and exact_decimation
                and info["termination_reason"] == expected_reason
            )
            if case_name == "straight":
                case_pass &= bool(not terminated and not truncated and displacement[0] > 0.50)
            elif case_name == "turn":
                case_pass &= bool(not terminated and not truncated and yaw_delta > math.radians(90.0))
            elif case_name == "collision":
                case_contacts = [row for row in contact_rows if row["case"] == case_name]
                case_pass &= bool(
                    terminated
                    and not truncated
                    and max(row["max_rock_chassis_contact_n"] for row in case_contacts) > 20.0
                )
            elif case_name == "success":
                case_pass &= bool(terminated and not truncated and info["success"])
            if terminated or truncated:
                case_pass &= bool(
                    np.array_equal(info["applied_vehicle_command"], np.zeros(2))
                    and all(value == 0.0 for value in info["drive_targets_rad_s"].values())
                )
            env.emergency_stop()
            raw_path = raw_dir / f"{case_name}.png"
            focus_position = rock_position
            if case_name != "collision":
                focus_position = np.asarray(scene["robot"].get_pose()[0], dtype=np.float64) + np.asarray(
                    [0.75, 0.0, 0.0]
                )
            _capture_rock_scene(scene["world"], scene["robot"], focus_position, raw_path)
            case_data[case_name] = {
                "case": case_name,
                "pass": case_pass,
                "start_pose": spec["start_pose"],
                "goal_pose": spec["goal_pose"],
                "action": spec["action"],
                "policy_steps": info["episode_steps"],
                "termination_reason": info["termination_reason"],
                "terminated": terminated,
                "truncated": truncated,
                "reward": reward,
                "distance_to_goal_m": info["distance_to_goal"],
                "displacement_xy_m": displacement,
                "yaw_delta_rad": yaw_delta,
                "settle_physics_steps": reset_info["settle_physics_steps"],
                "articulation_invalid_count": articulation_invalid,
                "exact_action_decimation": exact_decimation,
                "trace": trace,
                "raw_scene_path": str(raw_path),
            }

        _plot_visual_bundle(case_data, policy_rows, rock_position, visual_dir)
        public_cases = {
            name: {key: value for key, value in data.items() if key not in {"trace", "raw_scene_path"}}
            for name, data in case_data.items()
        }
        required_visuals = [
            visual_dir / "trajectory.png",
            visual_dir / "controls_rewards.png",
            *(visual_dir / f"{name}.png" for name in ("straight", "turn", "collision", "success")),
        ]
        passed = bool(
            all(data["pass"] for data in case_data.values())
            and contacts.is_ready
            and all(view.is_physics_handle_valid() for view in rock_views.values())
            and all(path.is_file() and path.stat().st_size > 0 for path in required_visuals)
        )
        summary = {
            "case": "P1 V0 visual bundle",
            "pass": passed,
            "seed": args.seed,
            "gpu": args.gpu,
            "isaacsim_version": importlib.metadata.version("isaacsim"),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "terrain_id": "Terrain_lidar01",
            "robot_usd": adapter_cfg.usd_path,
            "rock_usd": "assets/USD_Assets/rocks/lunalab_rocks/rock_4/rock4.usd",
            "rock_position_m": rock_position,
            "articulation_ready_physics_step": ready_step,
            "contact_view_ready": contacts.is_ready,
            "rock_contact_views_ready": all(view.is_physics_handle_valid() for view in rock_views.values()),
            "cases": public_cases,
            "required_visuals": [str(path.relative_to(args.output_dir)) for path in required_visuals],
        }
        (args.output_dir / "environment_versions.txt").write_text(
            f"python={sys.version.split()[0]}\n"
            f"isaacsim={summary['isaacsim_version']}\n"
            f"gymnasium={summary['gymnasium_version']}\n",
            encoding="utf-8",
        )
        _write_csv(args.output_dir / "transitions.csv", all_trace_rows)
        _write_csv(args.output_dir / "policy_steps.csv", policy_rows)
        _write_csv(args.output_dir / "contact_samples.csv", contact_rows)
        _write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "summary.md").write_text(
            "# P1 V0 visual summary\n\n"
            f"- pass: `{passed}`\n"
            f"- seed: `{args.seed}`\n"
            f"- GPU: `{args.gpu}`\n"
            "- cases: straight, turn, collision, success\n"
            "- details: `summary.json`, `policy_steps.csv`, `transitions.csv`\n",
            encoding="utf-8",
        )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "case_specs": specs,
                    "rock_position_m": rock_position.tolist(),
                    "rock_collision_paths": rock_collision_paths,
                }
            ),
            args.output_dir / "resolved_config.yaml",
        )
        final_exit_code = 0 if passed else 2
        end_utc = datetime.now(timezone.utc).isoformat()
        _write_json(
            args.output_dir / "manifest.json",
            {
                "protocol": "P1 V0",
                "command": command,
                "commit": _git_output("rev-parse", "HEAD").strip(),
                "diff_sha256": hashlib.sha256((args.output_dir / "git_diff.patch").read_bytes()).hexdigest(),
                "dirty": bool(_git_output("status", "--porcelain").strip()),
                "output_dir": args.output_dir,
                "case": "V0",
                "seed": args.seed,
                "gpu": args.gpu,
                "physics_dt_s": env_cfg.action.physics_dt_s,
                "action_decimation": env_cfg.action.action_decimation,
                "policy_dt_s": env_cfg.action.policy_dt_s,
                "warmup_steps": None,
                "measured_steps": None,
                "terrain_id": "Terrain_lidar01",
                "robot_usd": adapter_cfg.usd_path,
                "sensor_flags": {"camera_observation": False, "heightmap": True, "contact": True},
                "start_utc": start_utc,
                "end_utc": end_utc,
                "exit_code": final_exit_code,
                "validity": "valid" if passed else "invalid",
                "pass": passed,
            },
        )
        print("RL_V0_SUMMARY=" + json.dumps(summary, sort_keys=True, default=_json_default), flush=True)
        exit_code = final_exit_code
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
