#!/usr/bin/env python3
"""Drive Husky into a fixed Lunalab rock and verify Task-v0 collision."""

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
    parser.add_argument("--max-policy-steps", type=int, default=30)
    parser.add_argument("--capture-scene", action="store_true")
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


def _plot_results(trace_rows, contact_rows, rock_position, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray([row["root_x_m"] for row in trace_rows])
    y = np.asarray([row["root_y_m"] for row in trace_rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="tab:blue", linewidth=2, label="Husky root")
    ax.scatter(x[0], y[0], marker="o", s=70, color="tab:green", label="start")
    ax.scatter(x[-1], y[-1], marker="x", s=90, color="tab:red", label="collision stop")
    ax.scatter(rock_position[0], rock_position[1], marker="D", s=110, color="0.25", label="rock4 origin")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=160)
    plt.close(fig)

    time_s = np.asarray([row["sim_time_s"] for row in contact_rows])
    named_wheel = np.asarray([row["max_wheel_contact_n"] for row in contact_rows])
    named_chassis = np.asarray([row["max_chassis_contact_n"] for row in contact_rows])
    rock_wheel = np.asarray([row["max_rock_wheel_contact_n"] for row in contact_rows])
    rock_chassis = np.asarray([row["max_rock_chassis_contact_n"] for row in contact_rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time_s, named_wheel, label="all contact: wheel", alpha=0.75)
    ax.plot(time_s, named_chassis, label="all contact: chassis", linewidth=2)
    ax.plot(time_s, rock_wheel, "--", label="rock4 filtered: wheel", alpha=0.75)
    ax.plot(time_s, rock_chassis, "--", label="rock4 filtered: chassis", linewidth=2)
    ax.axhline(20.0, color="tab:red", linestyle=":", label="Task-v0 20 N")
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("net contact force [N]")
    ax.set_yscale("symlog", linthresh=20.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "contact_forces.png", dpi=160)
    plt.close(fig)


def _capture_rock_scene(world, robot, rock_position, output_path: Path) -> None:
    """Capture a side view containing both the stopped Husky and the rock."""

    import cv2
    import numpy as np
    import omni
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation

    from WorldBuilders.pxr_utils import addDefaultOps, setDefaultOps

    robot_position, _ = robot.get_pose()
    target = 0.5 * (np.asarray(robot_position, dtype=np.float64) + np.asarray(rock_position, dtype=np.float64))
    target[2] += 0.25
    eye = target + np.asarray([-2.4, -3.0, 1.8])
    view_direction = target - eye
    view_direction /= np.linalg.norm(view_direction)
    camera_right = np.cross(view_direction, np.asarray([0.0, 0.0, 1.0]))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, view_direction)
    rotation_matrix = np.column_stack((camera_right, camera_up, -view_direction))
    camera_quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    stage = omni.usd.get_context().get_stage()
    camera_path = "/RL/RockCollisionCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(42.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(20.25)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    camera_xform = addDefaultOps(camera.GetPrim())
    setDefaultOps(camera_xform, eye, camera_quaternion, (1.0, 1.0, 1.0))
    render_product = rep.create.render_product(camera_path, (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    rgba = np.empty((0,), dtype=np.uint8)
    for _ in range(5):
        for _ in range(8):
            world.step(render=True)
            robot.update_articulation_api()
        rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
        rgba = np.asarray(rgb.get_data())
        if rgba.size and np.any(rgba[..., :3]):
            break
    if rgba.size == 0:
        raise RuntimeError("rock scene camera returned an empty frame")
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba * (255.0 if rgba.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if not np.any(rgba[..., :3]):
        raise RuntimeError("rock scene camera returned an all-black frame after 5 render attempts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise RuntimeError(f"failed to write rock scene image: {output_path}")
    rgb.detach([render_product])
    render_product.destroy()


def main() -> int:
    args = parse_args()
    if args.max_policy_steps <= 0:
        raise ValueError("max-policy-steps must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "visual").mkdir()
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
        import omni
        from isaacsim.core.prims import RigidPrim
        from omegaconf import OmegaConf
        from pxr import Usd, UsdGeom, UsdPhysics

        from scripts.rl.smoke_env import _scene_and_robot, _wait_for_articulation
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
        rock_root_path = "/Lunalab/RLProbeAssets/rock4"
        static_assets = StaticAssetsManager(
            {
                "root_path": "/Lunalab/RLProbeAssets",
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
            raise RuntimeError(f"no collision geometry found below {rock_root_path}")

        # Rebuild physics once after authoring the fixed obstacle, before any
        # articulation/contact handle is acquired.
        scene["world"].reset()
        scene["timeline"].play()
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        contacts = NamedRigidContactProvider(
            robot_prim_path=adapter_cfg.prim_path,
            contact_links=adapter_cfg.contact_links,
            physics_dt_s=1.0 / 30.0,
        )

        rock_contact_views: dict[str, RigidPrim] = {}
        for index, link in enumerate(adapter_cfg.contact_links):
            view = RigidPrim(
                prim_paths_expr=f"{adapter_cfg.prim_path}/{link}",
                name=f"omnilrs_rock_contact_{index}",
                contact_filter_prim_paths_expr=rock_collision_paths,
                max_contact_count=64,
                prepare_contact_sensors=True,
            )
            view.initialize()
            if view.count != 1:
                raise RuntimeError(f"filtered rock contact view for {link!r} matched {view.count} prims")
            rock_contact_views[link] = view
        adapter.update()

        allowed = set(adapter_cfg.allowed_ground_contact_links)
        contact_rows: list[dict[str, Any]] = []
        physics_sample = [0]

        def filtered_rock_forces() -> dict[str, np.ndarray]:
            result = {}
            for link, view in rock_contact_views.items():
                matrix = np.asarray(view.get_contact_force_matrix(dt=1.0 / 30.0), dtype=np.float64)
                matrix = matrix.reshape(1, len(rock_collision_paths), 3)
                result[link] = np.sum(matrix[0], axis=0)
            return result

        def recorded_contacts(state):
            forces = contacts(state)
            rock_forces = filtered_rock_forces()
            wheel = {name: float(np.linalg.norm(force)) for name, force in forces.items() if name in allowed}
            chassis = {name: float(np.linalg.norm(force)) for name, force in forces.items() if name not in allowed}
            rock_wheel = {
                name: float(np.linalg.norm(force)) for name, force in rock_forces.items() if name in allowed
            }
            rock_chassis = {
                name: float(np.linalg.norm(force)) for name, force in rock_forces.items() if name not in allowed
            }
            wheel_name = max(wheel, key=wheel.get)
            chassis_name = max(chassis, key=chassis.get)
            rock_wheel_name = max(rock_wheel, key=rock_wheel.get)
            rock_chassis_name = max(rock_chassis, key=rock_chassis.get)
            contact_rows.append(
                {
                    "physics_sample": physics_sample[0],
                    "sim_time_s": physics_sample[0] / 30.0,
                    "root_x_m": float(state.position[0]),
                    "root_y_m": float(state.position[1]),
                    "max_wheel_link": wheel_name,
                    "max_wheel_contact_n": wheel[wheel_name],
                    "max_chassis_link": chassis_name,
                    "max_chassis_contact_n": chassis[chassis_name],
                    "max_rock_wheel_link": rock_wheel_name,
                    "max_rock_wheel_contact_n": rock_wheel[rock_wheel_name],
                    "max_rock_chassis_link": rock_chassis_name,
                    "max_rock_chassis_contact_n": rock_chassis[rock_chassis_name],
                    "chassis_above_20n": chassis[chassis_name] > 20.0,
                    "rock_chassis_above_20n": rock_chassis[rock_chassis_name] > 20.0,
                }
            )
            physics_sample[0] += 1
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
        options = {"start_pose": (1.25, 2.25, 0.0), "goal_pose": (5.25, 2.25, 0.0)}
        observation, reset_info = env.reset(seed=args.seed, options=options)
        initial_state = adapter.get_state()

        final = None
        articulation_invalid = 0
        exact_decimation = True
        for _ in range(args.max_policy_steps):
            observation, reward, terminated, truncated, info = env.step(np.asarray([0.75, 0.0], dtype=np.float32))
            articulation_invalid += int(not info["articulation_ready"])
            exact_decimation &= info["physics_steps_this_action"] == 6
            final = (reward, terminated, truncated, info)
            if terminated or truncated:
                break
        if final is None:
            raise RuntimeError("rock collision loop executed no policy steps")
        reward, terminated, truncated, info = final
        trace_rows = env.get_trace()

        named_above = [bool(row["chassis_above_20n"]) for row in contact_rows]
        rock_above = [bool(row["rock_chassis_above_20n"]) for row in contact_rows]
        max_named_chassis = max(float(row["max_chassis_contact_n"]) for row in contact_rows)
        max_rock_chassis = max(float(row["max_rock_chassis_contact_n"]) for row in contact_rows)
        max_rock_wheel = max(float(row["max_rock_wheel_contact_n"]) for row in contact_rows)
        peak_rock_row = max(contact_rows, key=lambda row: float(row["max_rock_chassis_contact_n"]))
        passed = bool(
            info["termination_reason"] == "collision"
            and terminated
            and not truncated
            and articulation_invalid == 0
            and exact_decimation
            and contacts.is_ready
            and all(view.is_physics_handle_valid() for view in rock_contact_views.values())
            and np.all(np.isfinite(observation))
            and observation.shape == (10206,)
            and max_named_chassis > 20.0
            and max_rock_chassis > 20.0
            and _max_consecutive(named_above) >= 2
            and _max_consecutive(rock_above) >= 2
            and np.array_equal(info["applied_vehicle_command"], np.zeros(2))
            and all(value == 0.0 for value in info["drive_targets_rad_s"].values())
        )
        summary = {
            "case": "Task-v0 fixed rock collision",
            "pass": passed,
            "seed": args.seed,
            "gpu": args.gpu,
            "isaacsim_version": importlib.metadata.version("isaacsim"),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "robot_usd": adapter_cfg.usd_path,
            "rock_usd": "assets/USD_Assets/rocks/lunalab_rocks/rock_4/rock4.usd",
            "rock_root_path": rock_root_path,
            "rock_collision_paths": rock_collision_paths,
            "rock_position_m": rock_position,
            "articulation_ready_physics_step": ready_step,
            "articulation_invalid_count": articulation_invalid,
            "contact_view_ready": contacts.is_ready,
            "rock_contact_views_ready": all(view.is_physics_handle_valid() for view in rock_contact_views.values()),
            "episode_policy_steps": info["episode_steps"],
            "physics_contact_samples": len(contact_rows),
            "physics_steps_this_action": info["physics_steps_this_action"],
            "termination_reason": info["termination_reason"],
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "terminal_reward": info["reward_components"]["terminal"],
            "initial_root_position_m": initial_state.position,
            "terminal_root_position_m": adapter.get_state().position,
            "max_named_chassis_contact_n": max_named_chassis,
            "max_named_chassis_consecutive_above_20n": _max_consecutive(named_above),
            "max_rock_chassis_contact_n": max_rock_chassis,
            "max_rock_chassis_consecutive_above_20n": _max_consecutive(rock_above),
            "max_rock_wheel_contact_n": max_rock_wheel,
            "peak_rock_chassis_link": peak_rock_row["max_rock_chassis_link"],
            "drive_targets_zero": all(value == 0.0 for value in info["drive_targets_rad_s"].values()),
            "applied_vehicle_command": info["applied_vehicle_command"],
            "reset_settle_physics_steps": reset_info["settle_physics_steps"],
        }
        _write_csv(args.output_dir / "contact_samples.csv", contact_rows)
        _write_csv(args.output_dir / "trace.csv", trace_rows)
        _plot_results(trace_rows, contact_rows, rock_position, args.output_dir / "visual")
        _write_json(args.output_dir / "summary.json", summary)
        if args.capture_scene:
            _capture_rock_scene(
                scene["world"], scene["robot"], rock_position, args.output_dir / "visual" / "rock_collision.png"
            )
        OmegaConf.save(
            OmegaConf.create(
                {
                    "environment": asdict(env_cfg),
                    "robot": asdict(adapter_cfg),
                    "launch": launch_config,
                    "fixed_task": options,
                    "rock_position_m": rock_position.tolist(),
                    "rock_collision_paths": rock_collision_paths,
                    "max_policy_steps": args.max_policy_steps,
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
        print("RL_ROCK_COLLISION_SUMMARY=" + json.dumps(summary, sort_keys=True, default=_json_default), flush=True)
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
