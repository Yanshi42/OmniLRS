#!/usr/bin/env python3
"""Run the P1 L0/C0 OmniLRS-native lifecycle smoke.

This process talks directly to Isaac Sim. It does not start ROS 2, Yamcs, the
SDG manager, or a training library.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("l0", "c0"), required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-scene", action="store_true")
    return parser.parse_args()


def _json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout


def _yaw_from_xyzw(quaternion) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unwrap_delta(final_yaw: float, initial_yaw: float) -> float:
    return math.atan2(math.sin(final_yaw - initial_yaw), math.cos(final_yaw - initial_yaw))


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


def _scene_and_robot(args, simulation_app, adapter_cfg):
    """Build Lunalab/Terrain_lidar01 and Husky without a transport manager."""

    import numpy as np
    import omni
    from hydra import compose, initialize_config_dir
    from isaacsim.core.api.world import World
    from omegaconf import OmegaConf

    from run import instantiateConfigs, omegaconfToDict
    from src.configurations.simulator_mode_enum import SimulatorMode
    from src.environments.lunalab import LunalabController
    from src.robots.robot import RobotManager

    with initialize_config_dir(config_dir=str(REPO_ROOT / "cfg"), version_base="1.3"):
        composed = compose(
            config_name="config",
            overrides=[
                "environment=lunalab_deformable" if getattr(args, "case", None) == "b3" else "environment=lunalab",
                "robot=husky",
                "environment.physics_dt=0.03333333333333333",
                "environment.rendering_dt=0.03333333333333333",
                "environment.enforce_realtime=false",
            ],
        )
    resolved = OmegaConf.to_container(composed, resolve=True)
    cfg = instantiateConfigs(omegaconfToDict(composed))

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 30.0,
        rendering_dt=1.0 / 30.0,
        backend="numpy",
    )
    physics_context = world.get_physics_context()
    physics_context.set_gravity(-1.62)
    physics_context.set_solver_type("PGS")
    world.reset()

    lab = LunalabController(mode=SimulatorMode.SDG, **cfg["environment"])
    lab.load()
    # Lunalab currently selects terrain id 0 using os.listdir order. Reapply by
    # name so the smoke cannot silently switch to Terrain_lidar02.
    lab.T.loadTerrainByName("Terrain_lidar01")
    lab.load_DEM()
    lab.RM.updateImageData(lab.dem, lab.mask)
    lab.enable_rocks(False)

    x_m, y_m = 1.25, 2.25
    x_index = int(round(x_m / 0.01))
    y_index = int(round(y_m / 0.01))
    terrain_height_m = float(lab.dem[y_index, x_index])
    start_position = np.asarray([x_m, y_m, terrain_height_m + 0.50], dtype=np.float64)
    start_orientation_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    manager = RobotManager(cfg["environment"]["robots_settings"], mode=SimulatorMode.SDG)
    params = manager.robot_parameters
    manager.add_robot(
        usd_path=params.usd_path,
        robot_name=params.robot_name,
        p=start_position,
        q=np.asarray([1.0, 0.0, 0.0, 0.0]),
        domain_id=params.domain_id,
        wheel_joints={"left": list(adapter_cfg.left_drive_joints), "right": list(adapter_cfg.right_drive_joints)},
        camera_conf={},
        imu_sensor_path="",
        dimensions=params.dimensions,
        turn_speed_coef=params.turn_speed_coef,
        pos_relative_to_prim=params.pos_relative_to_prim,
        solar_panel_joint=params.solar_panel_joint,
    )
    robot = manager.robot
    robot.control.init_retry_s = 0.0
    robot.telemetry.init_retry_s = 0.0

    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    return {
        "world": world,
        "timeline": timeline,
        "lab": lab,
        "robot": robot,
        "start_position": start_position,
        "start_orientation_xyzw": start_orientation_xyzw,
        "terrain_height_m": terrain_height_m,
        "resolved_hydra": resolved,
    }


def _wait_for_articulation(world, robot, adapter, *, render: bool) -> int:
    for physics_step in range(1, 181):
        world.step(render=render)
        robot.update_articulation_api()
        if adapter.is_ready:
            return physics_step
    raise RuntimeError(
        "articulation did not become ready within 180 physics steps; "
        f"available={robot.get_available_joint_names()}"
    )


def _run_l0(env, info: dict[str, Any]) -> dict[str, Any]:
    invalid_count = 0
    exact_decimation = True
    for _ in range(5):
        _, _, terminated, truncated, step_info = env.step([0.0, 0.0])
        invalid_count += int(not step_info["articulation_ready"])
        exact_decimation &= step_info["physics_steps_this_action"] == 6
        if terminated or truncated:
            raise RuntimeError("L0 zero-action smoke ended unexpectedly")
    passed = invalid_count == 0 and exact_decimation
    return {
        "case": "L0",
        "pass": passed,
        "articulation_invalid_count": invalid_count,
        "action_decimation_exact": exact_decimation,
        "physics_dt_s": env.cfg.action.physics_dt_s,
        "action_decimation": env.cfg.action.action_decimation,
        "initial_reset": info,
    }


def _run_c0(env, reset_options: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    all_rows: list[dict[str, Any]] = []
    exact_decimation = True
    invalid_count = 0

    straight_obs, straight_reset = env.reset(seed=42, options=reset_options)
    del straight_obs
    straight_start = env.adapter.get_state()
    for _ in range(25):
        _, _, terminated, truncated, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))
        exact_decimation &= info["physics_steps_this_action"] == 6
        invalid_count += int(not info["articulation_ready"])
        if terminated or truncated:
            raise RuntimeError("C0 straight segment ended unexpectedly")
    straight_end = env.adapter.get_state()
    straight_rows = env.get_trace()
    for row in straight_rows:
        row["segment"] = "straight"
    all_rows.extend(straight_rows)

    yaw0 = _yaw_from_xyzw(straight_start.orientation_xyzw)
    delta_world = straight_end.position[:2] - straight_start.position[:2]
    forward = np.asarray([math.cos(yaw0), math.sin(yaw0)])
    left = np.asarray([-math.sin(yaw0), math.cos(yaw0)])
    straight_forward_m = float(np.dot(delta_world, forward))
    straight_lateral_m = float(np.dot(delta_world, left))
    straight_yaw_rad = _unwrap_delta(_yaw_from_xyzw(straight_end.orientation_xyzw), yaw0)

    _, turn_reset = env.reset(seed=42, options=reset_options)
    turn_start = env.adapter.get_state()
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(np.asarray([0.0, 1.0], dtype=np.float32))
        exact_decimation &= info["physics_steps_this_action"] == 6
        invalid_count += int(not info["articulation_ready"])
        if terminated or truncated:
            raise RuntimeError("C0 turn segment ended unexpectedly")
    turn_end = env.adapter.get_state()
    turn_rows = env.get_trace()
    for row in turn_rows:
        row["segment"] = "turn"

    stop_state = env.emergency_stop(physics_steps=60)
    turn_stop_rows = env.get_trace()
    stop_rows = turn_stop_rows[len(turn_rows) :]
    for row in stop_rows:
        row["segment"] = "stop"
    all_rows.extend(turn_rows)
    all_rows.extend(stop_rows)

    turn_yaw_rad = _unwrap_delta(
        _yaw_from_xyzw(turn_end.orientation_xyzw), _yaw_from_xyzw(turn_start.orientation_xyzw)
    )
    turn_xy_drift_m = float(np.linalg.norm(turn_end.position[:2] - turn_start.position[:2]))
    stop_root_speed_m_s = float(np.linalg.norm(stop_state.linear_velocity_world_m_s))
    stop_yaw_rate_rad_s = abs(float(stop_state.body_yaw_rate_rad_s))
    stop_max_wheel_rad_s = max(abs(float(value)) for value in stop_state.joint_velocities_rad_s.values())

    checks = {
        "straight_forward": straight_forward_m >= 1.0,
        "straight_lateral": abs(straight_lateral_m) <= 0.25,
        "straight_yaw": abs(straight_yaw_rad) <= math.radians(10.0),
        "left_turn": turn_yaw_rad >= math.radians(90.0),
        "turn_xy_drift": turn_xy_drift_m <= 0.50,
        "stop_root_speed": stop_root_speed_m_s < 0.05,
        "stop_yaw_rate": stop_yaw_rate_rad_s < 0.05,
        "stop_wheel_speed": stop_max_wheel_rad_s < 0.10,
        "action_decimation": exact_decimation,
        "articulation_lifecycle": invalid_count == 0,
    }
    summary = {
        "case": "C0",
        "pass": all(checks.values()),
        "checks": checks,
        "straight": {
            "forward_m": straight_forward_m,
            "lateral_m": straight_lateral_m,
            "yaw_drift_rad": straight_yaw_rad,
            "yaw_drift_deg": math.degrees(straight_yaw_rad),
            "duration_sim_s": 5.0,
            "reset": straight_reset,
        },
        "turn": {
            "yaw_delta_rad": turn_yaw_rad,
            "yaw_delta_deg": math.degrees(turn_yaw_rad),
            "xy_drift_m": turn_xy_drift_m,
            "duration_sim_s": 4.0,
            "reset": turn_reset,
        },
        "stop": {
            "root_speed_m_s": stop_root_speed_m_s,
            "yaw_rate_rad_s": stop_yaw_rate_rad_s,
            "max_wheel_speed_rad_s": stop_max_wheel_rad_s,
            "duration_sim_s": 2.0,
        },
        "articulation_invalid_count": invalid_count,
        "action_decimation_exact": exact_decimation,
    }
    return summary, all_rows


def _plot_results(rows: list[dict[str, Any]], output_dir: Path, drive_joints: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    visual = output_dir / "visual"
    visual.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    for segment, color in (("straight", "tab:blue"), ("turn", "tab:orange"), ("stop", "tab:red")):
        selected = [row for row in rows if row["segment"] == segment]
        if not selected:
            continue
        x = np.asarray([row["root_x_m"] for row in selected])
        y = np.asarray([row["root_y_m"] for row in selected])
        ax.plot(x, y, color=color, label=segment)
        for index, marker in ((0, "o"), (-1, "X")):
            row = selected[index]
            yaw = _yaw_from_xyzw([row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"]])
            ax.scatter(row["root_x_m"], row["root_y_m"], color=color, marker=marker, s=70)
            ax.arrow(
                row["root_x_m"],
                row["root_y_m"],
                0.25 * math.cos(yaw),
                0.25 * math.sin(yaw),
                color=color,
                width=0.008,
                length_includes_head=True,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.set_title("C0 Husky trajectory (circle=start, X=end, arrows=body +X)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(visual / "trajectory.png", dpi=180)
    plt.close(fig)

    t = np.asarray([row["sim_time_s"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, [row["action_linear"] for row in rows], label="action linear")
    axes[0].plot(t, [row["action_yaw"] for row in rows], label="action yaw")
    axes[0].plot(t, [row["applied_v_m_s"] for row in rows], "--", label="applied v [m/s]")
    axes[0].plot(t, [row["applied_omega_rad_s"] for row in rows], "--", label="applied omega [rad/s]")
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.3)
    for name in drive_joints:
        axes[1].plot(t, [row[f"expected_{name}_rad_s"] for row in rows], "--", alpha=0.65, label=f"{name} target")
        axes[1].plot(t, [row[f"measured_{name}_rad_s"] for row in rows], alpha=0.8, label=f"{name} measured")
    axes[1].set_ylabel("wheel speed [rad/s]")
    axes[1].legend(ncol=2, fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, [row["actual_v_m_s"] for row in rows], label="actual body v")
    axes[2].plot(t, [row["actual_omega_rad_s"] for row in rows], label="actual yaw rate")
    axes[2].set_xlabel("simulation time [s]")
    axes[2].set_ylabel("actual velocity")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Action-v0 command, wheel tracking, and root velocity")
    fig.tight_layout()
    fig.savefig(visual / "controls.png", dpi=180)
    plt.close(fig)


def _capture_scene(world, robot, output_path: Path) -> None:
    import cv2
    import numpy as np
    import omni
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    from scipy.spatial.transform import Rotation

    from WorldBuilders.pxr_utils import addDefaultOps, setDefaultOps

    state_position, state_orientation = robot.get_pose()
    target = np.asarray(state_position, dtype=np.float64) + np.asarray([0.0, 0.0, 0.25])
    orientation = np.asarray(state_orientation, dtype=np.float64)
    body_forward = Rotation.from_quat(orientation).apply([1.0, 0.0, 0.0])
    eye = target - 3.2 * body_forward + np.asarray([0.0, 0.0, 1.8])

    view_direction = target - eye
    view_direction /= np.linalg.norm(view_direction)
    camera_right = np.cross(view_direction, np.asarray([0.0, 0.0, 1.0]))
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, view_direction)
    rotation_matrix = np.column_stack((camera_right, camera_up, -view_direction))
    camera_quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    stage = omni.usd.get_context().get_stage()
    camera_path = "/RL/VisualCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(24.0)
    camera.CreateHorizontalApertureAttr().Set(36.0)
    camera.CreateVerticalApertureAttr().Set(20.25)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    camera_xform = addDefaultOps(camera.GetPrim())
    setDefaultOps(camera_xform, eye, camera_quaternion, (1.0, 1.0, 1.0))
    render_product = rep.create.render_product(camera_path, (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    for _ in range(8):
        world.step(render=True)
        robot.update_articulation_api()
    rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
    rgba = np.asarray(rgb.get_data())
    if rgba.size == 0:
        raise RuntimeError("scene camera returned an empty frame")
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba * (255.0 if rgba.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if not np.any(rgba[..., :3]):
        raise RuntimeError("scene camera returned an all-black frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise RuntimeError(f"failed to write scene image: {output_path}")
    rgb.detach([render_product])
    render_product.destroy()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "visual").mkdir()
    (args.output_dir / "command.txt").write_text(
        "env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python "
        + " ".join(shlex.quote(part) for part in sys.argv)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "git_status.txt").write_text(_git_output("status", "--short", "--branch"), encoding="utf-8")
    (args.output_dir / "git_diff.patch").write_text(_git_output("diff", "--binary"), encoding="utf-8")

    process_start = time.perf_counter()
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
    app_ready = time.perf_counter()

    env = None
    scene = None
    exit_code = 1
    try:
        import numpy as np
        from omegaconf import OmegaConf

        from src.environments_wrappers.rl.env import OmniLRSEnv
        from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, RobotAdapterCfg
        from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter

        robot_values = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "cfg" / "rl" / "husky_v0.yaml"), resolve=True)
        adapter_cfg = RobotAdapterCfg.from_mapping(robot_values)
        scene = _scene_and_robot(args, simulation_app, adapter_cfg)
        scene_ready = time.perf_counter()
        adapter = DifferentialDriveRobotAdapter(scene["robot"], adapter_cfg)
        ready_step = _wait_for_articulation(scene["world"], scene["robot"], adapter, render=False)
        articulation_ready = time.perf_counter()

        env_cfg = OmniLRSEnvCfg(
            robot=adapter_cfg,
            default_start_position=tuple(float(v) for v in scene["start_position"]),
            default_start_orientation_xyzw=tuple(float(v) for v in scene["start_orientation_xyzw"]),
            render=False,
        )
        env = OmniLRSEnv(scene["world"], adapter, env_cfg)
        reset_options = {
            "start_position": scene["start_position"],
            "start_orientation_xyzw": scene["start_orientation_xyzw"],
        }
        _, reset_info = env.reset(seed=args.seed, options=reset_options)
        first_observation = time.perf_counter()

        transitions: list[dict[str, Any]] = []
        if args.case == "l0":
            summary = _run_l0(env, reset_info)
        else:
            summary, transitions = _run_c0(env, reset_options)
            _write_csv(args.output_dir / "transitions.csv", transitions)
            _plot_results(transitions, args.output_dir, adapter_cfg.drive_joints)
            if args.capture_scene:
                _capture_scene(scene["world"], scene["robot"], args.output_dir / "visual" / "husky_lunalab.png")

        summary["startup"] = {
            "process_to_app_ready_s": app_ready - process_start,
            "app_ready_to_scene_ready_s": scene_ready - app_ready,
            "scene_ready_to_articulation_ready_s": articulation_ready - scene_ready,
            "process_to_first_observation_s": first_observation - process_start,
            "articulation_ready_physics_step": ready_step,
        }
        summary["terrain"] = {
            "name": "Terrain_lidar01",
            "height_at_start_m": scene["terrain_height_m"],
        }
        summary["robot_usd"] = adapter_cfg.usd_path
        summary["drive_joints"] = list(adapter_cfg.drive_joints)
        summary["gymnasium_version"] = importlib.metadata.version("gymnasium")
        summary["isaacsim_version"] = importlib.metadata.version("isaacsim")
        summary["gpu"] = args.gpu
        summary["seed"] = args.seed

        resolved_config = {
            "environment": asdict(env_cfg),
            "launch_config": launch_config,
            "case": args.case,
            "terrain": "Terrain_lidar01",
        }
        OmegaConf.save(OmegaConf.create(resolved_config), args.output_dir / "resolved_config.yaml")
        _write_json(args.output_dir / "summary.json", summary)
        _write_json(
            args.output_dir / "manifest.json",
            {
                "case": args.case,
                "command": (args.output_dir / "command.txt").read_text(encoding="utf-8").strip(),
                "commit": _git_output("rev-parse", "HEAD").strip(),
                "dirty": bool(_git_output("status", "--porcelain").strip()),
                "output_dir": args.output_dir,
                "pass": summary["pass"],
            },
        )
        print("RL_SMOKE_SUMMARY=" + json.dumps(summary, sort_keys=True, default=_json_default), flush=True)
        exit_code = 0 if summary["pass"] else 2
    except BaseException:
        failure = traceback.format_exc()
        (args.output_dir / "failure.txt").write_text(failure, encoding="utf-8")
        print(failure, flush=True)
    finally:
        shutdown_start = time.perf_counter()
        if env is not None:
            env.close()
        if scene is not None:
            scene["timeline"].stop()
            scene["world"].stop()
        simulation_app.close()
        shutdown_s = time.perf_counter() - shutdown_start
        print(f"RL_SMOKE_SHUTDOWN_S={shutdown_s:.6f}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
