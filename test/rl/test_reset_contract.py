from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.rl.run_r0_reset_stress import _history_residual_count
from src.environments_wrappers.rl.env import OmniLRSEnv, _roll_pitch_yaw_quaternion_xyzw
from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg
from src.environments_wrappers.rl.observations import quaternion_yaw_xyzw
from src.environments_wrappers.rl.robot_adapter import RobotState
from src.environments_wrappers.rl.task_runtime import quaternion_roll_pitch_xyzw


class _World:
    def __init__(self):
        self.step_calls: list[bool] = []
        self.reset_calls = 0

    def step(self, render=False):
        self.step_calls.append(bool(render))

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("local episode reset must not call World.reset()")


class _Adapter:
    def __init__(self, drive_joints):
        self.cfg = type("Cfg", (), {"drive_joints": drive_joints})()
        self.events: list[str] = []
        self.last_vehicle_command = np.zeros(2, dtype=np.float32)
        self.last_drive_targets = {name: 9.0 for name in drive_joints}
        self.position = np.asarray([1.5, 2.0, 0.5], dtype=np.float64)
        self.orientation = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.update_calls = 0

    @property
    def is_ready(self):
        return True

    def emergency_stop(self):
        self.events.append("stop")
        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = {name: 0.0 for name in self.cfg.drive_joints}

    def zero_joint_velocities(self):
        self.events.append("zero_joints")

    def teleport_root(self, position, orientation):
        self.events.append("teleport")
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation = np.asarray(orientation, dtype=np.float64)

    def zero_root_velocities(self):
        self.events.append("zero_root")

    def clear_history(self):
        self.events.append("clear")
        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = {name: 0.0 for name in self.cfg.drive_joints}

    def update(self):
        self.update_calls += 1
        return RobotState(
            position=self.position.copy(),
            orientation_xyzw=self.orientation.copy(),
            linear_velocity_world_m_s=np.zeros(3),
            angular_velocity_world_rad_s=np.zeros(3),
            body_forward_velocity_m_s=0.0,
            body_yaw_rate_rad_s=0.0,
            joint_velocities_rad_s={name: 0.0 for name in self.cfg.drive_joints},
        )

    def set_velocity(self, linear_velocity_m_s, yaw_rate_rad_s):
        self.events.append("command")
        self.last_vehicle_command[:] = [linear_velocity_m_s, yaw_rate_rad_s]
        self.last_drive_targets = {name: linear_velocity_m_s for name in self.cfg.drive_joints}


def _make_env():
    cfg = OmniLRSEnvCfg(settle_min_physics_steps=3, settle_max_physics_steps=5)
    world = _World()
    adapter = _Adapter(cfg.robot.drive_joints)
    return OmniLRSEnv(world, adapter, cfg), world, adapter


def test_local_reset_order_and_zero_residuals():
    env, world, adapter = _make_env()
    observation, info = env.reset(seed=42)

    assert adapter.events[:5] == ["stop", "zero_joints", "teleport", "zero_root", "clear"]
    assert world.reset_calls == 0
    assert len(world.step_calls) == 3
    assert info["settle_physics_steps"] == 3
    assert info["reset_pose_projection_applied"]
    assert info["reset_pose_xy_error_m"] == pytest.approx(0.0)
    assert info["reset_pose_yaw_error_rad"] == pytest.approx(0.0)
    assert np.array_equal(info["applied_vehicle_command"], np.zeros(2))
    assert all(value == 0.0 for value in info["drive_targets_rad_s"].values())
    assert env.get_trace() == []
    assert observation.shape == (11,)
    assert observation.dtype == np.float32


def test_reset_pose_projection_quaternion_preserves_roll_pitch_and_sets_yaw():
    expected = (0.13, -0.07, 1.2)
    orientation = _roll_pitch_yaw_quaternion_xyzw(*expected)

    roll, pitch = quaternion_roll_pitch_xyzw(orientation)
    yaw = quaternion_yaw_xyzw(orientation)

    assert (roll, pitch, yaw) == pytest.approx(expected)


def test_r0_history_check_ignores_fresh_current_telemetry_state():
    telemetry = type(
        "Telemetry",
        (),
        {
            "joints_state": {"joint": object()},
            "contacts_state": {"contact": object()},
            "joints_history": {},
            "contacts_history": {},
        },
    )()
    robot = type("Robot", (), {"telemetry": telemetry})()
    assert _history_residual_count(robot) == 0
    telemetry.joints_history = {"joint": [object()]}
    assert _history_residual_count(robot) == 1


def test_each_action_is_held_for_exactly_six_physics_steps_and_slew_limited():
    env, world, adapter = _make_env()
    env.reset(seed=42)
    before = len(world.step_calls)

    _, _, terminated, truncated, info = env.step(np.asarray([1.0, 1.0], dtype=np.float32))

    assert not terminated and not truncated
    assert len(world.step_calls) - before == 6
    assert info["physics_steps_this_action"] == 6
    assert info["elapsed_sim_time"] == pytest.approx(0.2)
    assert info["applied_vehicle_command"] == pytest.approx([0.2, 0.4])
    assert len(env.get_trace()) == 6


def test_benchmark_can_disable_detailed_physics_trace_without_changing_step():
    env, world, adapter = _make_env()
    env.cfg = replace(env.cfg, record_trace=False)
    env.reset(seed=42)
    before = len(world.step_calls)

    _, _, terminated, truncated, info = env.step(np.asarray([0.5, 0.0], dtype=np.float32))

    assert not terminated and not truncated
    assert len(world.step_calls) - before == 6
    assert info["physics_steps_this_action"] == 6
    assert adapter.events[-1] == "command"
    assert env.get_trace() == []


def test_b2_render_cadence_is_one_frame_per_six_physics_steps():
    env, world, _ = _make_env()
    env.cfg = replace(env.cfg, render=True, render_interval_physics_steps=6)
    env.reset(seed=42)
    assert world.step_calls == [False, False, False]

    env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    env.step(np.asarray([0.0, 0.0], dtype=np.float32))

    assert len(world.step_calls) == 15
    assert [index + 1 for index, render in enumerate(world.step_calls) if render] == [6, 12]


@pytest.mark.parametrize(
    "action",
    [np.asarray([np.nan, 0.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32)],
)
def test_invalid_action_triggers_immediate_stop(action):
    env, _, adapter = _make_env()
    env.reset(seed=42)
    adapter.events.clear()

    with pytest.raises(ValueError):
        env.step(action)

    assert adapter.events[-1] == "stop"
    assert all(value == 0.0 for value in adapter.last_drive_targets.values())


def test_terminal_truncation_stops_without_slew():
    env, world, adapter = _make_env()
    env.cfg = replace(env.cfg, episode_policy_steps=1)
    env.reset(seed=42)

    _, _, terminated, truncated, _ = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert not terminated and truncated
    assert adapter.events[-1] == "stop"
    assert all(value == 0.0 for value in adapter.last_drive_targets.values())
    assert world.reset_calls == 0
