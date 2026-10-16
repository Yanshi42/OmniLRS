from __future__ import annotations

import numpy as np
import pytest

from src.environments_wrappers.rl.env import OmniLRSEnv
from src.environments_wrappers.rl.env_cfg import OmniLRSEnvCfg, TerminationV0Cfg
from src.environments_wrappers.rl.robot_adapter import RobotState
from src.environments_wrappers.rl.task_runtime import TaskV0Runtime


class _World:
    def __init__(self):
        self.step_calls = 0
        self.reset_calls = 0

    def step(self, render=False):
        self.step_calls += 1

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("episode reset must remain local")


class _Adapter:
    def __init__(self, drive_joints):
        self.cfg = type("Cfg", (), {"drive_joints": drive_joints})()
        self.last_vehicle_command = np.zeros(2, dtype=np.float32)
        self.last_drive_targets = {name: 0.0 for name in drive_joints}
        self.position = np.asarray([1.0, 2.0, 0.5], dtype=np.float64)
        self.orientation = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.stop_calls = 0

    @property
    def is_ready(self):
        return True

    def emergency_stop(self):
        self.stop_calls += 1
        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = {name: 0.0 for name in self.cfg.drive_joints}

    def zero_joint_velocities(self):
        pass

    def teleport_root(self, position, orientation):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.orientation = np.asarray(orientation, dtype=np.float64).copy()

    def zero_root_velocities(self):
        pass

    def clear_history(self):
        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = {name: 0.0 for name in self.cfg.drive_joints}

    def set_velocity(self, linear_velocity_m_s, yaw_rate_rad_s):
        self.last_vehicle_command[:] = [linear_velocity_m_s, yaw_rate_rad_s]
        self.last_drive_targets = {name: linear_velocity_m_s for name in self.cfg.drive_joints}

    def update(self):
        # Deterministic fake dynamics only for the contract test.
        self.position[0] += float(self.last_vehicle_command[0]) * (1.0 / 30.0)
        return RobotState(
            position=self.position.copy(),
            orientation_xyzw=self.orientation.copy(),
            linear_velocity_world_m_s=np.asarray([self.last_vehicle_command[0], 0.0, 0.0]),
            angular_velocity_world_rad_s=np.asarray([0.0, 0.0, self.last_vehicle_command[1]]),
            body_forward_velocity_m_s=float(self.last_vehicle_command[0]),
            body_yaw_rate_rad_s=float(self.last_vehicle_command[1]),
            joint_velocities_rad_s={name: float(self.last_drive_targets[name]) for name in self.cfg.drive_joints},
        )


def _terrain_height(points):
    return np.zeros(np.asarray(points).shape[0], dtype=np.float64)


def _make_task_env(*, collision_enabled, episode_policy_steps=300):
    cfg = OmniLRSEnvCfg(settle_min_physics_steps=1, settle_max_physics_steps=2)
    world = _World()
    adapter = _Adapter(cfg.robot.drive_joints)

    def contacts(_state):
        if collision_enabled[0]:
            return {"base_link": np.asarray([25.0, 0.0, 0.0])}
        return {
            cfg.robot.allowed_ground_contact_links[0]: np.asarray([100.0, 0.0, 0.0]),
            "base_link": np.zeros(3),
        }

    runtime = TaskV0Runtime(
        terrain_height_query=_terrain_height,
        footprint_is_clear=lambda _xy, _yaw: True,
        classified_contact_forces=contacts,
        robot_cfg=cfg.robot,
        termination_cfg=TerminationV0Cfg(episode_policy_steps=episode_policy_steps),
        terrain_id="unit-flat",
    )
    return OmniLRSEnv(world, adapter, cfg, runtime), world, adapter, runtime


_FIXED_TASK = {"start_pose": (1.0, 2.0, 0.0), "goal_pose": (3.5, 2.0, 0.0)}


def test_task_runtime_connects_observation_reward_and_exact_decimation():
    env, world, _, runtime = _make_task_env(collision_enabled=[False])
    observation, info = env.reset(seed=7, options=_FIXED_TASK)
    before = world.step_calls

    assert observation.shape == (10206,)
    assert observation.dtype == np.float32
    assert observation.flags.c_contiguous
    assert np.array_equal(observation[:2], np.zeros(2))
    assert info["observation_version"] == "Observation-v0"
    assert info["terrain_id"] == "unit-flat"

    observation, reward, terminated, truncated, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert world.step_calls - before == 6
    assert info["physics_steps_this_action"] == 6
    assert not terminated and not truncated
    assert observation[:2] == pytest.approx([0.25, 0.0])
    assert reward == pytest.approx(info["reward_components"]["total"])
    assert info["reward_components"]["progress"] > 0.0
    assert info["path_length"] > 0.0
    assert runtime.debounce.collision_physics_steps == 0


def test_collision_debounce_terminal_observation_and_immediate_stop_then_reset():
    collision_enabled = [True]
    env, world, adapter, runtime = _make_task_env(collision_enabled=collision_enabled)
    env.reset(seed=11, options=_FIXED_TASK)
    before = world.step_calls

    observation, reward, terminated, truncated, info = env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert world.step_calls - before == 6
    assert terminated and not truncated
    assert info["termination_reason"] == "collision"
    assert info["collision_count"] == 1
    assert info["reward_components"]["terminal"] == -10.0
    assert reward < -9.0
    assert observation[:2] == pytest.approx([0.25, 0.0])
    assert np.array_equal(info["applied_vehicle_command"], np.zeros(2))
    assert all(value == 0.0 for value in adapter.last_drive_targets.values())

    collision_enabled[0] = False
    observation, info = env.reset(seed=11, options=_FIXED_TASK)
    assert np.array_equal(observation[:2], np.zeros(2))
    assert runtime.debounce.collision_physics_steps == 0
    assert not runtime.debounce.collision_triggered
    assert info["collision_count"] == 0
    assert info["path_length"] == 0.0
    assert info["termination_reason"] is None
    assert world.reset_calls == 0


def test_task_time_limit_is_truncation_and_stops_immediately():
    env, _, adapter, _ = _make_task_env(collision_enabled=[False], episode_policy_steps=1)
    env.reset(seed=5, options=_FIXED_TASK)

    observation, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    assert info["reward_components"]["terminal"] == 0.0
    assert np.array_equal(observation[:2], np.zeros(2))
    assert all(value == 0.0 for value in adapter.last_drive_targets.values())


def test_post_physics_runtime_error_triggers_immediate_stop():
    env, _, adapter, runtime = _make_task_env(collision_enabled=[False])
    env.reset(seed=5, options=_FIXED_TASK)

    def fail_height_query(_points):
        raise RuntimeError("terrain provider failed")

    runtime.terrain_height_query = fail_height_query
    with pytest.raises(RuntimeError, match="terrain provider failed"):
        env.step(np.asarray([1.0, 0.0], dtype=np.float32))

    assert np.array_equal(adapter.last_vehicle_command, np.zeros(2))
    assert all(value == 0.0 for value in adapter.last_drive_targets.values())


def test_fixed_task_reset_validates_both_poses_and_sampling_region():
    env, _, _, _ = _make_task_env(collision_enabled=[False])

    with pytest.raises(ValueError, match="both start_pose and goal_pose"):
        env.reset(seed=1, options={"start_pose": (1.0, 2.0, 0.0)})
    with pytest.raises(ValueError, match="outside the sampling region"):
        env.reset(
            seed=1,
            options={"start_pose": (0.0, 2.0, 0.0), "goal_pose": (3.0, 2.0, 0.0)},
        )
    with pytest.raises(ValueError, match="yaw must be inside"):
        env.reset(
            seed=1,
            options={"start_pose": (1.0, 2.0, 4.0), "goal_pose": (3.5, 2.0, 0.0)},
        )
