"""Minimal OmniLRS-native Gymnasium lifecycle for the P1 L0/C0 smoke."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from .env_cfg import OmniLRSEnvCfg
from .robot_adapter import DifferentialDriveRobotAdapter, RobotState
from .observations import quaternion_yaw_xyzw
from .task_runtime import TaskV0Runtime, quaternion_roll_pitch_xyzw, yaw_quaternion_xyzw


class OmniLRSEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-robot environment using direct Isaac Sim APIs.

    With no task runtime it preserves the 11-value L0/C0 diagnostic boundary.
    Supplying :class:`TaskV0Runtime` enables the frozen Observation/Reward and
    termination contract without coupling the environment to a DEM/contact API.
    """

    metadata = {"render_modes": [None, "human"]}

    def __init__(
        self,
        world: Any,
        robot_adapter: DifferentialDriveRobotAdapter,
        cfg: OmniLRSEnvCfg,
        task_runtime: TaskV0Runtime | None = None,
    ):
        super().__init__()
        self.world = world
        self.adapter = robot_adapter
        self.cfg = cfg
        self.task_runtime = task_runtime
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        if task_runtime is None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(task_runtime.observation_cfg.observation_size,),
                dtype=np.float32,
            )

        self._applied_command = np.zeros(2, dtype=np.float32)
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._previous_applied_action = np.zeros(2, dtype=np.float32)
        self._episode_steps = 0
        self._episode_id = 0
        self._episode_elapsed_sim_s = 0.0
        self._physics_steps_total = 0
        self._terminated = False
        self._truncated = False
        self._trace: list[dict[str, Any]] = []
        self._last_state: RobotState | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if self.task_runtime is None:
            position = np.asarray(options.get("start_position", self.cfg.default_start_position), dtype=np.float64)
            orientation = np.asarray(
                options.get("start_orientation_xyzw", self.cfg.default_start_orientation_xyzw), dtype=np.float64
            )
        else:
            sample = self.task_runtime.prepare_reset(self.np_random, options)
            position = np.asarray(sample.start_root_position_m, dtype=np.float64)
            orientation = yaw_quaternion_xyzw(sample.start.yaw_rad)
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError("start_position must be (3,) and start_orientation_xyzw must be (4,)")
        norm = float(np.linalg.norm(orientation))
        if not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError("start orientation must be a finite non-zero quaternion")
        orientation = orientation / norm

        # Local reset contract: target zero -> joint-state zero -> teleport ->
        # root-state zero -> clear all episode/control/contact history.
        self.adapter.emergency_stop()
        self.adapter.zero_joint_velocities()
        self.adapter.teleport_root(position, orientation)
        self.adapter.zero_root_velocities()
        self.adapter.clear_history()
        self._clear_episode_history()

        settle_steps = 0
        state: RobotState | None = None
        while settle_steps < self.cfg.settle_max_physics_steps:
            self.adapter.emergency_stop()
            self._step_world()
            settle_steps += 1
            state = self.adapter.update()
            if settle_steps >= self.cfg.settle_min_physics_steps and self._reset_state_is_settled(state):
                break
        if state is None or not self._reset_state_is_settled(state):
            self.adapter.emergency_stop()
            raise RuntimeError(
                "local reset did not settle within "
                f"{self.cfg.settle_max_physics_steps} physics steps; "
                f"vz={state.linear_velocity_world_m_s[2] if state else None}"
            )

        # Landing on rough terrain can produce a small XY/yaw drift even after
        # all velocities settle. Preserve the physically settled Z/roll/pitch,
        # project XY/yaw back to the requested episode pose, then zero dynamic
        # state again. No physics step follows this projection, so reset returns
        # the requested pose rather than a post-landing approximation.
        roll, pitch = quaternion_roll_pitch_xyzw(state.orientation_xyzw)
        target_yaw = quaternion_yaw_xyzw(orientation)
        corrected_position = state.position.copy()
        corrected_position[:2] = position[:2]
        corrected_orientation = _roll_pitch_yaw_quaternion_xyzw(roll, pitch, target_yaw)
        self.adapter.emergency_stop()
        self.adapter.zero_joint_velocities()
        self.adapter.teleport_root(corrected_position, corrected_orientation)
        self.adapter.zero_root_velocities()
        state = self.adapter.update()
        xy_error_m = float(np.linalg.norm(state.position[:2] - position[:2]))
        yaw_error_rad = abs(_angle_error(quaternion_yaw_xyzw(state.orientation_xyzw), target_yaw))
        if (
            not self._reset_state_is_settled(state)
            or xy_error_m > 0.02
            or yaw_error_rad > 0.02
        ):
            self.adapter.emergency_stop()
            raise RuntimeError(
                "local reset final pose projection failed; "
                f"xy_error_m={xy_error_m}, yaw_error_rad={yaw_error_rad}"
            )

        # Settle/projection is outside the episode and must leave no
        # action/control/contact history state.
        self.adapter.emergency_stop()
        self.adapter.clear_history()
        self._clear_episode_history()
        self._last_state = state
        if self.task_runtime is not None:
            self.task_runtime.begin_episode(self._last_state)
        self._episode_id += 1
        observation = self._observation(self._last_state)
        info = self._info(settle_steps=settle_steps, physics_steps_this_action=0)
        info["reset_pose_xy_error_m"] = xy_error_m
        info["reset_pose_yaw_error_rad"] = yaw_error_rad
        info["reset_pose_projection_applied"] = True
        return observation, info

    def step(self, action: np.ndarray):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode end; call reset()")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (2,):
            self._safe_emergency_stop()
            raise ValueError(f"Action-v0 requires shape (2,), got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            self._safe_emergency_stop()
            raise ValueError("Action-v0 rejects NaN or Inf")

        clipped = np.clip(action_array, -1.0, 1.0)
        action_clipped = not np.array_equal(clipped, action_array)
        desired = np.asarray(
            [
                clipped[0] * self.cfg.action.linear_velocity_max_m_s,
                clipped[1] * self.cfg.action.yaw_rate_max_rad_s,
            ],
            dtype=np.float32,
        )
        max_delta = np.asarray(
            [
                self.cfg.action.linear_acceleration_max_m_s2 * self.cfg.action.policy_dt_s,
                self.cfg.action.yaw_acceleration_max_rad_s2 * self.cfg.action.policy_dt_s,
            ],
            dtype=np.float32,
        )
        applied = self._applied_command + np.clip(desired - self._applied_command, -max_delta, max_delta)
        applied_normalized = np.asarray(
            [
                applied[0] / self.cfg.action.linear_velocity_max_m_s,
                applied[1] / self.cfg.action.yaw_rate_max_rad_s,
            ],
            dtype=np.float32,
        )

        try:
            self.adapter.set_velocity(float(applied[0]), float(applied[1]))
            for substep in range(self.cfg.action.action_decimation):
                self._step_world()
                self._last_state = self.adapter.update()
                if self.task_runtime is not None:
                    self.task_runtime.update_physics(self._last_state)
                self._append_trace(clipped, applied, substep, self._last_state)
        except Exception:
            self._safe_emergency_stop()
            raise

        try:
            self._previous_action = clipped.copy()
            self._applied_command = applied.astype(np.float32, copy=True)
            self._episode_steps += 1
            self._episode_elapsed_sim_s = self._episode_steps * self.cfg.action.policy_dt_s
            reward = 0.0
            if self.task_runtime is None:
                self._truncated = self._episode_steps >= self.cfg.episode_policy_steps
                observation = self._observation(self._last_state)
            else:
                result = self.task_runtime.finish_policy_step(
                    self._last_state,
                    applied_action=applied_normalized,
                    episode_policy_steps=self._episode_steps,
                )
                observation = result.observation
                reward = result.reward
                self._terminated = result.termination.terminated
                self._truncated = result.termination.truncated
                self._previous_applied_action = applied_normalized.copy()
            if self._terminated or self._truncated:
                self.adapter.emergency_stop()
                self._applied_command.fill(0.0)
            info = self._info(
                action_clipped=action_clipped,
                settle_steps=0,
                physics_steps_this_action=self.cfg.action.action_decimation,
            )
            return observation, reward, self._terminated, self._truncated, info
        except Exception:
            self._safe_emergency_stop()
            raise

    def emergency_stop(self, physics_steps: int = 0) -> RobotState:
        """Write zero immediately and optionally record uncontrolled decay."""

        if physics_steps < 0:
            raise ValueError("physics_steps cannot be negative")
        self.adapter.emergency_stop()
        self._applied_command.fill(0.0)
        zero_action = np.zeros(2, dtype=np.float32)
        state = self.adapter.update()
        for substep in range(physics_steps):
            self._step_world()
            state = self.adapter.update()
            self._append_trace(zero_action, zero_action, substep, state, segment_kind="emergency_stop")
        self._last_state = state
        return state

    def get_trace(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._trace]

    def close(self) -> None:
        self._safe_emergency_stop()

    def _step_world(self) -> None:
        next_physics_step = self._physics_steps_total + 1
        render = bool(
            self.cfg.render
            and next_physics_step % self.cfg.render_interval_physics_steps == 0
        )
        self.world.step(render=render)
        self._physics_steps_total = next_physics_step

    def _safe_emergency_stop(self) -> None:
        try:
            self.adapter.emergency_stop()
        except Exception:
            pass
        self._applied_command.fill(0.0)

    def _clear_episode_history(self) -> None:
        self._applied_command.fill(0.0)
        self._previous_action.fill(0.0)
        self._previous_applied_action.fill(0.0)
        self._episode_steps = 0
        self._episode_elapsed_sim_s = 0.0
        self._terminated = False
        self._truncated = False
        self._trace.clear()
        self._last_state = None
        if self.task_runtime is not None:
            self.task_runtime.clear_episode_history()

    def _reset_state_is_settled(self, state: RobotState) -> bool:
        wheel_speeds = np.asarray(list(state.joint_velocities_rad_s.values()), dtype=np.float64)
        return bool(
            np.linalg.norm(state.linear_velocity_world_m_s) < 0.05
            and np.linalg.norm(state.angular_velocity_world_rad_s) < 0.05
            and np.all(np.isfinite(wheel_speeds))
            and np.all(np.abs(wheel_speeds) < 0.10)
        )

    def _observation(self, state: RobotState) -> np.ndarray:
        if self.task_runtime is not None:
            return self.task_runtime.observation(state, self._previous_applied_action)
        observation = np.concatenate(
            (
                state.position,
                state.orientation_xyzw,
                np.asarray([state.body_forward_velocity_m_s, state.body_yaw_rate_rad_s]),
                self._applied_command,
            )
        ).astype(np.float32)
        if observation.shape != (11,) or not np.all(np.isfinite(observation)):
            self._safe_emergency_stop()
            raise RuntimeError(f"invalid diagnostic observation: {observation}")
        return np.ascontiguousarray(observation)

    def _info(
        self,
        *,
        action_clipped: bool = False,
        settle_steps: int,
        physics_steps_this_action: int,
    ) -> dict[str, Any]:
        state = self._last_state
        info = {
            "episode_id": self._episode_id,
            "episode_steps": self._episode_steps,
            "elapsed_sim_time": self._episode_elapsed_sim_s,
            "physics_steps_total": self._physics_steps_total,
            "physics_steps_this_action": physics_steps_this_action,
            "settle_physics_steps": settle_steps,
            "articulation_ready": self.adapter.is_ready,
            "applied_vehicle_command": self._applied_command.copy(),
            "drive_targets_rad_s": dict(self.adapter.last_drive_targets),
            "measured_wheel_rad_s": dict(state.joint_velocities_rad_s) if state is not None else {},
            "action_clipped": action_clipped,
            "seed": getattr(self, "np_random_seed", None),
            "observation_version": "Observation-v0" if self.task_runtime is not None else self.cfg.observation_version,
            "action_version": self.cfg.action_version,
            "task_version": self.cfg.task_version,
            "termination_reason": "time_limit" if self._truncated else None,
        }
        if state is not None:
            info.update(
                {
                    "root_position_m": state.position.copy(),
                    "root_orientation_xyzw": state.orientation_xyzw.copy(),
                    "actual_v_m_s": state.body_forward_velocity_m_s,
                    "actual_omega_rad_s": state.body_yaw_rate_rad_s,
                }
            )
        if self.task_runtime is not None and state is not None:
            info.update(self.task_runtime.info(state))
        return info

    def _append_trace(
        self,
        normalized_action: np.ndarray,
        applied: np.ndarray,
        substep: int,
        state: RobotState,
        *,
        segment_kind: str = "action",
    ) -> None:
        if not self.cfg.record_trace:
            return
        row: dict[str, Any] = {
            "sim_time_s": float(self._physics_steps_total * self.cfg.action.physics_dt_s),
            "episode_step": self._episode_steps,
            "substep": substep,
            "segment_kind": segment_kind,
            "action_linear": float(normalized_action[0]),
            "action_yaw": float(normalized_action[1]),
            "applied_v_m_s": float(applied[0]),
            "applied_omega_rad_s": float(applied[1]),
            "root_x_m": float(state.position[0]),
            "root_y_m": float(state.position[1]),
            "root_z_m": float(state.position[2]),
            "quat_x": float(state.orientation_xyzw[0]),
            "quat_y": float(state.orientation_xyzw[1]),
            "quat_z": float(state.orientation_xyzw[2]),
            "quat_w": float(state.orientation_xyzw[3]),
            "actual_v_m_s": state.body_forward_velocity_m_s,
            "actual_omega_rad_s": state.body_yaw_rate_rad_s,
        }
        for name in self.cfg.robot.drive_joints:
            row[f"expected_{name}_rad_s"] = float(self.adapter.last_drive_targets[name])
            row[f"measured_{name}_rad_s"] = float(state.joint_velocities_rad_s[name])
        self._trace.append(row)


def _angle_error(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def _roll_pitch_yaw_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
