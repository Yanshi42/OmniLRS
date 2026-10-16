"""Stateful Task-v0 episode runtime built from the pure contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .env_cfg import ObservationV0Cfg, RewardV0Cfg, RobotAdapterCfg, TaskV0Cfg, TerminationV0Cfg
from .observations import build_observation_v0, quaternion_yaw_xyzw
from .rewards import RewardV0Input, reward_v0_components
from .robot_adapter import RobotState
from .task import GoalGeometry, TaskPose2D, TaskV0Sample, goal_geometry, sample_task_v0
from .terminations import (
    TerminationDebounceState,
    TerminationResult,
    evaluate_termination_v0,
    update_termination_debounce,
)

TerrainHeightQuery = Callable[[np.ndarray], np.ndarray]
FootprintClearQuery = Callable[[np.ndarray, float], bool]
ContactForceQuery = Callable[[RobotState], Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class TaskV0StepResult:
    observation: np.ndarray
    reward: float
    termination: TerminationResult
    reward_components: dict[str, float]
    geometry: GoalGeometry


class TaskV0Runtime:
    """Own episode state while keeping Isaac-specific data access injectable."""

    def __init__(
        self,
        *,
        terrain_height_query: TerrainHeightQuery,
        footprint_is_clear: FootprintClearQuery,
        classified_contact_forces: ContactForceQuery,
        robot_cfg: RobotAdapterCfg,
        task_cfg: TaskV0Cfg | None = None,
        observation_cfg: ObservationV0Cfg | None = None,
        reward_cfg: RewardV0Cfg | None = None,
        termination_cfg: TerminationV0Cfg | None = None,
        terrain_id: str = "unknown",
    ) -> None:
        self.terrain_height_query = terrain_height_query
        self.footprint_is_clear = footprint_is_clear
        self.classified_contact_forces = classified_contact_forces
        self.robot_cfg = robot_cfg
        self.task_cfg = task_cfg or TaskV0Cfg()
        self.observation_cfg = observation_cfg or ObservationV0Cfg()
        self.reward_cfg = reward_cfg or RewardV0Cfg()
        self.termination_cfg = termination_cfg or TerminationV0Cfg()
        self.terrain_id = str(terrain_id)

        self.sample: TaskV0Sample | None = None
        self.debounce = TerminationDebounceState()
        self.previous_distance_m = 0.0
        self.previous_heading_error_rad = 0.0
        self.previous_applied_action = np.zeros(2, dtype=np.float32)
        self.last_policy_xy_m: np.ndarray | None = None
        self.path_length_m = 0.0
        self.collision_count = 0
        self.max_roll_rad = 0.0
        self.max_pitch_rad = 0.0
        self.last_reward_components = _zero_reward_components()
        self.last_termination = TerminationResult(False, False, None, False)
        self._collision_was_triggered = False

    @property
    def goal_xy_m(self) -> np.ndarray:
        if self.sample is None:
            raise RuntimeError("Task-v0 episode has not been prepared")
        return np.asarray([self.sample.goal.x_m, self.sample.goal.y_m], dtype=np.float64)

    @property
    def goal_yaw_rad(self) -> float:
        if self.sample is None:
            raise RuntimeError("Task-v0 episode has not been prepared")
        return self.sample.goal.yaw_rad

    def prepare_reset(self, rng: np.random.Generator, options: Mapping[str, Any]) -> TaskV0Sample:
        """Choose and validate the next task, then clear all episode history."""

        has_start = "start_pose" in options
        has_goal = "goal_pose" in options
        if has_start != has_goal:
            raise ValueError("Task-v0 fixed reset requires both start_pose and goal_pose")
        if has_start:
            sample = self._fixed_sample(options["start_pose"], options["goal_pose"])
        else:
            sample = sample_task_v0(rng, self.terrain_height_query, self.footprint_is_clear, self.task_cfg)
        self.sample = sample
        self.clear_episode_history()
        return sample

    def clear_episode_history(self) -> None:
        """Clear contact, reward, path, and termination state without changing the task."""

        self.debounce = TerminationDebounceState()
        self.previous_distance_m = 0.0
        self.previous_heading_error_rad = 0.0
        self.previous_applied_action.fill(0.0)
        self.last_policy_xy_m = None
        self.path_length_m = 0.0
        self.collision_count = 0
        self.max_roll_rad = 0.0
        self.max_pitch_rad = 0.0
        self.last_reward_components = _zero_reward_components()
        self.last_termination = TerminationResult(False, False, None, False)
        self._collision_was_triggered = False

    def begin_episode(self, state: RobotState) -> None:
        geometry = self.geometry(state)
        roll, pitch = quaternion_roll_pitch_xyzw(state.orientation_xyzw)
        self.previous_distance_m = geometry.distance_m
        self.previous_heading_error_rad = geometry.heading_error_rad
        self.previous_applied_action.fill(0.0)
        self.last_policy_xy_m = np.asarray(state.position[:2], dtype=np.float64).copy()
        self.max_roll_rad = abs(roll)
        self.max_pitch_rad = abs(pitch)

    def update_physics(self, state: RobotState) -> None:
        """Advance Task-v0 debounce exactly once per physics step."""

        roll, pitch = quaternion_roll_pitch_xyzw(state.orientation_xyzw)
        self.max_roll_rad = max(self.max_roll_rad, abs(roll))
        self.max_pitch_rad = max(self.max_pitch_rad, abs(pitch))
        previous_triggered = self.debounce.collision_triggered
        self.debounce = update_termination_debounce(
            self.debounce,
            classified_contact_forces_by_link=self.classified_contact_forces(state),
            allowed_ground_contact_links=self.robot_cfg.allowed_ground_contact_links,
            roll_rad=roll,
            pitch_rad=pitch,
            cfg=self.termination_cfg,
        )
        if not previous_triggered and self.debounce.collision_triggered:
            self.collision_count += 1

    def observation(self, state: RobotState, previous_applied_action: np.ndarray) -> np.ndarray:
        return build_observation_v0(
            root_position_m=state.position,
            root_orientation_xyzw=state.orientation_xyzw,
            goal_xy_m=self.goal_xy_m,
            goal_yaw_rad=self.goal_yaw_rad,
            previous_applied_action=previous_applied_action,
            terrain_height_query=self.terrain_height_query,
            cfg=self.observation_cfg,
        )

    def finish_policy_step(
        self,
        state: RobotState,
        *,
        applied_action: np.ndarray,
        episode_policy_steps: int,
    ) -> TaskV0StepResult:
        action = np.asarray(applied_action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("applied_action must be a finite normalized (2,) vector")
        geometry = self.geometry(state)
        roll, pitch = quaternion_roll_pitch_xyzw(state.orientation_xyzw)
        current_xy = np.asarray(state.position[:2], dtype=np.float64)
        if self.last_policy_xy_m is None:
            raise RuntimeError("begin_episode() must be called before finish_policy_step()")
        self.path_length_m += float(np.linalg.norm(current_xy - self.last_policy_xy_m))
        self.last_policy_xy_m = current_xy.copy()

        termination = evaluate_termination_v0(
            debounce=self.debounce,
            root_xy_m=current_xy,
            goal_distance_m=geometry.distance_m,
            goal_heading_error_rad=geometry.heading_error_rad,
            episode_policy_steps=episode_policy_steps,
            cfg=self.termination_cfg,
        )
        components = reward_v0_components(
            RewardV0Input(
                previous_distance_m=self.previous_distance_m,
                distance_m=geometry.distance_m,
                previous_heading_error_rad=self.previous_heading_error_rad,
                heading_error_rad=geometry.heading_error_rad,
                previous_applied_action=self.previous_applied_action,
                applied_action=action,
                roll_rad=roll,
                pitch_rad=pitch,
                termination_reason=termination.reason,
            ),
            self.reward_cfg,
        )
        observation = self.observation(state, action)
        self.previous_distance_m = geometry.distance_m
        self.previous_heading_error_rad = geometry.heading_error_rad
        self.previous_applied_action = action.copy()
        self.last_reward_components = dict(components)
        self.last_termination = termination
        return TaskV0StepResult(observation, components["total"], termination, components, geometry)

    def geometry(self, state: RobotState) -> GoalGeometry:
        return goal_geometry(
            np.asarray(state.position[:2], dtype=np.float64),
            quaternion_yaw_xyzw(state.orientation_xyzw),
            self.goal_xy_m,
            self.goal_yaw_rad,
        )

    def info(self, state: RobotState) -> dict[str, Any]:
        geometry = self.geometry(state)
        return {
            "terrain_id": self.terrain_id,
            "success": self.last_termination.success,
            "termination_reason": self.last_termination.reason,
            "distance_to_goal": geometry.distance_m,
            "heading_error": geometry.heading_error_rad,
            "path_length": self.path_length_m,
            "collision_count": self.collision_count,
            "max_roll": self.max_roll_rad,
            "max_pitch": self.max_pitch_rad,
            "mean_slip": float("nan"),
            "reward_components": dict(self.last_reward_components),
        }

    def _fixed_sample(self, start_value: Any, goal_value: Any) -> TaskV0Sample:
        start = _pose3("start_pose", start_value)
        goal = _pose3("goal_pose", goal_value)
        if not (-np.pi <= start[2] <= np.pi and -np.pi <= goal[2] <= np.pi):
            raise ValueError("fixed Task-v0 yaw must be inside [-pi, pi]")
        xy = np.asarray([start[:2], goal[:2]], dtype=np.float64)
        distance = float(np.linalg.norm(xy[1] - xy[0]))
        cfg = self.task_cfg
        if not np.all((xy[:, 0] >= cfg.sample_x_min_m) & (xy[:, 0] <= cfg.sample_x_max_m)):
            raise ValueError("fixed Task-v0 X is outside the sampling region")
        if not np.all((xy[:, 1] >= cfg.sample_y_min_m) & (xy[:, 1] <= cfg.sample_y_max_m)):
            raise ValueError("fixed Task-v0 Y is outside the sampling region")
        if not cfg.start_goal_distance_min_m <= distance <= cfg.start_goal_distance_max_m:
            raise ValueError("fixed Task-v0 start-goal distance is outside [2, 8] m")
        heights = np.asarray(self.terrain_height_query(xy), dtype=np.float64)
        if heights.shape != (2,) or not np.all(np.isfinite(heights)):
            raise ValueError("fixed Task-v0 poses require finite terrain heights")
        if not self.footprint_is_clear(xy[0].copy(), start[2]):
            raise ValueError("fixed Task-v0 start footprint is not clear")
        if not self.footprint_is_clear(xy[1].copy(), goal[2]):
            raise ValueError("fixed Task-v0 goal footprint is not clear")
        return TaskV0Sample(
            TaskPose2D(start[0], start[1], start[2], float(heights[0])),
            TaskPose2D(goal[0], goal[1], goal[2], float(heights[1])),
        )


def quaternion_roll_pitch_xyzw(quaternion_xyzw: np.ndarray) -> tuple[float, float]:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must be a finite (4,) vector")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("quaternion_xyzw cannot have zero norm")
    x, y, z, w = quaternion / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def yaw_quaternion_xyzw(yaw_rad: float) -> np.ndarray:
    yaw = float(yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError("yaw_rad must be finite")
    return np.asarray([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)], dtype=np.float64)


def _pose3(name: str, value: Any) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        array = np.asarray([value.get("x"), value.get("y"), value.get("yaw")], dtype=np.float64)
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite (x, y, yaw)")
    return float(array[0]), float(array[1]), float(array[2])


def _zero_reward_components() -> dict[str, float]:
    return {name: 0.0 for name in ("progress", "heading", "action", "tilt", "time", "terminal", "total")}
