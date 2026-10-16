"""Pure Task-v0 sampling and goal-geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .env_cfg import TaskV0Cfg


@dataclass(frozen=True)
class TaskPose2D:
    x_m: float
    y_m: float
    yaw_rad: float
    terrain_height_m: float


@dataclass(frozen=True)
class TaskV0Sample:
    start: TaskPose2D
    goal: TaskPose2D

    @property
    def start_root_position_m(self) -> tuple[float, float, float]:
        return (self.start.x_m, self.start.y_m, self.start.terrain_height_m + 0.50)


@dataclass(frozen=True)
class GoalGeometry:
    distance_m: float
    bearing_body_rad: float
    heading_error_rad: float


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)


def goal_geometry(
    root_xy_m: np.ndarray,
    root_yaw_rad: float,
    goal_xy_m: np.ndarray,
    goal_yaw_rad: float,
) -> GoalGeometry:
    root_xy = _finite_vector("root_xy_m", root_xy_m, 2)
    goal_xy = _finite_vector("goal_xy_m", goal_xy_m, 2)
    yaw = _finite_scalar("root_yaw_rad", root_yaw_rad)
    goal_yaw = _finite_scalar("goal_yaw_rad", goal_yaw_rad)

    delta_world = goal_xy - root_xy
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    x_body = cosine * delta_world[0] + sine * delta_world[1]
    y_body = -sine * delta_world[0] + cosine * delta_world[1]
    return GoalGeometry(
        distance_m=float(np.linalg.norm(delta_world)),
        bearing_body_rad=wrap_to_pi(np.arctan2(y_body, x_body)),
        heading_error_rad=wrap_to_pi(goal_yaw - yaw),
    )


def sample_task_v0(
    rng: np.random.Generator,
    terrain_height_query: Callable[[np.ndarray], np.ndarray],
    footprint_is_clear: Callable[[np.ndarray, float], bool],
    cfg: TaskV0Cfg | None = None,
) -> TaskV0Sample:
    """Sample a valid Task-v0 start/goal pair using only the supplied RNG."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    cfg = cfg or TaskV0Cfg()
    for _ in range(cfg.max_sampling_attempts):
        xy = rng.uniform(
            low=(cfg.sample_x_min_m, cfg.sample_y_min_m),
            high=(cfg.sample_x_max_m, cfg.sample_y_max_m),
            size=(2, 2),
        )
        yaws = rng.uniform(-np.pi, np.pi, size=2)
        distance = float(np.linalg.norm(xy[1] - xy[0]))
        if not (cfg.start_goal_distance_min_m <= distance <= cfg.start_goal_distance_max_m):
            continue

        heights = np.asarray(terrain_height_query(xy), dtype=np.float64)
        if heights.shape != (2,) or not np.all(np.isfinite(heights)):
            continue
        if not footprint_is_clear(xy[0].copy(), float(yaws[0])):
            continue
        if not footprint_is_clear(xy[1].copy(), float(yaws[1])):
            continue
        return TaskV0Sample(
            start=TaskPose2D(float(xy[0, 0]), float(xy[0, 1]), float(yaws[0]), float(heights[0])),
            goal=TaskPose2D(float(xy[1, 0]), float(xy[1, 1]), float(yaws[1]), float(heights[1])),
        )
    raise RuntimeError(f"failed to sample a valid Task-v0 start/goal pair in {cfg.max_sampling_attempts} attempts")


def _finite_vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({length},) vector")
    return array


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar
