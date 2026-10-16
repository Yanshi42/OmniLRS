"""Pure Task-v0 termination classification and debounce state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .env_cfg import TerminationV0Cfg


@dataclass(frozen=True)
class TerminationDebounceState:
    collision_physics_steps: int = 0
    rollover_physics_steps: int = 0
    collision_triggered: bool = False
    rollover_triggered: bool = False


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    truncated: bool
    reason: str | None
    success: bool


def update_termination_debounce(
    previous: TerminationDebounceState,
    *,
    classified_contact_forces_by_link: Mapping[str, np.ndarray],
    allowed_ground_contact_links: tuple[str, ...],
    roll_rad: float,
    pitch_rad: float,
    cfg: TerminationV0Cfg | None = None,
) -> TerminationDebounceState:
    """Advance collision/rollover counters by one physics step."""

    cfg = cfg or TerminationV0Cfg()
    if previous.collision_physics_steps < 0 or previous.rollover_physics_steps < 0:
        raise ValueError("debounce counters cannot be negative")
    roll = float(roll_rad)
    pitch = float(pitch_rad)
    if not np.isfinite(roll) or not np.isfinite(pitch):
        raise ValueError("roll and pitch must be finite")

    allowed = set(allowed_ground_contact_links)
    collision = False
    for link_name, force_value in classified_contact_forces_by_link.items():
        force = np.asarray(force_value, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError(f"contact force for {link_name} must be a finite (3,) vector")
        if link_name not in allowed and float(np.linalg.norm(force)) > cfg.collision_force_threshold_n:
            collision = True
    rollover = abs(roll) > cfg.rollover_angle_rad or abs(pitch) > cfg.rollover_angle_rad
    collision_steps = previous.collision_physics_steps + 1 if collision else 0
    rollover_steps = previous.rollover_physics_steps + 1 if rollover else 0
    return TerminationDebounceState(
        collision_physics_steps=collision_steps,
        rollover_physics_steps=rollover_steps,
        collision_triggered=(
            previous.collision_triggered or collision_steps >= cfg.collision_debounce_physics_steps
        ),
        rollover_triggered=(
            previous.rollover_triggered or rollover_steps >= cfg.rollover_debounce_physics_steps
        ),
    )


def evaluate_termination_v0(
    *,
    debounce: TerminationDebounceState,
    root_xy_m: np.ndarray,
    goal_distance_m: float,
    goal_heading_error_rad: float,
    episode_policy_steps: int,
    cfg: TerminationV0Cfg | None = None,
) -> TerminationResult:
    """Apply Task-v0 priority at a policy-step boundary."""

    cfg = cfg or TerminationV0Cfg()
    root_xy = np.asarray(root_xy_m, dtype=np.float64)
    distance = float(goal_distance_m)
    heading_error = float(goal_heading_error_rad)
    if root_xy.shape != (2,) or not np.all(np.isfinite(root_xy)):
        raise ValueError("root_xy_m must be a finite (2,) vector")
    if not np.isfinite(distance) or not np.isfinite(heading_error):
        raise ValueError("goal distance and heading error must be finite")
    if episode_policy_steps < 0:
        raise ValueError("episode_policy_steps cannot be negative")

    reason: str | None = None
    if debounce.collision_triggered or debounce.collision_physics_steps >= cfg.collision_debounce_physics_steps:
        reason = "collision"
    elif debounce.rollover_triggered or debounce.rollover_physics_steps >= cfg.rollover_debounce_physics_steps:
        reason = "rollover"
    elif not (
        cfg.root_x_min_m <= root_xy[0] <= cfg.root_x_max_m
        and cfg.root_y_min_m <= root_xy[1] <= cfg.root_y_max_m
    ):
        reason = "out_of_bounds"
    elif distance <= cfg.success_distance_m and abs(heading_error) <= cfg.success_heading_error_rad:
        reason = "success"
    elif episode_policy_steps >= cfg.episode_policy_steps:
        reason = "time_limit"

    terminated = reason in {"collision", "rollover", "out_of_bounds", "success"}
    truncated = reason == "time_limit"
    return TerminationResult(
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        success=reason == "success",
    )
