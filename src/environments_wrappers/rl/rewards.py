"""Pure Reward-v0 components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env_cfg import RewardV0Cfg


@dataclass(frozen=True)
class RewardV0Input:
    previous_distance_m: float
    distance_m: float
    previous_heading_error_rad: float
    heading_error_rad: float
    previous_applied_action: np.ndarray
    applied_action: np.ndarray
    roll_rad: float
    pitch_rad: float
    termination_reason: str | None = None


def reward_v0_components(values: RewardV0Input, cfg: RewardV0Cfg | None = None) -> dict[str, float]:
    cfg = cfg or RewardV0Cfg()
    scalars = np.asarray(
        [
            values.previous_distance_m,
            values.distance_m,
            values.previous_heading_error_rad,
            values.heading_error_rad,
            values.roll_rad,
            values.pitch_rad,
        ],
        dtype=np.float64,
    )
    previous_action = np.asarray(values.previous_applied_action, dtype=np.float64)
    action = np.asarray(values.applied_action, dtype=np.float64)
    if not np.all(np.isfinite(scalars)):
        raise ValueError("Reward-v0 scalar inputs must be finite")
    if previous_action.shape != (2,) or action.shape != (2,):
        raise ValueError("Reward-v0 actions must have shape (2,)")
    if not np.all(np.isfinite(previous_action)) or not np.all(np.isfinite(action)):
        raise ValueError("Reward-v0 actions must be finite")

    progress = cfg.progress_weight * float(
        np.clip(values.previous_distance_m - values.distance_m, -cfg.progress_clip_m, cfg.progress_clip_m)
    )
    heading = 0.0
    if values.previous_distance_m < cfg.heading_active_distance_m and values.distance_m < cfg.heading_active_distance_m:
        heading = cfg.heading_weight * (
            abs(values.previous_heading_error_rad) - abs(values.heading_error_rad)
        ) / np.pi
    action_delta = action - previous_action
    action_penalty = -cfg.action_delta_weight * float(np.dot(action_delta, action_delta))
    tilt_ratio = max(abs(values.roll_rad), abs(values.pitch_rad)) / cfg.tilt_reference_rad
    tilt_penalty = -cfg.tilt_weight * float(np.clip(tilt_ratio * tilt_ratio, 0.0, 1.0))
    terminal = 0.0
    if values.termination_reason == "success":
        terminal = cfg.success_reward
    elif values.termination_reason in {"collision", "rollover", "out_of_bounds"}:
        terminal = cfg.failure_penalty
    elif values.termination_reason not in {None, "time_limit"}:
        raise ValueError(f"unknown termination_reason: {values.termination_reason}")

    components = {
        "progress": progress,
        "heading": float(heading),
        "action": action_penalty,
        "tilt": tilt_penalty,
        "time": cfg.time_penalty,
        "terminal": terminal,
    }
    components["total"] = float(sum(components.values()))
    return components
