"""Frozen P1 configuration for the native Gymnasium environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ActionV0Cfg:
    """Task-v0 vehicle command and timing constants."""

    linear_velocity_max_m_s: float = 0.8
    yaw_rate_max_rad_s: float = 1.2
    linear_acceleration_max_m_s2: float = 1.0
    yaw_acceleration_max_rad_s2: float = 2.0
    physics_dt_s: float = 1.0 / 30.0
    action_decimation: int = 6

    @property
    def policy_dt_s(self) -> float:
        return self.physics_dt_s * self.action_decimation


@dataclass(frozen=True)
class TaskV0Cfg:
    sample_x_min_m: float = 0.75
    sample_x_max_m: float = 5.74
    sample_y_min_m: float = 0.75
    sample_y_max_m: float = 9.24
    start_goal_distance_min_m: float = 2.0
    start_goal_distance_max_m: float = 8.0
    max_sampling_attempts: int = 1000


@dataclass(frozen=True)
class ObservationV0Cfg:
    heightmap_size: int = 101
    heightmap_resolution_m: float = 0.05
    height_normalization_m: float = 1.0
    invalid_height_value: float = -1.0
    goal_distance_normalization_m: float = 8.0

    def __post_init__(self) -> None:
        if self.heightmap_size <= 0 or self.heightmap_size % 2 != 1:
            raise ValueError("heightmap_size must be a positive odd integer")
        if self.heightmap_resolution_m <= 0.0 or self.height_normalization_m <= 0.0:
            raise ValueError("height-map resolution and normalization must be positive")
        if self.goal_distance_normalization_m <= 0.0:
            raise ValueError("goal-distance normalization must be positive")

    @property
    def observation_size(self) -> int:
        return 5 + self.heightmap_size * self.heightmap_size


@dataclass(frozen=True)
class RewardV0Cfg:
    progress_weight: float = 5.0
    progress_clip_m: float = 0.25
    heading_weight: float = 0.5
    heading_active_distance_m: float = 1.0
    action_delta_weight: float = 0.02
    tilt_weight: float = 0.05
    tilt_reference_rad: float = np.deg2rad(45.0)
    time_penalty: float = -0.01
    success_reward: float = 10.0
    failure_penalty: float = -10.0


@dataclass(frozen=True)
class TerminationV0Cfg:
    collision_force_threshold_n: float = 20.0
    collision_debounce_physics_steps: int = 2
    rollover_angle_rad: float = np.deg2rad(60.0)
    rollover_debounce_physics_steps: int = 3
    root_x_min_m: float = 0.45
    root_x_max_m: float = 6.04
    root_y_min_m: float = 0.45
    root_y_max_m: float = 9.54
    success_distance_m: float = 0.50
    success_heading_error_rad: float = np.deg2rad(20.0)
    episode_policy_steps: int = 300


@dataclass(frozen=True)
class RobotAdapterCfg:
    """Vehicle geometry and named drive-joint configuration.

    ``joint_velocity_signs`` maps a positive mathematical wheel speed to the
    sign required by each USD joint. Keeping it next to the joint names makes
    a four- or six-wheel vehicle a configuration change rather than an MDP
    change.
    """

    prim_path: str = "/Robots/husky"
    usd_path: str = "assets/USD_Assets/robots/nograph_husky.usd"
    left_drive_joints: tuple[str, ...] = ("front_left_wheel", "rear_left_wheel")
    right_drive_joints: tuple[str, ...] = ("front_right_wheel", "rear_right_wheel")
    joint_velocity_signs: Mapping[str, float] = field(
        default_factory=lambda: {
            "front_left_wheel": 1.0,
            "rear_left_wheel": 1.0,
            "front_right_wheel": 1.0,
            "rear_right_wheel": 1.0,
        }
    )
    wheel_radius_m: float = 0.17775
    wheel_track_m: float = 0.57080
    wheel_speed_limit_rad_s: float = 7.0
    contact_links: tuple[str, ...] = (
        "base_footprint",
        "base_link",
        "front_bumper_link",
        "front_left_wheel_link",
        "front_right_wheel_link",
        "imu_link",
        "rear_bumper_link",
        "rear_left_wheel_link",
        "rear_right_wheel_link",
        "top_plate_link",
        "user_rail_link",
        "d455/RSD455",
    )
    allowed_ground_contact_links: tuple[str, ...] = (
        "front_left_wheel_link",
        "rear_left_wheel_link",
        "front_right_wheel_link",
        "rear_right_wheel_link",
    )

    def __post_init__(self) -> None:
        drive_joints = self.drive_joints
        if not self.left_drive_joints or not self.right_drive_joints:
            raise ValueError("both left and right drive-joint lists must be non-empty")
        if len(set(drive_joints)) != len(drive_joints):
            raise ValueError("drive-joint names must be unique")
        missing = [name for name in drive_joints if name not in self.joint_velocity_signs]
        if missing:
            raise ValueError(f"missing joint_velocity_signs entries: {missing}")
        if self.wheel_radius_m <= 0.0 or self.wheel_track_m <= 0.0:
            raise ValueError("wheel radius and track must be positive")
        if self.wheel_speed_limit_rad_s <= 0.0:
            raise ValueError("wheel speed limit must be positive")
        if not self.contact_links or len(set(self.contact_links)) != len(self.contact_links):
            raise ValueError("contact_links must be non-empty and unique")
        unknown_allowed = set(self.allowed_ground_contact_links) - set(self.contact_links)
        if unknown_allowed:
            raise ValueError(f"allowed ground-contact links are not monitored: {sorted(unknown_allowed)}")

    @property
    def drive_joints(self) -> tuple[str, ...]:
        return self.left_drive_joints + self.right_drive_joints

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RobotAdapterCfg":
        data = dict(values)
        for key in ("left_drive_joints", "right_drive_joints", "contact_links", "allowed_ground_contact_links"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class OmniLRSEnvCfg:
    """P1 lifecycle configuration shared by diagnostic and Task-v0 modes.

    ``observation_version`` names the L0/C0 diagnostic boundary. An attached
    :class:`TaskV0Runtime` exposes the exact ``Observation-v0`` version instead.
    """

    action: ActionV0Cfg = field(default_factory=ActionV0Cfg)
    robot: RobotAdapterCfg = field(default_factory=RobotAdapterCfg)
    episode_policy_steps: int = 300
    settle_min_physics_steps: int = 15
    settle_max_physics_steps: int = 60
    settle_vertical_speed_m_s: float = 0.05
    default_start_position: tuple[float, float, float] = (1.5, 2.0, 0.5)
    default_start_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    render: bool = False
    render_interval_physics_steps: int = 1
    record_trace: bool = True
    observation_version: str = "DiagnosticObservation-P1"
    action_version: str = "Action-v0"
    task_version: str = "Task-v0"

    def __post_init__(self) -> None:
        if self.action.action_decimation <= 0:
            raise ValueError("action_decimation must be positive")
        if self.action.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if self.render_interval_physics_steps <= 0:
            raise ValueError("render_interval_physics_steps must be positive")
        if self.settle_min_physics_steps < 0:
            raise ValueError("settle_min_physics_steps cannot be negative")
        if self.settle_max_physics_steps < self.settle_min_physics_steps:
            raise ValueError("settle_max_physics_steps must be >= settle_min_physics_steps")
