"""Vehicle-level adapter between Action-v0 and an OmniLRS robot articulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .env_cfg import RobotAdapterCfg


@dataclass(frozen=True)
class RobotState:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray
    body_forward_velocity_m_s: float
    body_yaw_rate_rad_s: float
    joint_velocities_rad_s: dict[str, float]


class DifferentialDriveRobotAdapter:
    """Configurable differential/skid-steer adapter.

    The wrapped object is the existing :class:`src.robots.robot.Robot`. It is
    intentionally duck-typed here so action mapping and reset behavior remain
    testable without starting Isaac Sim.
    """

    def __init__(self, robot: Any, cfg: RobotAdapterCfg):
        self.robot = robot
        self.cfg = cfg
        self.last_vehicle_command = np.zeros(2, dtype=np.float32)
        self.last_drive_targets = {name: 0.0 for name in cfg.drive_joints}

    @property
    def is_ready(self) -> bool:
        if not bool(self.robot.is_articulation_api_ready()):
            return False
        available = set(self.robot.get_available_joint_names())
        return all(name in available for name in self.cfg.drive_joints)

    def update(self) -> RobotState:
        self.robot.update_articulation_api()
        if not self.is_ready:
            raise RuntimeError(
                "robot articulation is not ready or configured drive joints are missing: "
                f"configured={self.cfg.drive_joints}, available={self.robot.get_available_joint_names()}"
            )
        return self.get_state()

    def vehicle_to_joint_targets(self, linear_velocity_m_s: float, yaw_rate_rad_s: float) -> dict[str, float]:
        radius = self.cfg.wheel_radius_m
        half_track = 0.5 * self.cfg.wheel_track_m
        left = (float(linear_velocity_m_s) - float(yaw_rate_rad_s) * half_track) / radius
        right = (float(linear_velocity_m_s) + float(yaw_rate_rad_s) * half_track) / radius
        limit = self.cfg.wheel_speed_limit_rad_s
        left = float(np.clip(left, -limit, limit))
        right = float(np.clip(right, -limit, limit))

        targets: dict[str, float] = {}
        for name in self.cfg.left_drive_joints:
            targets[name] = left * float(self.cfg.joint_velocity_signs[name])
        for name in self.cfg.right_drive_joints:
            targets[name] = right * float(self.cfg.joint_velocity_signs[name])
        return targets

    def set_velocity(self, linear_velocity_m_s: float, yaw_rate_rad_s: float) -> dict[str, float]:
        if not self.is_ready:
            raise RuntimeError("cannot command a robot whose articulation is not ready")
        targets = self.vehicle_to_joint_targets(linear_velocity_m_s, yaw_rate_rad_s)
        if not self.robot.set_joint_velocities(targets):
            raise RuntimeError(f"failed to apply drive targets: {targets}")
        self.last_vehicle_command[:] = (linear_velocity_m_s, yaw_rate_rad_s)
        self.last_drive_targets = dict(targets)
        return targets

    def emergency_stop(self) -> None:
        """Immediately write zero drive targets, bypassing any slew limiter."""

        targets = {name: 0.0 for name in self.cfg.drive_joints}
        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = dict(targets)
        if self.is_ready and not self.robot.set_joint_velocities(targets):
            raise RuntimeError("failed to apply emergency-stop drive targets")

    def zero_joint_velocities(self) -> None:
        """Immediately zero the physical state of every configured drive joint."""

        if not self.is_ready:
            raise RuntimeError("cannot zero joint state before articulation is ready")
        articulation = self.robot.control.articulation
        index_by_name = self.robot.control.joint_name_to_index
        indices = np.asarray([index_by_name[name] for name in self.cfg.drive_joints], dtype=np.int64)
        articulation.set_joint_velocities(np.zeros(len(indices), dtype=np.float32), joint_indices=indices)

    def teleport_root(self, position: np.ndarray, orientation_xyzw: np.ndarray) -> None:
        self.robot.teleport(position.tolist(), orientation_xyzw.tolist())

    def zero_root_velocities(self) -> None:
        if not self.is_ready:
            raise RuntimeError("cannot zero root state before articulation is ready")
        articulation = self.robot.control.articulation
        articulation.set_linear_velocity(np.zeros(3, dtype=np.float32))
        articulation.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def clear_history(self) -> None:
        """Clear adapter and existing telemetry history without rebuilding views."""

        self.last_vehicle_command.fill(0.0)
        self.last_drive_targets = {name: 0.0 for name in self.cfg.drive_joints}
        telemetry = getattr(self.robot, "telemetry", None)
        if telemetry is not None:
            for attr in ("joints_state", "contacts_state", "joints_history", "contacts_history"):
                value = getattr(telemetry, attr, None)
                if hasattr(value, "clear"):
                    value.clear()

    def get_state(self) -> RobotState:
        if not self.is_ready:
            raise RuntimeError("cannot read state before articulation is ready")
        position, orientation_xyzw = self.robot.get_pose()
        articulation = self.robot.control.articulation
        linear_world = np.asarray(articulation.get_linear_velocity(), dtype=np.float64)
        angular_world = np.asarray(articulation.get_angular_velocity(), dtype=np.float64)
        orientation_xyzw = np.asarray(orientation_xyzw, dtype=np.float64)
        x, y, z, w = orientation_xyzw
        forward_world = np.asarray(
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
            dtype=np.float64,
        )
        cached = self.robot.get_joint_velocities()
        joint_velocities = {name: float(cached.get(name, np.nan)) for name in self.cfg.drive_joints}
        return RobotState(
            position=np.asarray(position, dtype=np.float64),
            orientation_xyzw=orientation_xyzw,
            linear_velocity_world_m_s=linear_world,
            angular_velocity_world_rad_s=angular_world,
            body_forward_velocity_m_s=float(np.dot(linear_world, forward_world)),
            body_yaw_rate_rad_s=float(angular_world[2]),
            joint_velocities_rad_s=joint_velocities,
        )
