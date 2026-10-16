from __future__ import annotations

import numpy as np
import pytest

from src.environments_wrappers.rl.env_cfg import RobotAdapterCfg
from src.environments_wrappers.rl.robot_adapter import DifferentialDriveRobotAdapter


class _ReadyRobot:
    def __init__(self, joint_names: tuple[str, ...]):
        self._joint_names = joint_names

    def is_articulation_api_ready(self):
        return True

    def get_available_joint_names(self):
        return list(self._joint_names)


def test_action_v0_straight_and_left_turn_mapping():
    cfg = RobotAdapterCfg()
    adapter = DifferentialDriveRobotAdapter(_ReadyRobot(cfg.drive_joints), cfg)

    straight = adapter.vehicle_to_joint_targets(0.8, 0.0)
    expected_straight = 0.8 / 0.17775
    assert set(straight) == set(cfg.drive_joints)
    assert all(value == pytest.approx(expected_straight) for value in straight.values())

    turn = adapter.vehicle_to_joint_targets(0.0, 1.2)
    expected = 1.2 * (0.57080 / 2.0) / 0.17775
    assert all(turn[name] == pytest.approx(-expected) for name in cfg.left_drive_joints)
    assert all(turn[name] == pytest.approx(expected) for name in cfg.right_drive_joints)


def test_wheel_speed_clipping_and_per_joint_direction():
    cfg = RobotAdapterCfg(
        left_drive_joints=("l1", "l2", "l3"),
        right_drive_joints=("r1", "r2", "r3"),
        joint_velocity_signs={"l1": 1, "l2": -1, "l3": 1, "r1": 1, "r2": -1, "r3": 1},
    )
    adapter = DifferentialDriveRobotAdapter(_ReadyRobot(cfg.drive_joints), cfg)
    targets = adapter.vehicle_to_joint_targets(100.0, 0.0)

    assert targets == {"l1": 7.0, "l2": -7.0, "l3": 7.0, "r1": 7.0, "r2": -7.0, "r3": 7.0}
    assert len(targets) == 6


@pytest.mark.parametrize("bad", [np.asarray([0.0]), np.asarray([0.0, 0.0, 0.0])])
def test_robot_config_rejects_incomplete_joint_signs(bad):
    names = tuple(f"joint_{index}" for index in range(len(bad) + 2))
    with pytest.raises(ValueError, match="missing joint_velocity_signs"):
        RobotAdapterCfg(
            left_drive_joints=names[:1],
            right_drive_joints=names[1:],
            joint_velocity_signs={names[0]: 1.0},
        )
