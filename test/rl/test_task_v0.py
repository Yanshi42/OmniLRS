import numpy as np
import pytest

from src.environments_wrappers.rl.env_cfg import TaskV0Cfg
from src.environments_wrappers.rl.task import goal_geometry, sample_task_v0


def test_goal_geometry_uses_body_frame_and_wrapped_heading():
    geometry = goal_geometry(
        root_xy_m=np.asarray([1.0, 2.0]),
        root_yaw_rad=np.pi / 2.0,
        goal_xy_m=np.asarray([1.0, 5.0]),
        goal_yaw_rad=-np.pi,
    )

    assert geometry.distance_m == pytest.approx(3.0)
    assert geometry.bearing_body_rad == pytest.approx(0.0)
    assert geometry.heading_error_rad == pytest.approx(np.pi / 2.0)


def test_task_sampler_is_seeded_and_respects_frozen_bounds():
    def heights(points):
        return 0.1 * points[:, 0] - 0.05 * points[:, 1]

    def clear(_xy, _yaw):
        return True

    first = sample_task_v0(np.random.default_rng(42), heights, clear)
    second = sample_task_v0(np.random.default_rng(42), heights, clear)

    assert first == second
    distance = np.hypot(first.goal.x_m - first.start.x_m, first.goal.y_m - first.start.y_m)
    assert 2.0 <= distance <= 8.0
    assert 0.75 <= first.start.x_m <= 5.74
    assert 0.75 <= first.start.y_m <= 9.24
    assert first.start_root_position_m[2] == pytest.approx(first.start.terrain_height_m + 0.50)


def test_task_sampler_fails_instead_of_returning_an_invalid_pose():
    cfg = TaskV0Cfg(max_sampling_attempts=3)

    with pytest.raises(RuntimeError, match="3 attempts"):
        sample_task_v0(
            np.random.default_rng(42),
            lambda points: np.zeros(len(points)),
            lambda _xy, _yaw: False,
            cfg,
        )
