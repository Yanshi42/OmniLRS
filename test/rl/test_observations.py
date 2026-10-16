import numpy as np
import pytest

from src.environments_wrappers.rl.observations import (
    build_observation_v0,
    heightmap_query_points_world,
    quaternion_yaw_xyzw,
)


def test_observation_v0_shape_slices_and_heightmap_convention():
    def plane(points):
        return points[:, 0] + 2.0 * points[:, 1]

    observation = build_observation_v0(
        root_position_m=np.asarray([3.0, 4.0, 0.5]),
        root_orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        goal_xy_m=np.asarray([7.0, 4.0]),
        goal_yaw_rad=np.pi / 2.0,
        previous_applied_action=np.asarray([0.25, -0.5]),
        terrain_height_query=plane,
    )

    assert observation.shape == (10206,)
    assert observation.dtype == np.float32
    assert observation.flags.c_contiguous
    assert observation[:5] == pytest.approx([0.25, -0.5, 0.5, 0.0, 0.5])
    assert observation[5 + 49 * 101 + 50] == pytest.approx(0.05)
    assert observation[5 + 50 * 101 + 49] == pytest.approx(0.10)
    assert np.all(np.isfinite(observation))
    assert np.max(np.abs(observation)) <= 1.0


def test_heightmap_grid_rotates_with_body_yaw():
    points = heightmap_query_points_world(np.asarray([3.0, 4.0]), np.pi / 2.0)

    assert points[50, 50] == pytest.approx([3.0, 4.0])
    assert points[49, 50] == pytest.approx([3.0, 4.05])
    assert points[50, 49] == pytest.approx([2.95, 4.0])


def test_invalid_height_cells_use_frozen_fill_value():
    def partially_invalid(points):
        heights = np.zeros(len(points))
        heights[points[:, 0] < -1.0] = np.nan
        return heights

    observation = build_observation_v0(
        root_position_m=np.asarray([0.0, 0.0, 0.5]),
        root_orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        goal_xy_m=np.asarray([2.0, 0.0]),
        goal_yaw_rad=0.0,
        previous_applied_action=np.zeros(2),
        terrain_height_query=partially_invalid,
    )

    assert -1.0 in observation[5:]
    assert observation[5 + 50 * 101 + 50] == 0.0


def test_quaternion_boundary_is_explicit_xyzw_and_rejects_zero_norm():
    assert quaternion_yaw_xyzw(np.asarray([0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])) == pytest.approx(
        np.pi / 2.0
    )
    with pytest.raises(ValueError, match="zero norm"):
        quaternion_yaw_xyzw(np.zeros(4))
