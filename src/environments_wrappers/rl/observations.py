"""Pure Observation-v0 construction for OmniLRS-native RL."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .env_cfg import ObservationV0Cfg
from .task import goal_geometry


def quaternion_yaw_xyzw(quaternion_xyzw: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_xyzw must be a finite (4,) vector")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("quaternion_xyzw cannot have zero norm")
    x, y, z, w = quaternion / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def heightmap_query_points_world(
    root_xy_m: np.ndarray,
    root_yaw_rad: float,
    cfg: ObservationV0Cfg | None = None,
) -> np.ndarray:
    """Return the ``(size, size, 2)`` Task-v0 world-space query grid."""

    cfg = cfg or ObservationV0Cfg()
    root_xy = np.asarray(root_xy_m, dtype=np.float64)
    if root_xy.shape != (2,) or not np.all(np.isfinite(root_xy)):
        raise ValueError("root_xy_m must be a finite (2,) vector")
    yaw = float(root_yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError("root_yaw_rad must be finite")

    indices = np.arange(cfg.heightmap_size, dtype=np.float64)
    center = cfg.heightmap_size // 2
    x_body = (center - indices)[:, None] * cfg.heightmap_resolution_m
    y_body = (center - indices)[None, :] * cfg.heightmap_resolution_m
    x_body = np.broadcast_to(x_body, (cfg.heightmap_size, cfg.heightmap_size))
    y_body = np.broadcast_to(y_body, (cfg.heightmap_size, cfg.heightmap_size))

    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    world_x = root_xy[0] + cosine * x_body - sine * y_body
    world_y = root_xy[1] + sine * x_body + cosine * y_body
    return np.stack((world_x, world_y), axis=-1)


def build_observation_v0(
    *,
    root_position_m: np.ndarray,
    root_orientation_xyzw: np.ndarray,
    goal_xy_m: np.ndarray,
    goal_yaw_rad: float,
    previous_applied_action: np.ndarray,
    terrain_height_query: Callable[[np.ndarray], np.ndarray],
    cfg: ObservationV0Cfg | None = None,
) -> np.ndarray:
    """Build the frozen 10,206-element Observation-v0 vector."""

    cfg = cfg or ObservationV0Cfg()
    root_position = np.asarray(root_position_m, dtype=np.float64)
    action = np.asarray(previous_applied_action, dtype=np.float64)
    if root_position.shape != (3,) or not np.all(np.isfinite(root_position)):
        raise ValueError("root_position_m must be a finite (3,) vector")
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise ValueError("previous_applied_action must be a finite (2,) vector")

    yaw = quaternion_yaw_xyzw(root_orientation_xyzw)
    geometry = goal_geometry(root_position[:2], yaw, goal_xy_m, goal_yaw_rad)
    query_points = heightmap_query_points_world(root_position[:2], yaw, cfg)
    heights = np.asarray(terrain_height_query(query_points.reshape(-1, 2)), dtype=np.float64)
    if heights.shape != (cfg.heightmap_size * cfg.heightmap_size,):
        raise ValueError("terrain_height_query must return one height per query point")
    heights = heights.reshape(cfg.heightmap_size, cfg.heightmap_size)
    center = cfg.heightmap_size // 2
    center_height = float(heights[center, center])
    if not np.isfinite(center_height):
        raise ValueError("height map center query must be finite")

    normalized_heights = np.full(heights.shape, cfg.invalid_height_value, dtype=np.float64)
    valid = np.isfinite(heights)
    normalized_heights[valid] = np.clip(
        (heights[valid] - center_height) / cfg.height_normalization_m,
        -1.0,
        1.0,
    )
    observation = np.concatenate(
        (
            np.clip(action, -1.0, 1.0),
            np.asarray(
                [
                    np.clip(geometry.distance_m / cfg.goal_distance_normalization_m, 0.0, 1.0),
                    geometry.bearing_body_rad / np.pi,
                    geometry.heading_error_rad / np.pi,
                ]
            ),
            normalized_heights.ravel(order="C"),
        )
    ).astype(np.float32)
    if observation.shape != (cfg.observation_size,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("Observation-v0 construction produced an invalid vector")
    return np.ascontiguousarray(observation)
