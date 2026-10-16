"""Isaac/Lunalab data providers for the Task-v0 runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .robot_adapter import RobotState


class DemTerrainProvider:
    """Vectorized bilinear world-XY query over a regular Lunalab DEM."""

    def __init__(self, dem: np.ndarray, *, resolution_m: float = 0.01) -> None:
        if not np.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive and finite")
        self.dem = self._validated_dem(dem)
        self.resolution_m = float(resolution_m)

    @staticmethod
    def _validated_dem(dem: np.ndarray) -> np.ndarray:
        array = np.asarray(dem, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
            raise ValueError("dem must be a 2-D array with at least 2x2 samples")
        return array

    def update_dem(self, dem: np.ndarray) -> None:
        """Replace the query surface after an in-place terrain update."""

        array = self._validated_dem(dem)
        if array.shape != self.dem.shape:
            raise ValueError(f"updated dem shape {array.shape} does not match {self.dem.shape}")
        self.dem = array

    def height(self, world_xy_m: np.ndarray) -> np.ndarray:
        points = np.asarray(world_xy_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("world_xy_m must have shape (N, 2)")
        result = np.full(points.shape[0], np.nan, dtype=np.float64)
        finite = np.all(np.isfinite(points), axis=1)
        x = points[:, 0] / self.resolution_m
        y = points[:, 1] / self.resolution_m
        # The last DEM row/column are valid at an exact vertex. Clip the upper
        # interpolation index while preserving an interpolation fraction of 1.
        inside = finite & (x >= 0.0) & (y >= 0.0) & (x <= self.dem.shape[1] - 1) & (y <= self.dem.shape[0] - 1)
        if not np.any(inside):
            return result
        xi = x[inside]
        yi = y[inside]
        x0 = np.minimum(np.floor(xi).astype(np.int64), self.dem.shape[1] - 2)
        y0 = np.minimum(np.floor(yi).astype(np.int64), self.dem.shape[0] - 2)
        fx = xi - x0
        fy = yi - y0
        z00 = self.dem[y0, x0]
        z10 = self.dem[y0, x0 + 1]
        z01 = self.dem[y0 + 1, x0]
        z11 = self.dem[y0 + 1, x0 + 1]
        valid_height = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
        interpolated = (
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        )
        selected = np.flatnonzero(inside)
        result[selected[valid_height]] = interpolated[valid_height]
        return result

    def footprint_is_finite(self, center_xy_m: np.ndarray, yaw_rad: float, *, length_m: float, width_m: float) -> bool:
        """Conservative finite-DEM check at center, corners, and edge midpoints."""

        center = np.asarray(center_xy_m, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            return False
        if length_m <= 0.0 or width_m <= 0.0 or not np.isfinite(yaw_rad):
            return False
        half_length = 0.5 * float(length_m)
        half_width = 0.5 * float(width_m)
        local = np.asarray(
            [
                [0.0, 0.0],
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
                [half_length, 0.0],
                [-half_length, 0.0],
                [0.0, half_width],
                [0.0, -half_width],
            ],
            dtype=np.float64,
        )
        cosine = np.cos(float(yaw_rad))
        sine = np.sin(float(yaw_rad))
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
        points = center + local @ rotation.T
        return bool(np.all(np.isfinite(self.height(points))))


class NamedRigidContactProvider:
    """Return per-link net contact force from an Isaac 6.1 named RigidPrim view."""

    def __init__(
        self,
        *,
        robot_prim_path: str,
        contact_links: Sequence[str],
        physics_dt_s: float,
        name: str = "omnilrs_task_contact_view",
    ) -> None:
        if not contact_links or len(set(contact_links)) != len(contact_links):
            raise ValueError("contact_links must be non-empty and unique")
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        # Isaac imports must occur only after SimulationApp starts.
        from isaacsim.core.prims import RigidPrim

        self.contact_links = tuple(str(link) for link in contact_links)
        self.physics_dt_s = float(physics_dt_s)
        # Isaac 6.1 requires the filter-pattern list to match the sensor-pattern
        # list length. A filter-free multi-pattern contact view fails to create,
        # so keep one explicitly named view per link (also used by RobotRigidGroup).
        self.views = []
        for index, link in enumerate(self.contact_links):
            view = RigidPrim(
                prim_paths_expr=f"{robot_prim_path.rstrip('/')}/{link}",
                name=f"{name}_{index}",
                track_contact_forces=True,
                prepare_contact_sensors=True,
            )
            view.initialize()
            if view.count != 1:
                raise RuntimeError(f"contact view for {link!r} matched {view.count} prims instead of one")
            self.views.append(view)
        self._views_by_link = dict(zip(self.contact_links, self.views, strict=True))

    @property
    def is_ready(self) -> bool:
        return bool(self.views) and all(view.is_physics_handle_valid() for view in self.views)

    def __call__(self, _state: RobotState) -> dict[str, np.ndarray]:
        if not self.is_ready:
            raise RuntimeError("Task-v0 named contact view is not ready")
        result: dict[str, np.ndarray] = {}
        for link, view in zip(self.contact_links, self.views, strict=True):
            force = np.asarray(_to_numpy(view.get_net_contact_forces(dt=self.physics_dt_s)), dtype=np.float64)
            force = force.reshape(-1, 3)
            if force.shape != (1, 3) or not np.all(np.isfinite(force)):
                raise RuntimeError(f"invalid contact force for {link!r}: shape={force.shape}")
            result[link] = force[0].copy()
        return result

    def sample_link_poses_and_forces(
        self, links: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return selected link position, Isaac ``wxyz`` pose, and net force."""

        selected = tuple(str(link) for link in links)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("links must be non-empty and unique")
        unknown = set(selected) - set(self.contact_links)
        if unknown:
            raise ValueError(f"links are not monitored: {sorted(unknown)}")
        if not self.is_ready:
            raise RuntimeError("Task-v0 named contact view is not ready")

        positions = np.zeros((len(selected), 3), dtype=np.float64)
        orientations_wxyz = np.zeros((len(selected), 4), dtype=np.float64)
        forces = np.zeros((len(selected), 3), dtype=np.float64)
        for index, link in enumerate(selected):
            view = self._views_by_link[link]
            position, orientation = view.get_world_poses()
            position = np.asarray(_to_numpy(position), dtype=np.float64).reshape(-1, 3)
            orientation = np.asarray(_to_numpy(orientation), dtype=np.float64).reshape(-1, 4)
            force = np.asarray(_to_numpy(view.get_net_contact_forces(dt=self.physics_dt_s)), dtype=np.float64)
            force = force.reshape(-1, 3)
            if position.shape != (1, 3) or orientation.shape != (1, 4) or force.shape != (1, 3):
                raise RuntimeError(
                    f"invalid pose/contact sample for {link!r}: "
                    f"position={position.shape}, orientation={orientation.shape}, force={force.shape}"
                )
            if not (np.all(np.isfinite(position)) and np.all(np.isfinite(orientation)) and np.all(np.isfinite(force))):
                raise RuntimeError(f"non-finite pose/contact sample for {link!r}")
            positions[index] = position[0]
            orientations_wxyz[index] = orientation[0]
            forces[index] = force[0]
        return positions, orientations_wxyz, forces


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
