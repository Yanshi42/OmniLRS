"""Terrain-deformation probe used only by the P1 B3 cost diagnostic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class TerrainDeformationCostProbe:
    """Apply one configured wheel-footprint deformation per policy step.

    The probe intentionally lives outside the Gym environment and does not
    alter Observation-v0. It uses adapter-configured ground-contact links, so
    the diagnostic does not assume a four-wheel vehicle.
    """

    def __init__(
        self,
        *,
        terrain_manager: Any,
        terrain_provider: Any,
        link_provider: Any,
        deformation_links: Sequence[str],
    ) -> None:
        links = tuple(str(link) for link in deformation_links)
        if not links or len(set(links)) != len(links):
            raise ValueError("deformation_links must be non-empty and unique")
        self.terrain_manager = terrain_manager
        self.terrain_provider = terrain_provider
        self.link_provider = link_provider
        self.deformation_links = links
        self._measuring = False
        self._measurement_initial_dem: np.ndarray | None = None
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.attempt_count = 0
        self.update_count = 0
        self.error_count = 0
        self.maximum_abs_normal_force_n = 0.0
        self.last_error: str | None = None

    def begin_measurement(self) -> None:
        self._reset_metrics()
        self._measurement_initial_dem = np.asarray(self.terrain_manager.getDEM(), dtype=np.float64).copy()
        self._measuring = True

    def __call__(self) -> None:
        if self._measuring:
            self.attempt_count += 1
        try:
            positions, orientations_wxyz, forces = self.link_provider.sample_link_poses_and_forces(
                self.deformation_links
            )
            expected = len(self.deformation_links)
            if positions.shape != (expected, 3) or orientations_wxyz.shape != (expected, 4):
                raise RuntimeError("deformation link pose shapes do not match configured link count")
            if forces.shape != (expected, 3):
                raise RuntimeError("deformation contact-force shape does not match configured link count")
            self.terrain_manager.deformTerrain(positions, orientations_wxyz, forces)
            self.terrain_provider.update_dem(self.terrain_manager.getDEM())
        except Exception as exc:
            if self._measuring:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            return
        if self._measuring:
            self.update_count += 1
            self.maximum_abs_normal_force_n = max(
                self.maximum_abs_normal_force_n,
                float(np.max(np.abs(forces[:, 2]))),
            )

    def summary(self) -> dict[str, Any]:
        current_dem = np.asarray(self.terrain_manager.getDEM(), dtype=np.float64)
        if self._measurement_initial_dem is None or current_dem.shape != self._measurement_initial_dem.shape:
            changed_samples = 0
            maximum_abs_change_m = 0.0
            rms_change_m = 0.0
        else:
            delta = current_dem - self._measurement_initial_dem
            changed_samples = int(np.count_nonzero(delta))
            maximum_abs_change_m = float(np.max(np.abs(delta)))
            rms_change_m = float(np.sqrt(np.mean(np.square(delta))))
        return {
            "deformation_links": list(self.deformation_links),
            "link_count": len(self.deformation_links),
            "cadence": "once_per_policy_step_after_physics",
            "orientation_order": "wxyz",
            "attempt_count": self.attempt_count,
            "update_count": self.update_count,
            "error_count": self.error_count,
            "changed_dem_sample_count": changed_samples,
            "maximum_abs_dem_change_m": maximum_abs_change_m,
            "rms_dem_change_m": rms_change_m,
            "maximum_abs_normal_force_n": self.maximum_abs_normal_force_n,
            "last_error": self.last_error,
        }
