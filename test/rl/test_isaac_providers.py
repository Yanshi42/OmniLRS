from __future__ import annotations

import numpy as np
import pytest

from src.environments_wrappers.rl.isaac_providers import DemTerrainProvider


def test_dem_query_is_bilinear_and_marks_out_of_bounds_invalid():
    dem = np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    provider = DemTerrainProvider(dem, resolution_m=1.0)

    heights = provider.height(np.asarray([[0.5, 0.5], [2.0, 2.0], [-0.1, 0.0], [0.0, 2.1]]))

    assert heights[:2] == pytest.approx([5.5, 22.0])
    assert np.isnan(heights[2:]).all()


def test_footprint_finite_checks_rotated_boundary_samples():
    provider = DemTerrainProvider(np.zeros((11, 11)), resolution_m=0.1)

    assert provider.footprint_is_finite(np.asarray([0.5, 0.5]), np.pi / 4.0, length_m=0.4, width_m=0.2)
    assert not provider.footprint_is_finite(np.asarray([0.05, 0.05]), 0.0, length_m=0.4, width_m=0.2)


def test_dem_can_be_replaced_after_deformation_without_changing_shape():
    provider = DemTerrainProvider(np.zeros((3, 3)), resolution_m=1.0)
    provider.update_dem(np.full((3, 3), 2.0))

    assert provider.height(np.asarray([[0.5, 0.5]])).tolist() == pytest.approx([2.0])
    with pytest.raises(ValueError, match="does not match"):
        provider.update_dem(np.zeros((4, 4)))
