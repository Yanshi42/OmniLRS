import numpy as np

from src.environments_wrappers.rl.benchmark_deformation import TerrainDeformationCostProbe


class _TerrainManager:
    def __init__(self):
        self.dem = np.zeros((5, 5), dtype=np.float32)

    def getDEM(self):
        return self.dem

    def deformTerrain(self, positions, orientations, forces):
        assert positions.shape == (2, 3)
        assert orientations.shape == (2, 4)
        assert forces.shape == (2, 3)
        self.dem[1, 1] -= 0.001


class _TerrainProvider:
    def __init__(self):
        self.updates = 0

    def update_dem(self, dem):
        assert dem.shape == (5, 5)
        self.updates += 1


class _LinkProvider:
    def sample_link_poses_and_forces(self, links):
        assert tuple(links) == ("left", "right")
        return (
            np.zeros((2, 3)),
            np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0]]),
        )


def test_deformation_probe_ignores_warmup_metrics_and_tracks_dem_change():
    manager = _TerrainManager()
    provider = _TerrainProvider()
    probe = TerrainDeformationCostProbe(
        terrain_manager=manager,
        terrain_provider=provider,
        link_provider=_LinkProvider(),
        deformation_links=("left", "right"),
    )

    probe()
    probe.begin_measurement()
    probe()
    result = probe.summary()

    assert provider.updates == 2
    assert result["attempt_count"] == 1
    assert result["update_count"] == 1
    assert result["error_count"] == 0
    assert result["changed_dem_sample_count"] == 1
    assert result["maximum_abs_dem_change_m"] > 0.0
    assert result["maximum_abs_normal_force_n"] == 6.0
