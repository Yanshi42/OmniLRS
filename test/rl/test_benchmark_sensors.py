import numpy as np

from src.environments_wrappers.rl.benchmark_sensors import D455RgbDepthCostProbe


class _Camera:
    def __init__(self, rgba, depth):
        self.rgba = rgba
        self.depth = depth
        self.initialized = False
        self.depth_enabled = False

    def initialize(self):
        self.initialized = True

    def add_distance_to_image_plane_to_frame(self):
        self.depth_enabled = True

    def get_rgba(self):
        return self.rgba

    def get_depth(self):
        return self.depth


def test_d455_probe_ignores_warmup_and_accepts_rgb_depth_frame():
    camera = _Camera(
        np.ones((240, 320, 4), dtype=np.uint8),
        np.ones((240, 320), dtype=np.float32),
    )
    probe = D455RgbDepthCostProbe("/camera", camera=camera)
    assert camera.initialized and camera.depth_enabled
    probe()
    probe.begin_measurement()
    probe()

    summary = probe.summary()
    assert summary["capture_count"] == 1
    assert summary["valid_capture_count"] == 1
    assert summary["invalid_capture_count"] == 0
    assert summary["last_rgb_shape"] == [240, 320, 4]
    assert summary["last_depth_shape"] == [240, 320]


def test_d455_probe_records_invalid_frames_without_raising():
    camera = _Camera(
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.full((240, 320), np.inf, dtype=np.float32),
    )
    probe = D455RgbDepthCostProbe("/camera", camera=camera)
    probe.begin_measurement()
    probe()

    summary = probe.summary()
    assert summary["capture_count"] == 1
    assert summary["invalid_capture_count"] == 1
    assert summary["rgb_shape_error_count"] == 1
    assert summary["depth_no_finite_value_count"] == 1
    assert summary["rgb_all_black_count"] == 1
