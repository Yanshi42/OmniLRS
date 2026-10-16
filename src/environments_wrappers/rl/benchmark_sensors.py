"""Direct Isaac sensor probes used only by P1 cost diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


class D455RgbDepthCostProbe:
    """Read Husky's existing D455 color prim as RGB and image-plane depth.

    Isaac imports stay inside ``__init__`` so pure RL unit tests do not require
    SimulationApp. The probe deliberately does not alter Observation-v0.
    """

    def __init__(
        self,
        camera_prim_path: str,
        *,
        resolution: tuple[int, int] = (320, 240),
        camera: Any | None = None,
    ) -> None:
        if len(resolution) != 2 or min(resolution) <= 0:
            raise ValueError("camera resolution must contain two positive values")
        self.camera_prim_path = str(camera_prim_path)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        if camera is None:
            from isaacsim.sensors.camera import Camera

            camera = Camera(self.camera_prim_path, resolution=self.resolution)
        self.camera = camera
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        self._measuring = False
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.capture_count = 0
        self.valid_capture_count = 0
        self.invalid_capture_count = 0
        self.rgb_shape_error_count = 0
        self.depth_shape_error_count = 0
        self.rgb_nonfinite_count = 0
        self.depth_no_finite_value_count = 0
        self.rgb_all_black_count = 0
        self.capture_exception_count = 0
        self.minimum_depth_finite_fraction = 1.0
        self.minimum_rgb_nonzero_fraction = 1.0
        self.last_error: str | None = None
        self.last_rgb_shape: tuple[int, ...] | None = None
        self.last_depth_shape: tuple[int, ...] | None = None

    def begin_measurement(self) -> None:
        self._reset_metrics()
        self._measuring = True

    def __call__(self) -> None:
        try:
            rgba = np.asarray(self.camera.get_rgba())
            depth = np.asarray(self.camera.get_depth())
        except Exception as exc:
            if self._measuring:
                self.capture_count += 1
                self.invalid_capture_count += 1
                self.capture_exception_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            return
        if not self._measuring:
            return

        self.capture_count += 1
        self.last_rgb_shape = tuple(int(value) for value in rgba.shape)
        self.last_depth_shape = tuple(int(value) for value in depth.shape)
        height, width = self.resolution[1], self.resolution[0]
        rgb_shape_ok = rgba.shape in ((height, width, 3), (height, width, 4))
        depth_shape_ok = depth.shape == (height, width)
        rgb_finite = bool(rgba.size and np.all(np.isfinite(rgba)))
        depth_finite_fraction = float(np.count_nonzero(np.isfinite(depth)) / depth.size) if depth.size else 0.0
        rgb_values = rgba[..., :3] if rgba.ndim == 3 and rgba.shape[-1] >= 3 else rgba
        rgb_nonzero_fraction = float(np.count_nonzero(rgb_values) / rgb_values.size) if rgb_values.size else 0.0

        self.rgb_shape_error_count += int(not rgb_shape_ok)
        self.depth_shape_error_count += int(not depth_shape_ok)
        self.rgb_nonfinite_count += int(not rgb_finite)
        self.depth_no_finite_value_count += int(depth_finite_fraction <= 0.0)
        self.rgb_all_black_count += int(rgb_nonzero_fraction <= 0.0)
        self.minimum_depth_finite_fraction = min(self.minimum_depth_finite_fraction, depth_finite_fraction)
        self.minimum_rgb_nonzero_fraction = min(self.minimum_rgb_nonzero_fraction, rgb_nonzero_fraction)
        valid = bool(
            rgb_shape_ok
            and depth_shape_ok
            and rgb_finite
            and depth_finite_fraction > 0.0
            and rgb_nonzero_fraction > 0.0
        )
        self.valid_capture_count += int(valid)
        self.invalid_capture_count += int(not valid)

    def summary(self) -> dict[str, Any]:
        return {
            "camera_prim_path": self.camera_prim_path,
            "resolution": list(self.resolution),
            "capture_count": self.capture_count,
            "valid_capture_count": self.valid_capture_count,
            "invalid_capture_count": self.invalid_capture_count,
            "capture_exception_count": self.capture_exception_count,
            "rgb_shape_error_count": self.rgb_shape_error_count,
            "depth_shape_error_count": self.depth_shape_error_count,
            "rgb_nonfinite_count": self.rgb_nonfinite_count,
            "depth_no_finite_value_count": self.depth_no_finite_value_count,
            "rgb_all_black_count": self.rgb_all_black_count,
            "minimum_depth_finite_fraction": self.minimum_depth_finite_fraction,
            "minimum_rgb_nonzero_fraction": self.minimum_rgb_nonzero_fraction,
            "last_rgb_shape": list(self.last_rgb_shape) if self.last_rgb_shape is not None else None,
            "last_depth_shape": list(self.last_depth_shape) if self.last_depth_shape is not None else None,
            "last_error": self.last_error,
        }
