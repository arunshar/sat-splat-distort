"""High-level pipeline that wraps a forked CUDA rasterizer with our
distortion-aware Jacobian and prior grid.

The pipeline exposes a `from_pretrained` API matching HuggingFace conventions
so users can load a published model card with a single line.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from sat_splat.cameras import from_name as camera_from_name
from sat_splat.cameras.base import CameraModel
from sat_splat.models.distortion_grid import DistortionPriorGrid


@dataclass
class SplatScene:
    """Container for the per-scene state that gets serialized to disk."""

    means: Tensor          # (N, 3)
    scales: Tensor         # (N, 3) log-scales
    quats: Tensor          # (N, 4) unit quaternions
    opacities: Tensor      # (N, 1)
    colors: Tensor         # (N, 3, K) spherical-harmonic coefficients

    def to(self, device, dtype=None) -> "SplatScene":
        return SplatScene(
            **{k: (v.to(device=device, dtype=dtype) if dtype else v.to(device))
               for k, v in self.__dict__.items()}
        )


class SatSplatPipeline(nn.Module):
    """End-to-end distortion-aware splatting pipeline.

    Wraps:
      - a per-scene Gaussian field (means, scales, quats, opacities, colors)
      - a learned DistortionPriorGrid
      - a pluggable camera model

    The CUDA rasterizer fork is loaded lazily so the package imports without
    requiring a built CUDA extension on systems where you only need the
    Python pieces (e.g. for cross-validating Jacobians on CPU).
    """

    def __init__(
        self,
        scene: SplatScene,
        camera: CameraModel,
        image_hw: tuple[int, int],
        distortion_grid: DistortionPriorGrid | None = None,
    ):
        super().__init__()
        self.camera = camera
        self.image_hw = image_hw
        self.means = nn.Parameter(scene.means)
        self.scales = nn.Parameter(scene.scales)
        self.quats = nn.Parameter(scene.quats)
        self.opacities = nn.Parameter(scene.opacities)
        self.colors = nn.Parameter(scene.colors)
        self.distortion_grid = distortion_grid or DistortionPriorGrid()

    # -- forward ----------------------------------------------------------------

    def render(self, *, with_distortion: bool = True) -> Tensor:
        """Render the scene under the configured camera.

        Loads the CUDA rasterizer fork on first call. The Jacobian and
        distortion-grid perturbation are passed in to the rasterizer's
        ``computeCov2D`` replacement so 2D covariances are computed correctly
        for the camera model.
        """
        rasterizer = self._lazy_rasterizer()
        uv = self.camera.project(self.means)
        J = self.camera.jacobian(self.means)
        if with_distortion:
            dSigma = self.distortion_grid.perturbation(uv, self.image_hw)
        else:
            dSigma = None
        return rasterizer(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            colors=self.colors,
            uv=uv,
            jacobian=J,
            distortion_perturbation=dSigma,
            image_hw=self.image_hw,
        )

    def _lazy_rasterizer(self):
        try:
            from sat_splat._cuda_rasterizer import distortion_aware_rasterize  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "CUDA rasterizer fork not built. Run "
                "`pip install -v -e cuda_rasterizer/` from the repo root."
            ) from exc
        return distortion_aware_rasterize

    # -- (de)serialization ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        camera_kind: str,
        image_hw: tuple[int, int],
        camera_kwargs: dict | None = None,
    ) -> "SatSplatPipeline":
        """Load weights from a HuggingFace repo or a local checkpoint dir."""
        from huggingface_hub import snapshot_download  # local import for speed

        path = (
            Path(repo_id_or_path)
            if Path(repo_id_or_path).exists()
            else Path(snapshot_download(repo_id=str(repo_id_or_path)))
        )
        scene_state = torch.load(path / "scene.pt", map_location="cpu", weights_only=True)
        grid_state = torch.load(path / "distortion_grid.pt", map_location="cpu", weights_only=True)
        scene = SplatScene(**scene_state)
        camera = camera_from_name(camera_kind, **(camera_kwargs or {}))
        pipe = cls(scene, camera, image_hw)
        pipe.distortion_grid.load_state_dict(grid_state)
        return pipe

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        scene_state = {
            "means": self.means.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "quats": self.quats.detach().cpu(),
            "opacities": self.opacities.detach().cpu(),
            "colors": self.colors.detach().cpu(),
        }
        torch.save(scene_state, path / "scene.pt")
        torch.save(self.distortion_grid.state_dict(), path / "distortion_grid.pt")

    # -- iteration helpers ------------------------------------------------------

    def gaussians(self) -> Iterable[Tensor]:
        return (self.means, self.scales, self.quats, self.opacities, self.colors)
