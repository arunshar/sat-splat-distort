"""Synthetic data for the reference rasterizer.

STATUS: SYNTHETIC ONLY. The real loaders (DFC2019 satellite stereo, Matterport
360 panoramas) are NOT implemented; this module fabricates a small Gaussian
field and renders it through the portable PyTorch rasterizer so the training and
eval paths run end to end on CPU without external data or a CUDA build. Treat the
rendered images as synthetic targets, not real imagery.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from sat_splat.cameras.equidist import EquidistantFisheye
from sat_splat.models.splat_pipeline import SatSplatPipeline, SplatScene


def default_fisheye(image_hw: tuple[int, int]) -> EquidistantFisheye:
    """An equidistant-fisheye camera centered on the image, focal ~ 0.35 * width."""
    h, w = image_hw
    f = 0.35 * w
    return EquidistantFisheye(fx=f, fy=f, cx=w / 2.0, cy=h / 2.0)


def make_scene(
    num_points: int = 40,
    *,
    seed: int = 0,
    sh_dim: int = 16,
    z_range: tuple[float, float] = (3.0, 4.0),
    xy_spread: float = 1.2,
    log_scale: float = -2.6,          # exp(-2.6) ~ 0.074 world units
    opacity_logit: float = 1.5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SplatScene:
    """A random Gaussian field placed in front of the camera (camera frame, z forward)."""
    g = torch.Generator().manual_seed(int(seed))
    z = z_range[0] + (z_range[1] - z_range[0]) * torch.rand(num_points, 1, generator=g)
    xy = (2.0 * torch.rand(num_points, 2, generator=g) - 1.0) * xy_spread
    means = torch.cat([xy, z], dim=-1).to(device=device, dtype=dtype)
    scales = torch.full((num_points, 3), log_scale, device=device, dtype=dtype)
    scales = scales + 0.1 * torch.randn(num_points, 3, generator=g).to(device, dtype)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_points, 4).clone().to(device, dtype)
    quats = quats + 0.05 * torch.randn(num_points, 4, generator=g).to(device, dtype)
    opacities = torch.full((num_points, 1), opacity_logit, device=device, dtype=dtype)
    colors = torch.zeros(num_points, 3, sh_dim, device=device, dtype=dtype)
    # vivid, varied DC colors so the rendered target is non-trivial
    colors[:, :, 0] = 3.0 * (2.0 * torch.rand(num_points, 3, generator=g) - 1.0).to(device, dtype)
    return SplatScene(means=means, scales=scales, quats=quats, opacities=opacities, colors=colors)


@torch.no_grad()
def render_target(
    scene: SplatScene,
    image_hw: tuple[int, int],
    camera: EquidistantFisheye | None = None,
    with_distortion: bool = False,
) -> Tensor:
    """Render a scene to an ``(H, W, 3)`` target image with the torch backend."""
    camera = camera or default_fisheye(image_hw)
    pipe = SatSplatPipeline(scene=scene, camera=camera, image_hw=image_hw)
    return pipe.render(with_distortion=with_distortion, backend="torch").clamp(0.0, 1.0)


class SyntheticSplatDataset(Dataset):
    """Yields a single synthetic target image (and its camera) for the train loop."""

    def __init__(self, image_hw: tuple[int, int] = (48, 48), num_points: int = 40, seed: int = 0):
        self.image_hw = image_hw
        self.camera = default_fisheye(image_hw)
        scene = make_scene(num_points, seed=seed)
        self.image = render_target(scene, image_hw, self.camera)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        if idx != 0:
            raise IndexError(idx)
        return {"image": self.image}


__all__ = ["default_fisheye", "make_scene", "render_target", "SyntheticSplatDataset"]
