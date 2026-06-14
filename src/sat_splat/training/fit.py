"""Self-contained synthetic fit: train a Gaussian field to match a target image.

This is the runnable, measurable counterpart to the Hydra ``train.py`` entry
point. It renders a target from a random ground-truth scene through the portable
PyTorch rasterizer, then optimizes a freshly-initialized scene (positions,
scales, rotations, opacities, colors) to reconstruct it, reporting image PSNR
before and after. It needs no external data, no Hydra config, and no CUDA build,
so it doubles as the end-to-end smoke test for the rasterizer.

The numbers are SYNTHETIC (a fit to a rendered target), not a DFC2019 / Matterport
benchmark; they show the reference rasterizer renders, backpropagates, and trains.
"""
from __future__ import annotations

import math

import torch

from sat_splat.data import default_fisheye, make_scene, render_target
from sat_splat.models.splat_pipeline import SatSplatPipeline


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = (pred.clamp(0.0, 1.0) - target).pow(2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def fit_synthetic(
    steps: int = 120,
    *,
    image_hw: tuple[int, int] = (32, 32),
    num_points_gt: int = 30,
    num_points_fit: int = 40,
    lr: float = 2.0e-2,
    seed: int = 0,
    with_distortion: bool = False,
) -> dict:
    """Fit a fresh Gaussian field to a synthetic target; report PSNR start vs end."""
    torch.manual_seed(seed)
    camera = default_fisheye(image_hw)
    target = render_target(make_scene(num_points_gt, seed=seed), image_hw, camera)

    pipe = SatSplatPipeline(
        scene=make_scene(num_points_fit, seed=seed + 1), camera=camera, image_hw=image_hw
    )
    opt = torch.optim.Adam(pipe.parameters(), lr=lr)

    with torch.no_grad():
        psnr_start = _psnr(pipe.render(with_distortion=with_distortion, backend="torch"), target)

    loss_first = loss_last = float("nan")
    for step in range(steps):
        opt.zero_grad()
        pred = pipe.render(with_distortion=with_distortion, backend="torch")
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
        loss_last = float(loss.detach())
        if step == 0:
            loss_first = loss_last

    with torch.no_grad():
        psnr_end = _psnr(pipe.render(with_distortion=with_distortion, backend="torch"), target)

    return {
        "psnr_start": psnr_start,
        "psnr_end": psnr_end,
        "loss_start": loss_first,
        "loss_end": loss_last,
        "steps": steps,
    }


__all__ = ["fit_synthetic"]
