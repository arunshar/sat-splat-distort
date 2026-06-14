"""Tests for the reference PyTorch rasterizer and the synthetic fit.

CPU-only, no CUDA build, no downloads. Verifies the rasterizer renders the
documented shape, is differentiable, that the distortion perturbation actually
changes the render, and that a synthetic fit reduces image error (PSNR rises).
"""
from __future__ import annotations

import torch

from sat_splat.data import default_fisheye, make_scene
from sat_splat.models.splat_pipeline import SatSplatPipeline
from sat_splat.models.torch_rasterizer import (
    covariance_3d,
    distortion_aware_rasterize,
    quat_to_rotmat,
)
from sat_splat.training.fit import fit_synthetic


def test_quat_to_rotmat_is_a_rotation():
    quats = torch.randn(8, 4)
    R = quat_to_rotmat(quats)
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8), atol=1e-5)


def test_covariance_3d_is_symmetric_psd():
    cov = covariance_3d(torch.randn(6, 3) * 0.1, torch.randn(6, 4))
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > -1e-6).all()


def test_render_shape_and_range():
    image_hw = (24, 24)
    scene = make_scene(20, seed=0)
    pipe = SatSplatPipeline(scene=scene, camera=default_fisheye(image_hw), image_hw=image_hw)
    img = pipe.render(backend="torch")
    assert img.shape == (24, 24, 3)
    assert torch.isfinite(img).all()
    assert (img >= -1e-4).all() and (img <= 1.0 + 1e-4).all()


def test_render_is_differentiable():
    image_hw = (20, 20)
    scene = make_scene(16, seed=1)
    pipe = SatSplatPipeline(scene=scene, camera=default_fisheye(image_hw), image_hw=image_hw)
    img = pipe.render(backend="torch")
    img.mean().backward()
    assert pipe.means.grad is not None and torch.isfinite(pipe.means.grad).all()
    assert pipe.colors.grad is not None
    assert pipe.opacities.grad is not None


def test_distortion_perturbation_changes_render():
    image_hw = (20, 20)
    scene = make_scene(16, seed=2)
    camera = default_fisheye(image_hw)
    kw = dict(
        means=scene.means, scales=scene.scales, quats=scene.quats,
        opacities=scene.opacities, colors=scene.colors,
        uv=camera.project(scene.means), jacobian=camera.jacobian(scene.means),
        image_hw=image_hw,
    )
    base = distortion_aware_rasterize(distortion_perturbation=None, **kw)
    n = scene.means.shape[0]
    perturb = 5.0 * torch.eye(2).expand(n, 2, 2)        # large, obvious covariance bump
    bumped = distortion_aware_rasterize(distortion_perturbation=perturb, **kw)
    assert not torch.allclose(base, bumped, atol=1e-3)


def test_fit_reduces_image_error():
    out = fit_synthetic(steps=120, seed=0)
    assert out["loss_end"] < out["loss_start"]
    assert out["psnr_end"] > out["psnr_start"] + 1.0   # real reconstruction gain
