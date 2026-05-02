"""Validate analytic Jacobians for every camera model against torch.autograd.

Each test samples random 3D points in a sane operating region for the model,
computes the analytic 2x3 Jacobian, and compares it to the autograd reference
elementwise. The closed-form Jacobian is the heart of the technical
contribution; if any of these tests fail the rasterizer is wrong.
"""
from __future__ import annotations

import math

import pytest
import torch

from sat_splat.cameras import (
    EquidistantFisheye,
    Equirectangular,
    Pushbroom,
    RPCCamera,
)
from sat_splat.cameras.rpc import RPCMeta

TOLERANCE = 1e-4
torch.manual_seed(0)


def _check(camera, X, atol=TOLERANCE, rtol=1e-3):
    analytic = camera.jacobian(X)
    reference = camera.autograd_jacobian(X)
    diff = (analytic - reference).abs()
    rel = diff / reference.abs().clamp_min(1e-6)
    bad = (diff > atol) & (rel > rtol)
    assert not bad.any(), (
        f"max abs diff {diff.max().item():.3e}, max rel diff {rel.max().item():.3e}"
    )


# -------- equirectangular ---------------------------------------------------


def test_equirect_jacobian_matches_autograd():
    cam = Equirectangular(width=2048, height=1024)
    # Sample points on the unit sphere shifted off the optical axis.
    X = torch.randn(64, 3)
    X = X / X.norm(dim=-1, keepdim=True) * (1.0 + torch.rand(64, 1) * 5.0)
    _check(cam, X)


def test_equirect_handles_general_directions():
    cam = Equirectangular(width=4096, height=2048)
    # Avoid the poles where d theta is degenerate; clamp |y| / r < 0.99.
    X = torch.randn(32, 3) * 4.0
    yhat = X[:, 1] / X.norm(dim=-1)
    keep = yhat.abs() < 0.95
    X = X[keep]
    _check(cam, X)


# -------- equidistant fisheye -----------------------------------------------


def test_equidist_jacobian_matches_autograd():
    cam = EquidistantFisheye(fx=400.0, fy=400.0, cx=512.0, cy=512.0)
    # In-front-of-camera points with non-zero rho.
    xy = torch.randn(64, 2) * 2.0
    z = torch.rand(64, 1) * 5.0 + 0.5
    X = torch.cat([xy, z], dim=-1)
    _check(cam, X, atol=1e-3, rtol=1e-3)


def test_equidist_handles_wide_angles():
    cam = EquidistantFisheye(fx=300.0, fy=300.0, cx=256.0, cy=256.0)
    # Points near 80 degrees from optical axis: theta ~ 1.4 rad, well within fisheye FoV.
    theta = torch.full((32,), math.radians(75.0))
    phi = torch.linspace(0, 2 * math.pi, 32)
    rho = torch.sin(theta)
    X = torch.stack([rho * torch.cos(phi), rho * torch.sin(phi), torch.cos(theta)], dim=-1) * 3.0
    _check(cam, X, atol=1e-3, rtol=1e-3)


# -------- pushbroom ---------------------------------------------------------


def test_pushbroom_jacobian_matches_autograd():
    cam = Pushbroom(
        fu=2000.0,
        fv=1.0,
        a=(1.0, 0.0, 0.0),
        b=(0.0, 1.0, 0.0),
        c=(0.0, 0.0, 1.0),
    )
    X = torch.randn(64, 3) * 100.0
    X[:, 2] = X[:, 2].abs() + 200.0  # in front of sensor
    _check(cam, X)


# -------- RPC ----------------------------------------------------------------


def _toy_rpc_meta() -> RPCMeta:
    """A toy but well-conditioned RPC. Numerator / denominator are simple
    polynomials so the toy is easy to differentiate by autograd."""
    line_num = (0.0, 0.0, 0.5, 0.0,) + (0.0,) * 16
    line_den = (1.0,) + (0.0,) * 19
    samp_num = (0.0, 0.5, 0.0, 0.0,) + (0.0,) * 16
    samp_den = (1.0,) + (0.0,) * 19
    return RPCMeta(
        line_offset=2000.0,
        sample_offset=2000.0,
        lat_offset=30.0,
        lon_offset=-90.0,
        height_offset=10.0,
        line_scale=4000.0,
        sample_scale=4000.0,
        lat_scale=0.05,
        lon_scale=0.05,
        height_scale=500.0,
        line_num_coef=line_num,
        line_den_coef=line_den,
        samp_num_coef=samp_num,
        samp_den_coef=samp_den,
    )


def test_rpc_jacobian_matches_autograd_simple():
    cam = RPCCamera(_toy_rpc_meta())
    lon = -90.0 + (torch.rand(32) - 0.5) * 0.04
    lat = 30.0 + (torch.rand(32) - 0.5) * 0.04
    h = 10.0 + (torch.rand(32) - 0.5) * 200.0
    X = torch.stack([lon, lat, h], dim=-1).double()
    cam._cache.clear()  # force recompute in float64
    _check(cam, X, atol=1e-2, rtol=1e-3)


def test_rpc_jacobian_full_cubic_terms():
    """Exercise the cubic monomials by giving the RPC nontrivial numerators."""
    line_num = tuple(0.1 * (i + 1) for i in range(20))
    line_den = (1.0,) + tuple(0.01 * (i + 1) for i in range(19))
    samp_num = tuple(0.05 * (20 - i) for i in range(20))
    samp_den = (1.0,) + tuple(0.005 * (i + 1) for i in range(19))
    meta = RPCMeta(
        line_offset=2000.0,
        sample_offset=2000.0,
        lat_offset=30.0,
        lon_offset=-90.0,
        height_offset=10.0,
        line_scale=4000.0,
        sample_scale=4000.0,
        lat_scale=0.05,
        lon_scale=0.05,
        height_scale=500.0,
        line_num_coef=line_num,
        line_den_coef=line_den,
        samp_num_coef=samp_num,
        samp_den_coef=samp_den,
    )
    cam = RPCCamera(meta)
    lon = -90.0 + (torch.rand(32) - 0.5) * 0.04
    lat = 30.0 + (torch.rand(32) - 0.5) * 0.04
    h = 10.0 + (torch.rand(32) - 0.5) * 200.0
    X = torch.stack([lon, lat, h], dim=-1).double()
    _check(cam, X, atol=5e-2, rtol=5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
