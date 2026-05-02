"""Equirectangular (360 panorama) camera model.

Maps a 3D point X = (x, y, z) in camera frame to (u, v) on an equirectangular
image of resolution (W, H):

    phi   = atan2(x, z)                in [-pi, pi]
    theta = asin(y / r),  r = ||X||    in [-pi/2, pi/2]
    u     = (phi   / (2 pi)) * W + W/2
    v     = (-theta / pi)    * H + H/2

So the image x-axis is longitude (left -> -pi, right -> +pi) and the y-axis
is latitude (top -> +pi/2, bottom -> -pi/2). Conventions match Matterport3D
and the OmniGS / 360-GS literature.

The 2x3 Jacobian J = d (u, v) / d (x, y, z) has a closed form derived in
docs/architecture.md. We implement and validate it against autograd in tests.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from sat_splat.cameras.base import CameraModel


class Equirectangular(CameraModel):
    name = "equirect"

    def __init__(self, width: int, height: int):
        self.W = int(width)
        self.H = int(height)

    # -- forward -----------------------------------------------------------------

    def project(self, X: Tensor) -> Tensor:
        x, y, z = X.unbind(-1)
        phi = torch.atan2(x, z)
        r = torch.linalg.norm(X, dim=-1).clamp_min(1e-8)
        theta = torch.asin((y / r).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        u = (phi / (2.0 * math.pi)) * self.W + 0.5 * self.W
        v = (-theta / math.pi) * self.H + 0.5 * self.H
        return torch.stack([u, v], dim=-1)

    # -- analytic Jacobian -------------------------------------------------------

    def jacobian(self, X: Tensor) -> Tensor:
        """Closed-form Jacobian of equirectangular projection.

        Let r = ||X||, rho = sqrt(x^2 + z^2). Then

            d phi / d x  =   z / rho^2
            d phi / d y  =   0
            d phi / d z  = - x / rho^2

            d theta / d x = - x y / (r^2 * rho)
            d theta / d y =   rho / r^2
            d theta / d z = - z y / (r^2 * rho)

        Multiplying by the (W, H) scale factors and signs from project() gives
        the 2x3 Jacobian.
        """
        x, y, z = X.unbind(-1)
        r2 = x * x + y * y + z * z
        rho2 = x * x + z * z
        rho = rho2.clamp_min(1e-12).sqrt()
        r2_safe = r2.clamp_min(1e-12)

        # phi gradients
        dphi_dx = z / rho2.clamp_min(1e-12)
        dphi_dy = torch.zeros_like(x)
        dphi_dz = -x / rho2.clamp_min(1e-12)

        # theta gradients
        dth_dx = -(x * y) / (r2_safe * rho)
        dth_dy = rho / r2_safe
        dth_dz = -(z * y) / (r2_safe * rho)

        u_scale = self.W / (2.0 * math.pi)
        v_scale = -self.H / math.pi

        du = torch.stack([u_scale * dphi_dx, u_scale * dphi_dy, u_scale * dphi_dz], dim=-1)
        dv = torch.stack([v_scale * dth_dx, v_scale * dth_dy, v_scale * dth_dz], dim=-1)
        return torch.stack([du, dv], dim=-2)  # (..., 2, 3)
