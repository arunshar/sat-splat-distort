"""Equidistant fisheye camera model.

Many fisheye lenses follow the equidistant projection model: the radial
distance r_2d in the image plane is proportional to the angle theta from
the optical axis, i.e. r_2d = f * theta. (Pinhole would be r_2d = f * tan(theta).)

For X = (x, y, z) with z > 0 in front of the camera:

    r_3d = ||X||
    theta = acos(z / r_3d)               (angle from +z axis)
    rho   = sqrt(x^2 + y^2)
    s     = (f * theta) / rho            (radial scale; rho>0 needed)
    u     = s * x + cx
    v     = s * y + cy

The Jacobian is derived in docs/architecture.md. The key analytic identities
are:

    d theta / d x =   x z / (r_3d^2 * rho)
    d theta / d y =   y z / (r_3d^2 * rho)
    d theta / d z = - rho / r_3d^2

The Jacobian of (u, v) is then assembled via product rule on s = f*theta/rho.
"""
from __future__ import annotations

import torch
from torch import Tensor

from sat_splat.cameras.base import CameraModel


class EquidistantFisheye(CameraModel):
    name = "equidist"

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    # -- forward -----------------------------------------------------------------

    def project(self, X: Tensor) -> Tensor:
        x, y, z = X.unbind(-1)
        r_3d = torch.linalg.norm(X, dim=-1).clamp_min(1e-8)
        rho = torch.sqrt((x * x + y * y).clamp_min(1e-12))
        cos_t = (z / r_3d).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_t)
        s_x = (self.fx * theta) / rho
        s_y = (self.fy * theta) / rho
        u = s_x * x + self.cx
        v = s_y * y + self.cy
        return torch.stack([u, v], dim=-1)

    # -- analytic Jacobian -------------------------------------------------------

    def jacobian(self, X: Tensor) -> Tensor:
        x, y, z = X.unbind(-1)
        r2 = (x * x + y * y + z * z).clamp_min(1e-12)
        r = r2.sqrt()
        rho2 = (x * x + y * y).clamp_min(1e-12)
        rho = rho2.sqrt()

        # Angle from optical axis and its gradient w.r.t. X.
        cos_t = (z / r).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_t)

        dtheta_dx = (x * z) / (r2 * rho)
        dtheta_dy = (y * z) / (r2 * rho)
        dtheta_dz = -rho / r2

        # Radial scale s = (f * theta) / rho. Derivative of s_axis (axis in {fx, fy}):
        #   d/dq [theta / rho] = (d theta / d q) / rho - theta * (d rho / d q) / rho^2
        # where d rho / d (x, y, z) = (x/rho, y/rho, 0). Substituting d rho / d x = x/rho:
        #   d (theta/rho) / d x = (d theta/dx) / rho - theta * x / rho^3
        # so the residual term carries an inv_rho^3 = inv_rho * inv_rho2.
        inv_rho = 1.0 / rho
        inv_rho3 = inv_rho / rho2  # = 1 / rho^3

        d_tor_dx = dtheta_dx * inv_rho - theta * x * inv_rho3
        d_tor_dy = dtheta_dy * inv_rho - theta * y * inv_rho3
        d_tor_dz = dtheta_dz * inv_rho  # d rho / d z = 0

        # u = fx * (theta / rho) * x + cx
        # du/dx = fx * (d (theta/rho)/dx * x + (theta/rho))
        # du/dy = fx * x * d (theta/rho)/dy
        # du/dz = fx * x * d (theta/rho)/dz
        tor = theta * inv_rho
        du_dx = self.fx * (d_tor_dx * x + tor)
        du_dy = self.fx * (d_tor_dy * x)
        du_dz = self.fx * (d_tor_dz * x)

        dv_dx = self.fy * (d_tor_dx * y)
        dv_dy = self.fy * (d_tor_dy * y + tor)
        dv_dz = self.fy * (d_tor_dz * y)

        du = torch.stack([du_dx, du_dy, du_dz], dim=-1)
        dv = torch.stack([dv_dx, dv_dy, dv_dz], dim=-1)
        return torch.stack([du, dv], dim=-2)
