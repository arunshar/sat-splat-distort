"""Linear pushbroom camera model (single-line scanner approximation).

A pushbroom sensor (Sentinel-2, Landsat, WorldView, SPOT) captures one image
row per scan line as the platform moves. We use the standard linearized model
from Gupta & Hartley (PAMI 1997):

    u = f_u * (X . a) / (X . c)        (cross-track; pinhole within a row)
    v = f_v * (X . b)                   (along-track; linear in time)

where ``a``, ``b``, ``c`` are unit row vectors of the rotation/translation
basis at scan epoch and X is a 3D point in the camera-local frame. f_u and
f_v are the cross-track focal length and along-track scan rate.

This is a degenerate camera in the EWA sense: the v-row direction is *not*
projective, so the standard 3DGS Jacobian is wrong by construction. The
distortion-aware rasterizer needs the analytic 2x3 below.
"""
from __future__ import annotations

import torch
from torch import Tensor

from sat_splat.cameras.base import CameraModel


class Pushbroom(CameraModel):
    name = "pushbroom"

    def __init__(self, fu: float, fv: float, a: tuple, b: tuple, c: tuple):
        self.fu = float(fu)
        self.fv = float(fv)
        self.register("a", a)
        self.register("b", b)
        self.register("c", c)

    def register(self, name: str, vec: tuple) -> None:
        t = torch.tensor(vec, dtype=torch.float32)
        if t.shape != (3,):
            raise ValueError(f"{name} must be a length-3 vector, got {tuple(t.shape)}")
        setattr(self, name, t)

    def to(self, device, dtype=None) -> "Pushbroom":
        for axis in ("a", "b", "c"):
            t = getattr(self, axis).to(device=device)
            if dtype is not None:
                t = t.to(dtype=dtype)
            setattr(self, axis, t)
        return self

    # -- forward -----------------------------------------------------------------

    def project(self, X: Tensor) -> Tensor:
        a = self.a.to(X)
        b = self.b.to(X)
        c = self.c.to(X)
        Xa = (X * a).sum(-1)
        Xb = (X * b).sum(-1)
        Xc = (X * c).sum(-1).clamp_min(1e-8)
        u = self.fu * Xa / Xc
        v = self.fv * Xb
        return torch.stack([u, v], dim=-1)

    # -- analytic Jacobian -------------------------------------------------------

    def jacobian(self, X: Tensor) -> Tensor:
        """Closed form:

            du/dX = fu * (a / (X . c) - (X . a) * c / (X . c)^2)
            dv/dX = fv * b
        """
        a = self.a.to(X)
        b = self.b.to(X)
        c = self.c.to(X)

        Xa = (X * a).sum(-1, keepdim=True)
        Xc = (X * c).sum(-1, keepdim=True).clamp_min(1e-8)

        du = self.fu * (a / Xc - Xa * c / (Xc * Xc))
        dv = self.fv * b.expand_as(du)
        return torch.stack([du, dv], dim=-2)
