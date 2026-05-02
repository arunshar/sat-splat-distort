"""Rational Polynomial Coefficient (RPC) camera model for satellite imagery.

The RPC model is the de-facto standard format for distributing satellite
camera information for products like WorldView, GeoEye, Pleiades, and the
DFC2019 Track-3 benchmark. It maps a 3D point in geographic coordinates
(longitude, latitude, height) to image coordinates (sample, line) via four
20-term cubic polynomials in normalized geographic inputs:

    P_i(L, P, H) = sum_{m,n,k : m+n+k <= 3} c_{imnk} * L^m * P^n * H^k

with i in {1, 2, 3, 4}. The image coordinates are:

    sample = (P_1 / P_2) * sample_scale + sample_offset
    line   = (P_3 / P_4) * line_scale   + line_offset

where (L, P, H) are normalizations of (lon, lat, height) by per-image offsets
and scales also stored in the RPC metadata. This module implements the
forward pass plus an analytic Jacobian via the chain rule on the polynomial
ratios.

Reference: NGA SDD RPC00B specification.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sat_splat.cameras.base import CameraModel

# Order of the 20 cubic-monomial terms used in the RPC standard. Each entry
# is (m, n, k) with m+n+k <= 3 in (L, P, H) order.
RPC_MONO_ORDER: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (2, 0, 0), (0, 2, 0), (0, 0, 2),
    (1, 1, 1),
    (3, 0, 0), (1, 2, 0), (1, 0, 2),
    (2, 1, 0), (0, 3, 0), (0, 1, 2),
    (2, 0, 1), (0, 2, 1), (0, 0, 3),
)


@dataclass
class RPCMeta:
    """Per-image RPC metadata. All fields are scalars unless noted."""

    line_offset: float
    sample_offset: float
    lat_offset: float
    lon_offset: float
    height_offset: float
    line_scale: float
    sample_scale: float
    lat_scale: float
    lon_scale: float
    height_scale: float
    line_num_coef: tuple   # length 20
    line_den_coef: tuple
    samp_num_coef: tuple
    samp_den_coef: tuple


class RPCCamera(CameraModel):
    """RPC camera with a closed-form analytic Jacobian.

    Inputs to ``project`` are in *world* coordinates (lon, lat, h) in degrees
    and meters. The class internally normalizes to (L, P, H), evaluates the
    polynomials, denormalizes the result to (sample, line) pixels.
    """

    name = "rpc"

    def __init__(self, meta: RPCMeta):
        self.meta = meta
        self._cache: dict[str, Tensor] = {}

    # -- coefficient handling ---------------------------------------------------

    def _coef(self, key: str, device, dtype) -> Tensor:
        if key not in self._cache or self._cache[key].device != device:
            raw = getattr(self.meta, key)
            self._cache[key] = torch.tensor(raw, device=device, dtype=dtype)
        return self._cache[key]

    @staticmethod
    def _monomials(L: Tensor, P: Tensor, H: Tensor) -> Tensor:
        """Return the (..., 20) monomial vector in RPC standard order."""
        L2, P2, H2 = L * L, P * P, H * H
        L3, P3, H3 = L * L2, P * P2, H * H2
        return torch.stack(
            [
                torch.ones_like(L),
                L, P, H,
                L * P, L * H, P * H,
                L2, P2, H2,
                L * P * H,
                L3, L * P2, L * H2,
                L2 * P, P3, P * H2,
                L2 * H, P2 * H, H3,
            ],
            dim=-1,
        )

    @staticmethod
    def _monomial_grads(L: Tensor, P: Tensor, H: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Gradients of the 20 monomials w.r.t. (L, P, H), each (..., 20)."""
        zero = torch.zeros_like(L)
        one = torch.ones_like(L)
        L2, P2, H2 = L * L, P * P, H * H
        # d/dL
        dL = torch.stack(
            [
                zero,
                one, zero, zero,
                P, H, zero,
                2 * L, zero, zero,
                P * H,
                3 * L2, P2, H2,
                2 * L * P, zero, zero,
                2 * L * H, zero, zero,
            ],
            dim=-1,
        )
        # d/dP
        dP = torch.stack(
            [
                zero,
                zero, one, zero,
                L, zero, H,
                zero, 2 * P, zero,
                L * H,
                zero, 2 * L * P, zero,
                L2, 3 * P2, H2,
                zero, 2 * P * H, zero,
            ],
            dim=-1,
        )
        # d/dH
        dH = torch.stack(
            [
                zero,
                zero, zero, one,
                zero, L, P,
                zero, zero, 2 * H,
                L * P,
                zero, zero, 2 * L * H,
                zero, zero, 2 * P * H,
                L2, P2, 3 * H2,
            ],
            dim=-1,
        )
        return dL, dP, dH

    # -- forward ----------------------------------------------------------------

    def _norm(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize world (lon, lat, h) to RPC's internal (L, P, H)."""
        m = self.meta
        lon, lat, h = X.unbind(-1)
        L = (lon - m.lon_offset) / m.lon_scale
        P = (lat - m.lat_offset) / m.lat_scale
        H = (h - m.height_offset) / m.height_scale
        return L, P, H

    def project(self, X: Tensor) -> Tensor:
        m = self.meta
        L, P, H = self._norm(X)
        mono = self._monomials(L, P, H)  # (..., 20)
        ln_num = self._coef("line_num_coef", X.device, X.dtype)
        ln_den = self._coef("line_den_coef", X.device, X.dtype)
        sm_num = self._coef("samp_num_coef", X.device, X.dtype)
        sm_den = self._coef("samp_den_coef", X.device, X.dtype)
        line_norm = (mono * ln_num).sum(-1) / (mono * ln_den).sum(-1).clamp_min(1e-12)
        samp_norm = (mono * sm_num).sum(-1) / (mono * sm_den).sum(-1).clamp_min(1e-12)
        line = line_norm * m.line_scale + m.line_offset
        samp = samp_norm * m.sample_scale + m.sample_offset
        # (samp, line) -> (u, v) pixel order
        return torch.stack([samp, line], dim=-1)

    # -- analytic Jacobian ------------------------------------------------------

    def jacobian(self, X: Tensor) -> Tensor:
        """Analytic Jacobian via quotient rule on the polynomial ratios."""
        m = self.meta
        L, P, H = self._norm(X)
        mono = self._monomials(L, P, H)
        dL, dP, dH = self._monomial_grads(L, P, H)

        ln_num = self._coef("line_num_coef", X.device, X.dtype)
        ln_den = self._coef("line_den_coef", X.device, X.dtype)
        sm_num = self._coef("samp_num_coef", X.device, X.dtype)
        sm_den = self._coef("samp_den_coef", X.device, X.dtype)

        N_line = (mono * ln_num).sum(-1)
        D_line = (mono * ln_den).sum(-1).clamp_min(1e-12)
        N_samp = (mono * sm_num).sum(-1)
        D_samp = (mono * sm_den).sum(-1).clamp_min(1e-12)

        def grad_ratio(Nv, Dv, num_coef, den_coef, dMono):
            dN = (dMono * num_coef).sum(-1)
            dD = (dMono * den_coef).sum(-1)
            return (dN * Dv - Nv * dD) / (Dv * Dv)

        # gradients in normalized (L, P, H)
        d_samp_dL = grad_ratio(N_samp, D_samp, sm_num, sm_den, dL)
        d_samp_dP = grad_ratio(N_samp, D_samp, sm_num, sm_den, dP)
        d_samp_dH = grad_ratio(N_samp, D_samp, sm_num, sm_den, dH)
        d_line_dL = grad_ratio(N_line, D_line, ln_num, ln_den, dL)
        d_line_dP = grad_ratio(N_line, D_line, ln_num, ln_den, dP)
        d_line_dH = grad_ratio(N_line, D_line, ln_num, ln_den, dH)

        # chain through (lon, lat, h) -> (L, P, H) and through pixel scaling
        s_lon = m.sample_scale / m.lon_scale
        s_lat = m.sample_scale / m.lat_scale
        s_h = m.sample_scale / m.height_scale
        l_lon = m.line_scale / m.lon_scale
        l_lat = m.line_scale / m.lat_scale
        l_h = m.line_scale / m.height_scale

        du = torch.stack(
            [s_lon * d_samp_dL, s_lat * d_samp_dP, s_h * d_samp_dH], dim=-1
        )
        dv = torch.stack(
            [l_lon * d_line_dL, l_lat * d_line_dP, l_h * d_line_dH], dim=-1
        )
        return torch.stack([du, dv], dim=-2)
