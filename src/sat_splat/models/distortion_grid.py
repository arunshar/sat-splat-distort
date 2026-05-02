"""Learned distortion-prior token grid.

Captures *residual* projection error that the analytic camera Jacobian does
not model: lens artifacts, RPC fit noise, mounting / scan-rate calibration
errors, and atmospheric refraction near the limb in equirectangular imagery.

We park a 2D grid of latent codes over the image plane (G_h x G_w x C). For
each Gaussian projected to pixel (u, v) we bilinearly sample a token, run it
through a small MLP, and emit a 2x2 covariance perturbation that is added to
the analytic 2x2 covariance from EWA splatting. The grid is learned per-scene
during 3DGS fit.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DistortionPriorGrid(nn.Module):
    """Learned 2D token grid feeding a per-pixel covariance perturbation MLP.

    Args:
        grid_h: token grid height. Default 64.
        grid_w: token grid width. Default 64.
        token_dim: latent dim of each grid cell. Default 16.
        mlp_hidden: width of MLP head. Default 128.
        eps_init: scale of the initial perturbation magnitude (we want this
            small at init so the analytic Jacobian dominates early training).
    """

    def __init__(
        self,
        grid_h: int = 64,
        grid_w: int = 64,
        token_dim: int = 16,
        mlp_hidden: int = 128,
        eps_init: float = 1e-3,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.token_dim = token_dim
        self.eps_init = eps_init

        # Tokens stored as (1, C, H, W) so we can use grid_sample directly.
        self.tokens = nn.Parameter(
            torch.randn(1, token_dim, grid_h, grid_w) * eps_init
        )

        # MLP outputs a 2x2 symmetric perturbation parametrized by (a, b, c)
        # such that Sigma = [[a, b], [b, c]] is symmetric. We squash through
        # softplus on the diagonal to keep it positive semi-definite.
        self.head = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 3),
        )
        # Init last layer near zero so initial perturbation is small.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def sample(self, uv: Tensor, image_hw: tuple[int, int]) -> Tensor:
        """Bilinearly sample tokens at pixel coords uv = (..., 2) in [0, W] x [0, H].

        Returns a (..., token_dim) tensor.
        """
        H, W = image_hw
        # grid_sample expects normalized coords in [-1, 1].
        norm = uv.clone()
        norm[..., 0] = 2.0 * uv[..., 0] / max(W - 1, 1) - 1.0
        norm[..., 1] = 2.0 * uv[..., 1] / max(H - 1, 1) - 1.0

        # Reshape to (1, N, 1, 2) for grid_sample.
        flat = norm.reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            self.tokens, flat, mode="bilinear", padding_mode="border", align_corners=True
        )
        # (1, C, N, 1) -> (..., C)
        out = sampled.squeeze(-1).squeeze(0).permute(1, 0)
        return out.reshape(*uv.shape[:-1], self.token_dim)

    def perturbation(self, uv: Tensor, image_hw: tuple[int, int]) -> Tensor:
        """Return symmetric 2x2 covariance perturbation per pixel."""
        z = self.sample(uv, image_hw)
        params = self.head(z)
        a = F.softplus(params[..., 0]) * 1e-3   # positive scale
        c = F.softplus(params[..., 2]) * 1e-3
        b = params[..., 1] * 1e-3                # off-diagonal can be negative
        sigma = torch.zeros((*uv.shape[:-1], 2, 2), device=uv.device, dtype=uv.dtype)
        sigma[..., 0, 0] = a
        sigma[..., 1, 1] = c
        sigma[..., 0, 1] = b
        sigma[..., 1, 0] = b
        return sigma

    def regularization(self) -> Tensor:
        """L2 on tokens encourages low-frequency distortion fields."""
        return self.tokens.pow(2).mean()
