"""Reference differentiable Gaussian rasterizer in pure PyTorch.

This is the CPU/GPU-portable fallback for the optimized CUDA fork
(``sat_splat._cuda_rasterizer``). It implements distortion-aware EWA splatting:
the 3D covariance of each Gaussian is projected to 2D through the *per-camera
analytic Jacobian* (so the splat is correct for fisheye / pushbroom / RPC /
equirectangular, not just the pinhole affine approximation), an optional learned
distortion perturbation is added to the 2D covariance, and the Gaussians are
alpha-composited front-to-back over a full pixel grid.

It is intentionally un-tiled and O(N x H x W): correct and fully differentiable,
not fast. Use it to render, train, and cross-validate on CPU; build the CUDA
fork for speed. The function signature matches what ``SatSplatPipeline.render``
passes to the rasterizer, so it is a drop-in backend.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def quat_to_rotmat(quats: Tensor) -> Tensor:
    """Unit-normalize (w, x, y, z) quaternions and return rotation matrices (N, 3, 3)."""
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    n = quats.shape[0]
    R = torch.empty(n, 3, 3, device=quats.device, dtype=quats.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def covariance_3d(scales_log: Tensor, quats: Tensor) -> Tensor:
    """Sigma = R S S^T R^T from log-scales and quaternions. -> (N, 3, 3)."""
    s = torch.exp(scales_log)                 # (N, 3)
    rs = quat_to_rotmat(quats) * s.unsqueeze(-2)   # scale each column j of R by s_j
    return rs @ rs.transpose(-1, -2)


def distortion_aware_rasterize(
    *,
    means: Tensor,                 # (N, 3) Gaussian centers (camera frame, z forward)
    scales: Tensor,                # (N, 3) log-scales
    quats: Tensor,                 # (N, 4) quaternions
    opacities: Tensor,             # (N, 1) opacity logits
    colors: Tensor,                # (N, 3, K) SH coeffs (DC term used) or (N, 3) RGB logits
    uv: Tensor,                    # (N, 2) projected pixel coordinates
    jacobian: Tensor,              # (N, 2, 3) analytic projection Jacobian
    distortion_perturbation: Tensor | None,  # (N, 2, 2) or None
    image_hw: tuple[int, int],
    blur: float = 0.3,             # dilation added to the 2D covariance diagonal
    background: float = 0.0,
) -> Tensor:
    """Render the Gaussian field to an ``(H, W, 3)`` image. Fully differentiable."""
    height, width = image_hw
    device, dtype = means.device, means.dtype
    n = means.shape[0]

    # 3D -> 2D covariance through the per-camera Jacobian (EWA splatting).
    cov3 = covariance_3d(scales, quats)                       # (N, 3, 3)
    cov2 = jacobian @ cov3 @ jacobian.transpose(-1, -2)       # (N, 2, 2)
    if distortion_perturbation is not None:
        cov2 = cov2 + distortion_perturbation
    cov2 = cov2 + blur * torch.eye(2, device=device, dtype=dtype)

    # 2x2 inverse (conic) in closed form.
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-9)
    conic = torch.stack([c, -b, -b, a], dim=-1).reshape(n, 2, 2) / det.view(n, 1, 1)

    rgb = colors[..., 0] if colors.dim() == 3 else colors     # (N, 3) DC term / RGB logits
    rgb = torch.sigmoid(rgb)
    opacity = torch.sigmoid(opacities).reshape(n)

    # Pixel grid as (P, 2) = (x, y).
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)   # (P, 2)
    p = pix.shape[0]

    # Front-to-back composite (nearest first by camera-frame depth).
    order = torch.argsort(means[:, 2]).tolist()
    img = torch.zeros(p, 3, device=device, dtype=dtype)
    transmittance = torch.ones(p, device=device, dtype=dtype)
    for i in order:
        d = pix - uv[i]                                  # (P, 2)
        power = -0.5 * ((d @ conic[i]) * d).sum(-1)      # (P,)
        alpha = (opacity[i] * torch.exp(power.clamp(max=0.0))).clamp(0.0, 0.99)
        weight = transmittance * alpha                   # (P,)
        img = img + weight.unsqueeze(-1) * rgb[i]
        transmittance = transmittance * (1.0 - alpha)
    img = img + transmittance.unsqueeze(-1) * background
    return img.reshape(height, width, 3)


__all__ = ["distortion_aware_rasterize", "covariance_3d", "quat_to_rotmat"]
