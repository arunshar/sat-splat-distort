"""Camera model interface for distortion-aware Gaussian splatting.

Every camera model exposes:

- ``project(X)``: world-frame 3D point -> 2D image coordinate (pixels).
- ``jacobian(X)``: closed-form 2x3 Jacobian of ``project`` evaluated at X.
  This Jacobian replaces the affine EWA Jacobian in the standard 3DGS
  rasterizer when projecting Gaussian covariances.

Both operations are batched: the leading dim of X is the batch dim, and the
returned tensors are shaped accordingly. All implementations are pure PyTorch
and differentiable, so the analytic Jacobian can be cross-checked against
``torch.autograd`` (see tests/test_cameras.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class CameraModel(ABC):
    """Abstract camera model.

    Subclasses must implement ``project`` and ``jacobian``. Coordinate
    conventions: X is a (..., 3) tensor in the camera's reference frame
    (subclasses define the frame). Output is (..., 2) in pixel units.
    """

    name: str = "base"

    @abstractmethod
    def project(self, X: Tensor) -> Tensor:
        """Project world-frame 3D points to image-plane pixel coordinates."""

    @abstractmethod
    def jacobian(self, X: Tensor) -> Tensor:
        """Return the analytic Jacobian J = d project / d X, shape (..., 2, 3)."""

    def autograd_jacobian(self, X: Tensor) -> Tensor:
        """Reference Jacobian computed by autograd. Use for validation."""
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.detach().clone().requires_grad_(True)
        out = self.project(X)
        # Shape: (B, 2). Build the full Jacobian per batch element via stacking.
        jac_rows = []
        for j in range(out.shape[-1]):
            grad = torch.autograd.grad(
                out[..., j].sum(), X, create_graph=False, retain_graph=True
            )[0]
            jac_rows.append(grad)
        return torch.stack(jac_rows, dim=-2)  # (B, 2, 3)


def project(camera: CameraModel, X: Tensor) -> Tensor:
    """Functional alias for camera.project(X)."""
    return camera.project(X)


def projection_jacobian(camera: CameraModel, X: Tensor) -> Tensor:
    """Functional alias for camera.jacobian(X)."""
    return camera.jacobian(X)
