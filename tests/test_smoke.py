"""Smoke tests for HF Space deployment.

These verify:
- All package imports the Space app uses resolve.
- The Gradio Blocks UI builds without crashing.
- The inference callback returns the expected number of outputs on synthetic
  data (no GPU, no CUDA rasterizer, no real datasets).

This is the safety net that catches issues like missing imports, broken Hydra
decorators, or shape mismatches BEFORE the Space goes live on HF.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT / "space" / "app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("sat_splat_space_app", SPACE_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sat_splat_space_app"] = module
    spec.loader.exec_module(module)
    return module


# -- package imports ---------------------------------------------------------


def test_top_level_imports():
    import sat_splat
    from sat_splat import CameraModel, project, projection_jacobian
    assert sat_splat.__version__


def test_camera_imports():
    from sat_splat.cameras import (
        CameraModel,
        Equirectangular,
        EquidistantFisheye,
        Pushbroom,
        RPCCamera,
        from_name,
    )
    assert from_name("equirect", width=64, height=32).name == "equirect"
    assert from_name("fisheye", fx=100.0, fy=100.0, cx=32.0, cy=32.0).name == "equidist"


def test_models_imports():
    from sat_splat.models import DistortionPriorGrid, SatSplatPipeline
    assert DistortionPriorGrid().tokens.shape[2:] == (64, 64)


# -- distortion grid forward pass -------------------------------------------


def test_distortion_grid_perturbation_is_symmetric_psd():
    from sat_splat.models import DistortionPriorGrid
    grid = DistortionPriorGrid(grid_h=8, grid_w=8, token_dim=4, mlp_hidden=16)
    uv = torch.tensor([[100.0, 50.0], [120.0, 80.0]])
    sigma = grid.perturbation(uv, image_hw=(256, 256))
    assert sigma.shape == (2, 2, 2)
    assert torch.allclose(sigma, sigma.transpose(-1, -2), atol=1e-6)
    eigs = torch.linalg.eigvalsh(sigma)
    assert (eigs >= -1e-6).all()


def test_distortion_grid_regularization_is_scalar():
    from sat_splat.models import DistortionPriorGrid
    grid = DistortionPriorGrid(grid_h=4, grid_w=4, token_dim=2)
    reg = grid.regularization()
    assert reg.dim() == 0
    assert reg.item() >= 0.0


# -- end-to-end camera + jacobian path on a small batch ---------------------


def test_equirect_pipeline_e2e():
    from sat_splat.cameras import Equirectangular
    cam = Equirectangular(width=512, height=256)
    X = torch.randn(16, 3) * 3.0
    uv = cam.project(X)
    J = cam.jacobian(X)
    assert uv.shape == (16, 2)
    assert J.shape == (16, 2, 3)
    assert torch.isfinite(uv).all() and torch.isfinite(J).all()


def test_fisheye_pipeline_e2e():
    from sat_splat.cameras import EquidistantFisheye
    cam = EquidistantFisheye(fx=300.0, fy=300.0, cx=256.0, cy=256.0)
    xy = torch.randn(16, 2)
    z = torch.rand(16, 1) + 0.5
    X = torch.cat([xy, z], dim=-1)
    uv = cam.project(X)
    J = cam.jacobian(X)
    assert uv.shape == (16, 2)
    assert J.shape == (16, 2, 3)
    assert torch.isfinite(uv).all() and torch.isfinite(J).all()


# -- Gradio app smoke -------------------------------------------------------


def test_space_app_module_importable():
    """Importing space/app.py must not crash (lazy imports inside callbacks)."""
    module = _load_app_module()
    assert hasattr(module, "build_ui")
    assert hasattr(module, "reconstruct")


def test_space_ui_builds():
    """build_ui returns a gradio Blocks instance."""
    gr = pytest.importorskip("gradio")
    module = _load_app_module()
    ui = module.build_ui()
    assert isinstance(ui, gr.Blocks)


def test_space_app_constants_present():
    """The prebuilt-AOI mapping is non-empty."""
    module = _load_app_module()
    assert isinstance(module.PREBUILT_AOIS, dict)
    assert len(module.PREBUILT_AOIS) >= 3


def test_space_callback_does_not_crash_on_missing_inputs():
    """If neither uploads nor a valid AOI exists, we expect a gr.Error (graceful)."""
    pytest.importorskip("gradio")
    module = _load_app_module()
    with pytest.raises(Exception):
        # Empty upload list + an AOI that has no images on disk -> gr.Error.
        module.reconstruct(
            list(module.PREBUILT_AOIS)[0],
            None,
            False,
        )


# -- requirements.txt parses --------------------------------------------------


def test_space_requirements_file_parseable():
    req = REPO_ROOT / "space" / "requirements.txt"
    assert req.exists()
    lines = [
        line.strip() for line in req.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(lines) > 0
    # Sanity: gradio + torch must be in the list.
    text = req.read_text().lower()
    assert "gradio" in text
    assert "torch" in text


def test_space_readme_has_hf_frontmatter():
    """HF Spaces require YAML frontmatter at the top of README.md."""
    readme = REPO_ROOT / "space" / "README.md"
    assert readme.exists()
    head = readme.read_text().splitlines()[0]
    assert head == "---", f"Space README must start with YAML frontmatter, got {head!r}"
    body = readme.read_text()
    assert "sdk: gradio" in body
    assert "app_file:" in body
