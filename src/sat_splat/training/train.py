"""Hydra-driven training entry point for distortion-aware splatting.

Usage:

    python -m sat_splat.training.train +experiment=dfc2019_jax_001

Configs live under configs/. The training loop fits a per-scene SplatScene +
DistortionPriorGrid with the standard L1+SSIM loss and a covariance-smoothness
regularizer. Logs go to W&B under project ``sat-splat-distort``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def make_loader(cfg: DictConfig) -> DataLoader:
    if cfg.dataset.kind == "dfc2019":
        from sat_splat.data import DFC2019Dataset
        ds = DFC2019Dataset(root=cfg.dataset.root, site=cfg.dataset.site, split=cfg.dataset.split)
    elif cfg.dataset.kind == "matterport360":
        from sat_splat.data import Matterport360Dataset
        ds = Matterport360Dataset(
            root=cfg.dataset.root,
            scene=cfg.dataset.scene,
            image_hw=tuple(cfg.dataset.image_hw),
        )
    else:
        raise ValueError(f"unknown dataset kind {cfg.dataset.kind!r}")
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=cfg.dataset.workers)


def init_scene(num_points: int, device, dtype) -> "SplatScene":
    from sat_splat.models.splat_pipeline import SplatScene
    means = torch.randn(num_points, 3, device=device, dtype=dtype) * 0.5
    scales = torch.zeros(num_points, 3, device=device, dtype=dtype)  # log-scales
    quats = torch.tensor([1, 0, 0, 0], device=device, dtype=dtype).expand(num_points, 4).clone()
    opacities = torch.full((num_points, 1), -2.0, device=device, dtype=dtype)
    colors = torch.zeros(num_points, 3, 16, device=device, dtype=dtype)
    return SplatScene(means=means, scales=scales, quats=quats, opacities=opacities, colors=colors)


def l1_ssim_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    l1 = (pred - target).abs().mean()
    # Crude SSIM stand-in for the skeleton; replace with kornia.metrics.ssim
    # in production training.
    mu_pred = pred.mean()
    mu_t = target.mean()
    var_pred = pred.var()
    var_t = target.var()
    cov = ((pred - mu_pred) * (target - mu_t)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_pred * mu_t + c1) * (2 * cov + c2)) / (
        (mu_pred ** 2 + mu_t ** 2 + c1) * (var_pred + var_t + c2)
    )
    return alpha * (1.0 - ssim) + (1 - alpha) * l1


@hydra.main(version_base=None, config_path=str(Path(__file__).parents[3] / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    logger.info("loaded config:\n%s", OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    dtype = torch.float32

    loader = make_loader(cfg)
    scene = init_scene(cfg.train.num_points, device, dtype)

    from sat_splat.cameras import from_name as camera_from_name
    from sat_splat.models import DistortionPriorGrid, SatSplatPipeline

    camera = camera_from_name(cfg.camera.kind, **cfg.camera.kwargs)
    grid = DistortionPriorGrid(
        grid_h=cfg.distortion_grid.h,
        grid_w=cfg.distortion_grid.w,
        token_dim=cfg.distortion_grid.dim,
    )
    image_hw = tuple(cfg.image_hw)
    pipeline = SatSplatPipeline(scene=scene, camera=camera, image_hw=image_hw, distortion_grid=grid).to(device)

    opt = torch.optim.Adam(pipeline.parameters(), lr=cfg.train.lr)

    if cfg.logging.wandb:
        import wandb
        wandb.init(project="sat-splat-distort", config=OmegaConf.to_container(cfg, resolve=True))

    step = 0
    for epoch in range(cfg.train.epochs):
        for sample in loader:
            opt.zero_grad()
            pred = pipeline.render(with_distortion=cfg.distortion_grid.enabled)
            target = sample["image"][0].to(device)
            loss = l1_ssim_loss(pred, target)
            loss = loss + cfg.distortion_grid.reg_weight * pipeline.distortion_grid.regularization()
            loss.backward()
            opt.step()
            step += 1
            if step % cfg.logging.every == 0:
                logger.info("step=%d loss=%.4f", step, loss.item())
                if cfg.logging.wandb:
                    wandb.log({"loss": loss.item(), "step": step})

    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(out)
    logger.info("saved checkpoint to %s", out)


if __name__ == "__main__":
    main()
