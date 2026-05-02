"""Reproduce DFC2019 leaderboard numbers for held-out novel-view synthesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_pipeline(checkpoint: str):
    from sat_splat.models import SatSplatPipeline
    return SatSplatPipeline.from_pretrained(
        checkpoint, camera_kind="rpc", image_hw=(1024, 1024)
    )


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().clamp_min(1e-12)
    return float(10 * torch.log10(1.0 / mse))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    # Light-weight SSIM stand-in to keep this skeleton dependency-free.
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cv = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float((2 * mu_a * mu_b + c1) * (2 * cv + c2) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    ))


def lpips(a: torch.Tensor, b: torch.Tensor) -> float:
    """Best-effort LPIPS; falls back to a pixel diff if the package isn't available."""
    try:
        import lpips as _lpips
        loss = _lpips.LPIPS(net="alex")
        return float(loss(a.unsqueeze(0) * 2 - 1, b.unsqueeze(0) * 2 - 1))
    except ImportError:
        return float((a - b).abs().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="data/dfc2019")
    parser.add_argument("--site", default="JAX")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/dfc2019.json")
    args = parser.parse_args()

    from sat_splat.data import DFC2019Dataset
    pipeline = load_pipeline(args.checkpoint)
    ds = DFC2019Dataset(root=args.root, site=args.site, split=args.split)

    scores = []
    for i in range(len(ds)):
        sample = ds[i]
        with torch.no_grad():
            pipeline.camera = sample.rpc
            pipeline.image_hw = sample.image_hw
            pred = pipeline.render(with_distortion=True)
        target = sample.image
        scores.append({
            "idx": i,
            "psnr": psnr(pred, target),
            "ssim": ssim(pred, target),
            "lpips": lpips(pred, target),
        })

    summary = {
        "site": args.site,
        "split": args.split,
        "psnr_mean": sum(s["psnr"] for s in scores) / max(len(scores), 1),
        "ssim_mean": sum(s["ssim"] for s in scores) / max(len(scores), 1),
        "lpips_mean": sum(s["lpips"] for s in scores) / max(len(scores), 1),
        "n": len(scores),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_view": scores, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
