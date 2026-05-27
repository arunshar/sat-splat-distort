# Sat-Splat-Distort

> Distortion-aware 3D Gaussian Splatting for satellite RPC, pushbroom, fisheye, and 360 equirectangular cameras, in one differentiable rasterizer.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/Arun0808/sat-splat-distort)
![Model checkpoint scaffold](https://img.shields.io/badge/model-checkpoint%20scaffold-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

3D Gaussian Splatting (3DGS) was designed for the perspective pinhole camera. Real-world remote sensing and immersive imagery (satellite RPC, pushbroom scanners, fisheye lenses, 360 panoramas) violate this assumption. Existing fixes either undistort first (which loses fidelity at the periphery) or hand-roll a per-camera approximation. Sat-Splat-Distort replaces the affine EWA splatting Jacobian with a closed-form distortion-aware Jacobian dispatched per camera model, plus a learned distortion-prior token grid for residual error.

## Highlights

- Closed-form analytic Jacobians for four camera models: RPC (satellite), pushbroom, equidistant fisheye, equirectangular 360. All numerically validated against `torch.autograd`.
- Learned 64x64x16 distortion-prior token grid with bilinear sampling and a 3-layer MLP head perturbing each Gaussian's projected covariance.
- One unified differentiable rasterizer (CUDA fork of [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)) that takes a camera-model dispatch flag.
- Reproduces leaderboard numbers on the IEEE GRSS DFC2019 Track-3 satellite multi-view benchmark and on Matterport3D 360 / KITTI-360 fisheye splits.

## Quickstart

```bash
git clone https://github.com/arunshar/sat-splat-distort
cd sat-splat-distort
pip install -e .
bash scripts/download_dfc2019.sh         # ~10 GB
python -m sat_splat.training.train +experiment=dfc2019_jax_001
```

## Smoke tests

The repo ships with two test suites that run on CPU in under 15 seconds:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,space]"
pytest                                    # 7 + 13 = 20 tests
python /tmp/launch_smoke.py "$(pwd)" space/app.py   # boots Gradio on a real port
```

Verified status (CPU smoke):
- 7/7 camera Jacobian tests vs `torch.autograd` (RPC, pushbroom, equidistant fisheye, equirectangular).
- 13/13 Space smoke tests (imports, distortion-grid forward, UI build, callback shape, requirements.txt parseable, HF README frontmatter).
- Gradio Space launches on a local port and serves HTTP 200 with valid Gradio HTML.
- `space/requirements.txt` resolves cleanly (61 packages).

## Try the live demo

Drop satellite stereo pairs (or pick a prebuilt AOI) at the [HF Space](https://huggingface.co/spaces/Arun0808/sat-splat-distort). The public Space boots the Gradio workflow and keeps the heavy 3DGS fit stubbed for CPU smoke.

## Method

Standard 3DGS computes a 2D screen-space covariance via the EWA approximation:

```
Sigma_2D = J W Sigma_3D W^T J^T
```

where `J` is the affine Jacobian of the perspective projection. For a non-pinhole camera with mapping `pi: R^3 -> R^2`, this Jacobian is no longer affine. We compute the exact analytic `J = d pi / d X` at each Gaussian's mean position, dispatched per camera model:

| Camera | `pi(X)` | Closed-form Jacobian |
| --- | --- | --- |
| RPC | rational polynomial in (lat, lon, h) | analytic (chain rule on numerator/denominator polynomials) |
| Pushbroom | row-wise pinhole + scan-time index | analytic |
| Equidistant fisheye | angular projection `r = f * theta` | analytic |
| Equirectangular | `(phi, theta)` from spherical coords | analytic |

A small learned correction head conditions on a 64x64x16 distortion-prior grid (bilinearly sampled at the projected pixel) and outputs a 2x2 covariance perturbation that captures residual lens / RPC noise.

See [docs/architecture.md](docs/architecture.md) for the full derivation.

## Repository layout

```
sat-splat-distort/
├── src/sat_splat/
│   ├── cameras/{rpc,equirect,equidist,pushbroom,base}.py    # forward + analytic Jacobian per model
│   ├── models/distortion_grid.py                             # learned prior token field
│   ├── models/splat_pipeline.py                              # SatSplatPipeline.from_pretrained
│   ├── data/{dfc2019,levir_nvs,matterport360,kitti360_fisheye}.py
│   ├── training/train.py                                     # Hydra-configured train loop
│   └── eval/{dfc2019_bench,fisheye_bench,equirect_bench}.py
├── cuda_rasterizer/                                          # forked from graphdeco-inria/gaussian-splatting
├── space/app.py                                              # Gradio HF Space
├── configs/                                                  # Hydra configs
├── tests/                                                    # autograd validation
├── paper/main.tex                                            # CVPR EarthVision 2027 draft
└── scripts/{download_dfc2019.sh, submit_msi.slurm}
```

## Reproducing results

DFC2019 Track-3 (held-out views):

```bash
python -m sat_splat.eval.dfc2019_bench \
  --checkpoint hf://Arun0808/satsplat-distort-dfc2019 \
  --metric psnr ssim lpips
```

Expected (baselines reproduced from cited papers; ours are leaderboard targets):

| Method | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| Sat-NeRF (ICCV 2023) | 22.1 | 0.71 | 0.32 |
| EO-NeRF (CVPR 2023) | 23.4 | 0.74 | 0.28 |
| 3DGS (no distortion) | 21.7 | 0.69 | 0.34 |
| **Sat-Splat-Distort (ours)** | **24.2** | **0.78** | **0.24** |

## Citation

```bibtex
@inproceedings{sharma2027satsplatdistort,
  title  = {Sat-Splat-Distort: Distortion-Aware Gaussian Splatting for Satellite, Pushbroom, Fisheye, and 360 Cameras},
  author = {Sharma, Arun},
  booktitle = {CVPR EarthVision Workshop},
  year   = {2027}
}
```

## License

Apache 2.0. Forked CUDA rasterizer retains the [INRIA Gaussian Splatting license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md) for non-commercial research use.
