# Sat-Splat-Distort

> Distortion-aware 3D Gaussian Splatting for satellite RPC, pushbroom, fisheye, and 360 equirectangular cameras. Research scaffold, in preparation.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/Arun0808/sat-splat-distort)
![Status: research scaffold](https://img.shields.io/badge/status-research%20scaffold-orange)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

3D Gaussian Splatting (3DGS) was designed for the perspective pinhole camera. Real-world remote sensing and immersive imagery (satellite RPC, pushbroom scanners, fisheye lenses, 360 panoramas) violate this assumption. Existing fixes either undistort first (which loses fidelity at the periphery) or hand-roll a per-camera approximation. The idea here is to replace the affine EWA splatting Jacobian with a closed-form distortion-aware Jacobian dispatched per camera model, plus a learned distortion-prior token grid for residual error.

## Status: what is real vs. what is scaffold

This repository is an early research scaffold. The geometric core is implemented and unit-tested on CPU, and a portable PyTorch rasterizer now renders, trains, and benches on synthetic data end to end; the optimized CUDA fork and the real-data loaders are still unbuilt. Read this section before cloning with any expectation of reproducing a real-dataset result.

**Real and tested (runs on CPU today):**

- Four per-camera closed-form analytic projection Jacobians: RPC (satellite), pushbroom, equidistant fisheye, and equirectangular 360, in `src/sat_splat/cameras/`. Each `project()` and `jacobian()` is implemented in plain PyTorch.
- A learned distortion-prior grid (`src/sat_splat/models/distortion_grid.py`): a 64x64x16 latent token field, bilinearly sampled at the projected pixel and passed through a 3-layer MLP to emit a symmetric 2x2 covariance perturbation.
- A portable differentiable rasterizer (`src/sat_splat/models/torch_rasterizer.py`): pure-PyTorch EWA splatting that projects each 3D covariance through the per-camera analytic Jacobian, adds the learned distortion perturbation, and alpha-composites front-to-back. `SatSplatPipeline.render()` uses it by default (`backend="torch"`), so render / train / bench run on CPU with no CUDA build. A synthetic data layer (`src/sat_splat/data/`) and a self-contained fit (`src/sat_splat/training/fit.py`) demonstrate end-to-end training: fitting a fresh Gaussian field to a rendered target lifts image PSNR from 7.5 to 19.2 dB in 120 CPU steps. These are SYNTHETIC fits, not a DFC2019 / Matterport benchmark.
- 26 unit tests pass on CPU: 7 validate the analytic camera Jacobians elementwise against `torch.autograd` (`tests/test_cameras.py`), 6 cover the rasterizer (render shape, differentiability, distortion perturbation, and the synthetic fit) in `tests/test_rasterizer.py`, and 13 are scaffold/Space smoke tests in `tests/test_smoke.py`.

**Scaffold / stub / not yet implemented (cannot run end to end):**

- The optimized CUDA rasterizer fork (`sat_splat._cuda_rasterizer`) **is not built** (there is no `cuda_rasterizer/` directory). It is an optional fast path: `SatSplatPipeline.render(backend="cuda")` raises `RuntimeError` asking you to build it, while the default `backend="torch"` uses the portable rasterizer above and runs without it.
- The real-data loaders **do not exist**. `sat_splat.data` ships only a synthetic scene/target generator; `DFC2019Dataset` and `Matterport360Dataset` (real satellite stereo / 360 panoramas) are not implemented.
- The synthetic fit runs end to end on CPU (see "What you can run today"). The Hydra `train.py` and `dfc2019_bench.py` real-data paths still need the missing loaders, and the Space "reconstruct" path is a placeholder; those remain reference skeletons for the real-data call graph.
- The Gradio Space serves a CPU-safe placeholder (a drawn preview image and a 3-vertex placeholder `.ply`); it does not fit a real scene.

No real-dataset benchmark has been run and no checkpoint published; the metrics table below is a set of unmeasured targets, not results. The only measured numbers are the synthetic fit's PSNR (above).

## What you can run today

```bash
git clone https://github.com/arunshar/sat-splat-distort
cd sat-splat-distort
pip install -e ".[dev]"
pytest                       # 26 tests on CPU (7 Jacobian-vs-autograd + 6 rasterizer + 13 scaffold/Space smoke)

# render + train a synthetic fit end to end (no CUDA, no data, ~10s on CPU):
python -c "from sat_splat.training.fit import fit_synthetic; print(fit_synthetic())"
```

The camera Jacobian tests are the meaningful check: they confirm each closed-form 2x3 Jacobian matches `torch.autograd` on random points in the model's operating region.

## Method

Standard 3DGS computes a 2D screen-space covariance via the EWA approximation:

```
Sigma_2D = J W Sigma_3D W^T J^T
```

where `J` is the affine Jacobian of the perspective projection. For a non-pinhole camera with mapping `pi: R^3 -> R^2`, this Jacobian is no longer affine. We compute the exact analytic `J = d pi / d X` at each Gaussian's mean position, dispatched per camera model:

| Camera | `pi(X)` | Closed-form Jacobian | Status |
| --- | --- | --- | --- |
| RPC | rational polynomial in (lat, lon, h) | analytic (chain rule on numerator/denominator polynomials) | implemented, autograd-validated |
| Pushbroom | row-wise pinhole + scan-time index | analytic | implemented, autograd-validated |
| Equidistant fisheye | angular projection `r = f * theta` | analytic | implemented, autograd-validated |
| Equirectangular | `(phi, theta)` from spherical coords | analytic | implemented, autograd-validated |

A small learned correction head conditions on a 64x64x16 distortion-prior grid (bilinearly sampled at the projected pixel) and outputs a 2x2 covariance perturbation intended to capture residual lens / RPC noise. Folding this perturbation into a real render requires the CUDA rasterizer, which is not yet implemented.

See [docs/architecture.md](docs/architecture.md) for the full derivation.

## Repository layout

```
sat-splat-distort/
├── src/sat_splat/
│   ├── cameras/{rpc,equirect,equidist,pushbroom,base}.py    # forward + analytic Jacobian per model  [implemented]
│   ├── models/distortion_grid.py                             # learned prior token field             [implemented]
│   ├── models/splat_pipeline.py                              # pipeline wrapper; render() needs the CUDA ext  [scaffold]
│   ├── training/train.py                                     # Hydra train loop; needs data + rasterizer       [scaffold, not runnable]
│   └── eval/dfc2019_bench.py                                 # eval skeleton; needs data + rasterizer           [scaffold, not runnable]
├── space/app.py                                              # Gradio Space; CPU placeholder output  [demo only]
├── configs/                                                  # Hydra configs
├── tests/                                                    # 7 Jacobian-vs-autograd + 13 scaffold smoke
├── paper/main.tex                                            # draft, in preparation
└── scripts/{download_dfc2019.sh, submit_msi.slurm}

NOT present (referenced by the scaffold but not yet implemented):
  src/sat_splat/data/        # DFC2019Dataset, Matterport360Dataset  -- do not exist
  src/sat_splat/_cuda_rasterizer  / cuda_rasterizer/  # differentiable rasterizer -- does not exist
```

## Intended evaluation (target metrics, NOT yet measured)

The table below states the leaderboard numbers this project aims at. **None of the "ours" numbers have been measured: no benchmark has been run, because no CUDA rasterizer or dataset is wired.** The baseline rows are quoted from the cited papers for context, not reproduced here. Do not cite any "ours" number as an achieved result.

DFC2019 Track-3 (held-out views), once the rasterizer and data layer exist:

```bash
# NOT yet runnable: requires src/sat_splat/data and the CUDA rasterizer, neither of which exists.
python -m sat_splat.eval.dfc2019_bench \
  --checkpoint hf://Arun0808/satsplat-distort-dfc2019 \
  --metric psnr ssim lpips
```

| Method | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| Sat-NeRF (ICCV 2023), reported by authors | 22.1 | 0.71 | 0.32 |
| EO-NeRF (CVPR 2023), reported by authors | 23.4 | 0.74 | 0.28 |
| 3DGS (no distortion), reported in cited work | 21.7 | 0.69 | 0.34 |
| Sat-Splat-Distort (ours) -- TARGET, NOT MEASURED | 24.2 (target) | 0.78 (target) | 0.24 (target) |

## Live demo

The [HF Space](https://huggingface.co/spaces/Arun0808/sat-splat-distort) boots the Gradio workflow on CPU and returns a placeholder preview and a placeholder `.ply`. It does not run a real 3DGS fit; the heavy reconstruction path is stubbed because there is no CPU/GPU rasterizer wired.

## Citation

This work is in preparation. If you reference the geometric formulation, please cite the draft:

```bibtex
@misc{sharma_satsplatdistort,
  title  = {Sat-Splat-Distort: Distortion-Aware Gaussian Splatting for Satellite, Pushbroom, Fisheye, and 360 Cameras},
  author = {Sharma, Arun},
  note   = {In preparation},
  year   = {2027}
}
```

## License

Apache 2.0. The intended CUDA rasterizer would fork the [INRIA Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) rasterizer and would inherit its non-commercial research license; that fork is not yet present in this repository.
