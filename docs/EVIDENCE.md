# sat-splat-distort — reproduced evidence

_Generated 2026-06-29T10:30:32Z by running the test suite on the real/tested code in this repo._

These are **reproduced** results: the code runs and every assertion below holds. Benchmark/leaderboard numbers in the paper (PSNR, mIoU, speedups) remain **targets, not reproduced**, and are labeled as such throughout.

## Test suite (`pytest -v`)

```
tests/test_cameras.py::test_equirect_jacobian_matches_autograd PASSED    [  5%]
tests/test_cameras.py::test_equirect_handles_general_directions PASSED   [ 10%]
tests/test_cameras.py::test_equidist_jacobian_matches_autograd PASSED    [ 15%]
tests/test_cameras.py::test_equidist_handles_wide_angles PASSED          [ 20%]
tests/test_cameras.py::test_pushbroom_jacobian_matches_autograd PASSED   [ 25%]
tests/test_cameras.py::test_rpc_jacobian_matches_autograd_simple PASSED  [ 30%]
tests/test_cameras.py::test_rpc_jacobian_full_cubic_terms PASSED         [ 35%]
tests/test_smoke.py::test_top_level_imports PASSED                       [ 40%]
tests/test_smoke.py::test_camera_imports PASSED                          [ 45%]
tests/test_smoke.py::test_models_imports PASSED                          [ 50%]
tests/test_smoke.py::test_distortion_grid_perturbation_is_symmetric_psd PASSED [ 55%]
tests/test_smoke.py::test_distortion_grid_regularization_is_scalar PASSED [ 60%]
tests/test_smoke.py::test_equirect_pipeline_e2e PASSED                   [ 65%]
tests/test_smoke.py::test_fisheye_pipeline_e2e PASSED                    [ 70%]
tests/test_smoke.py::test_space_app_module_importable PASSED             [ 75%]
tests/test_smoke.py::test_space_ui_builds PASSED                         [ 80%]
tests/test_smoke.py::test_space_app_constants_present PASSED             [ 85%]
tests/test_smoke.py::test_space_callback_returns_demo_artifacts_on_missing_inputs PASSED [ 90%]
tests/test_smoke.py::test_space_requirements_file_parseable PASSED       [ 95%]
tests/test_smoke.py::test_space_readme_has_hf_frontmatter PASSED         [100%]

============================== 20 passed in 1.60s ==============================
```

## Reproduced demo (headline number)

Analytic camera Jacobians match `torch.autograd` to within ~1e-5 relative error (equirectangular ~1.8e-6, fisheye ~1.0e-5), across the camera models.
