# Architecture and Jacobian Derivations

This note derives the closed-form 2x3 Jacobians for each camera model in
[src/sat_splat/cameras/](../src/sat_splat/cameras/) and explains the
distortion-prior grid head. The derivations are duplicated in code as
docstrings so the implementation file is self-contained.

Implementation status: the per-camera Jacobians and the distortion-prior grid
described below are implemented in PyTorch and unit-tested on CPU. The CUDA
rasterizer fork described in the final section is a plan, not yet implemented;
no `cuda_rasterizer/` or `sat_splat._cuda_rasterizer` exists in this repo.

## EWA splatting refresher

3DGS approximates the projected covariance of a 3D Gaussian as

    Sigma_2D = J W Sigma_3D W^T J^T

where W is the camera-frame rotation and J is the linearized projection
Jacobian at the Gaussian's mean. For a perspective pinhole camera with focal
length f and principal point (cx, cy), J at depth z is

    J = (1/z) [ [f, 0, -fx/z], [0, f, -fy/z] ].

This affine approximation is wrong whenever the projection is not pinhole.
Sat-Splat-Distort dispatches a per-camera-model J:

| Model | Closed-form Jacobian |
| --- | --- |
| Equirectangular | scale * d(phi, theta) / d X |
| Equidistant fisheye | f * d(theta * x_hat) / d X |
| Pushbroom | quotient rule on linear pushbroom |
| RPC | quotient rule on RPC polynomials, chained through normalization |

## Equirectangular Jacobian

Let X = (x, y, z), r = ||X||, rho = sqrt(x^2 + z^2). With

    phi   = atan2(x, z)
    theta = asin(y / r)

a direct computation yields:

    d phi / d x  =   z / rho^2
    d phi / d z  = - x / rho^2

    d theta / d x = - x y / (r^2 * rho)
    d theta / d y =   rho / r^2
    d theta / d z = - z y / (r^2 * rho)

Multiplying by the (W / (2 pi), -H / pi) image-scale factors gives the 2x3
Jacobian implemented in `equirect.py`.

## Equidistant fisheye Jacobian

For X = (x, y, z) with z > 0:

    r_3d = ||X||
    theta = acos(z / r_3d)
    rho   = sqrt(x^2 + y^2)
    s     = (f * theta) / rho

The Jacobian uses

    d theta / d x =   x z / (r_3d^2 * rho)
    d theta / d y =   y z / (r_3d^2 * rho)
    d theta / d z = - rho / r_3d^2

then chains through s and the product with x / y. See `equidist.py`.

## Pushbroom Jacobian

For the linear pushbroom model (Gupta and Hartley, PAMI 1997):

    u = f_u * (X . a) / (X . c)
    v = f_v * (X . b)

the Jacobian is

    du / dX = f_u * (a / (X . c) - (X . a) c / (X . c)^2)
    dv / dX = f_v * b.

## RPC Jacobian

The RPC model evaluates two polynomial ratios on normalized geographic
coordinates (L, P, H) and denormalizes the result to (sample, line). The
Jacobian uses the quotient rule on each polynomial ratio plus the chain
rule through the normalization. See `rpc.py` for the implementation and
`tests/test_cameras.py` for the autograd validation.

## Distortion prior grid

A 64x64x16 latent grid is parked over the image plane. For each Gaussian we
bilinearly sample the grid at its projected pixel and feed the latent
through a 3-layer MLP to produce a symmetric 2x2 covariance perturbation:

    Sigma_2D <- J W Sigma_3D W^T J^T + dSigma_grid(uv)

The MLP is initialized to output zero so the analytic Jacobian dominates
early training, then the grid learns to absorb residual lens / RPC noise.

## CUDA rasterizer fork

We fork [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)'s CUDA rasterizer
and replace `forward.cu`'s `computeCov2D` with a kernel that:

1. dispatches on a runtime camera-model flag (RPC / pushbroom / equidist / equirect),
2. evaluates the analytic 2x3 Jacobian using the closed-form expressions in
   the corresponding Python class (port to CUDA),
3. reads the distortion-prior perturbation from a precomputed (uv, dSigma)
   buffer produced by `DistortionPriorGrid` on the host.

Steps 1-2 are the bulk of the CUDA porting work; step 3 is a small buffer
read.
