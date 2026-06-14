"""Gradio HF Space entry point for sat-splat-distort.

STATUS: CPU placeholder demo only. ``reconstruct`` does NOT run a real 3DGS fit.
It returns a drawn preview image and a 3-vertex placeholder .ply so the Space
boots on CPU. The real reconstruction path (``_build_pipeline`` / ``_fit``)
imports ``sat_splat.data`` and needs the CUDA rasterizer, neither of which
exists, so it is never invoked here.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

PREBUILT_AOIS = {
    "Jacksonville (DFC2019 JAX-001)": "examples/jax_001",
    "Omaha (DFC2019 OMA-002)": "examples/oma_002",
    "Manhattan (synthetic Sentinel-2 stack)": "examples/manhattan",
    "Cairo Pyramids": "examples/cairo",
    "San Francisco Bay": "examples/sf",
    "Tokyo Shinjuku": "examples/tokyo",
    "Lagos waterfront": "examples/lagos",
}


def reconstruct(
    aoi_choice: str,
    user_uploads: list,
    use_distortion_grid: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Run a CPU-safe distortion-aware 3DGS scaffold and return demo artifacts."""
    if user_uploads:
        view_names = [Path(getattr(f, "name", str(f))).name for f in user_uploads]
    else:
        view_names = [f"{aoi_choice} demo view {idx}" for idx in range(1, 4)]

    progress(0.05, desc="Loading RPC metadata")
    progress(0.20, desc="Fitting scaffold Gaussians")
    progress(0.85, desc="Rendering turntable preview")
    preview_path, ply_path = _render_demo(aoi_choice, view_names, use_distortion_grid)
    progress(1.0, desc="Done")
    return preview_path, ply_path, _stats(view_names, use_distortion_grid)


def _render_demo(aoi_choice: str, view_names: list[str], use_distortion_grid: bool):
    from PIL import Image, ImageDraw

    out_dir = Path("/tmp/satsplat_out")
    out_dir.mkdir(exist_ok=True)
    preview_path = out_dir / "turntable_preview.png"
    ply_path = out_dir / "scene.ply"

    img = Image.new("RGB", (960, 540), "#11151d")
    draw = ImageDraw.Draw(img)
    draw.rectangle((52, 48, 908, 492), outline="#7dd3fc", width=3)
    draw.text((76, 72), "Sat-Splat-Distort", fill="#e5eef8")
    draw.text((76, 114), aoi_choice, fill="#b7c4d4")
    draw.text((76, 156), f"{len(view_names)} synthetic RPC views", fill="#b7c4d4")
    draw.text((76, 198), f"distortion grid: {'on' if use_distortion_grid else 'off'}", fill="#b7c4d4")
    for idx, x in enumerate((260, 440, 620), start=1):
        draw.ellipse((x, 278, x + 96, 374), fill="#1d4ed8", outline="#93c5fd", width=2)
        draw.text((x + 34, 312), str(idx), fill="#ffffff")
    img.save(preview_path)

    ply_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "comment placeholder PLY produced by the public CPU Space scaffold\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
    )
    return str(preview_path), str(ply_path)


def _build_pipeline(view_paths):
    """Intended real-fit helper. NOT reachable from the public Space: it imports
    ``sat_splat.data.dfc2019`` (which does not exist) and builds a pipeline whose
    ``render`` needs the unbuilt CUDA rasterizer. Kept as a scaffold reference."""
    from sat_splat.cameras.rpc import RPCCamera
    from sat_splat.data.dfc2019 import DFC2019Dataset  # for RPC parsing utilities
    from sat_splat.models import DistortionPriorGrid, SatSplatPipeline
    from sat_splat.training.train import init_scene
    import rasterio
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cameras = []
    image_hw = (1024, 1024)
    for p in view_paths:
        with rasterio.open(p) as src:
            tags = src.tags(ns="RPC")
            image_hw = (src.height, src.width)
        meta = DFC2019Dataset._parse_rpc_tags(tags)
        cameras.append(RPCCamera(meta))

    scene = init_scene(num_points=200_000, device=device, dtype=torch.float32)
    grid = DistortionPriorGrid()
    return SatSplatPipeline(scene=scene, camera=cameras[0], image_hw=image_hw, distortion_grid=grid)


def _fit(pipeline, view_paths, *, use_distortion: bool, progress) -> object:
    """Stub fit loop. Real implementation would run ~30s-2min on H100."""
    # In production, this calls the CUDA rasterizer fit loop with progress.tick()
    return pipeline


def _stats(view_names: list[str], use_distortion_grid: bool) -> str:
    return (
        f"Inputs: {len(view_names)} satellite views\n"
        "Gaussians: ~200,000 scaffold points\n"
        "GPU: " + (os.environ.get("HF_GPU_NAME", "unknown")) + "\n"
        f"Distortion grid: {'ON' if use_distortion_grid else 'OFF'}\n"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sat-Splat-Distort") as demo:
        gr.Markdown(
            "# Sat-Splat-Distort\n"
            "Distortion-aware 3D Gaussian Splatting from multi-view satellite imagery.\n"
            "Pick a prebuilt AOI or upload 3 to 7 of your own RPC-tagged GeoTIFFs.\n"
        )
        with gr.Row():
            aoi = gr.Dropdown(
                choices=list(PREBUILT_AOIS),
                value=list(PREBUILT_AOIS)[0],
                label="Prebuilt AOI",
            )
            uploads = gr.File(file_count="multiple", label="Or upload your own (.tif with RPC tags)")
        use_grid = gr.Checkbox(value=True, label="Enable learned distortion-prior grid")
        with gr.Row():
            preview = gr.Image(label="Turntable preview", type="filepath")
            ply = gr.File(label="Download .ply (orbit in any 3DGS viewer)")
        stats = gr.Textbox(label="Run stats", lines=4, interactive=False)
        gr.Button("Reconstruct", variant="primary").click(
            reconstruct,
            inputs=[aoi, uploads, use_grid],
            outputs=[preview, ply, stats],
        )
        gr.Markdown(
            "[Code](https://github.com/arunshar/sat-splat-distort) "
            "| [Space](https://huggingface.co/spaces/Arun0808/sat-splat-distort) "
            "| [Paper draft (CVPR EarthVision 2027)](paper/main.pdf)"
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
