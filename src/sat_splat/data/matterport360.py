"""Minimal Matterport3D 360 panorama loader.

Matterport3D ships per-room panorama bundles under skybox/<scene>/<idx>.
We load the equirectangular pano + the camera-to-world transform and yield
samples ready for an Equirectangular projection.

Note: Matterport requires registering an academic license. Place the
unpacked archive at data/matterport3d/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sat_splat.cameras.equirect import Equirectangular


@dataclass
class Matterport360Sample:
    scene: str
    image: torch.Tensor      # (3, H, W)
    camera: Equirectangular
    pose_w2c: torch.Tensor   # (4, 4)


class Matterport360Dataset(Dataset):
    """Loads a single Matterport3D scene's set of panoramas."""

    def __init__(self, root: str | Path, scene: str, image_hw: tuple[int, int] = (1024, 2048)):
        self.root = Path(root) / scene
        self.scene = scene
        self.image_hw = image_hw
        self.entries = sorted(self.root.glob("*.png"))
        if not self.entries:
            raise FileNotFoundError(
                f"No panoramas found at {self.root}. Did you unpack matterport3d?"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Matterport360Sample:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("imageio is required to load Matterport panoramas.") from exc

        img = iio.imread(self.entries[idx])
        image = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        H, W = self.image_hw
        if image.shape[-2:] != (H, W):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)

        pose = self._load_pose(self.entries[idx])
        camera = Equirectangular(width=W, height=H)
        return Matterport360Sample(
            scene=self.scene, image=image, camera=camera, pose_w2c=pose
        )

    def _load_pose(self, image_path: Path) -> torch.Tensor:
        pose_path = image_path.with_suffix(".pose.txt")
        if not pose_path.exists():
            return torch.eye(4)
        return torch.from_numpy(
            __import__("numpy").loadtxt(pose_path, dtype="float32")
        )
