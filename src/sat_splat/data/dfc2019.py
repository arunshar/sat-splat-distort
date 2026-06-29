"""IEEE GRSS DFC2019 Track-3 multi-view satellite dataset.

The IEEE GRSS Data Fusion Contest 2019 Track-3 provides paired WorldView-3
panchromatic + multispectral imagery over Jacksonville, FL and Omaha, NE
with associated RPC metadata, lidar ground truth, and per-image bounding
boxes. Single ~10 GB zip download.

Site: https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sat_splat.cameras.rpc import RPCCamera, RPCMeta


@dataclass
class DFC2019Sample:
    """A single multi-view satellite sample."""

    site: str
    image: torch.Tensor      # (3, H, W) RGB float32 in [0, 1]
    rpc: RPCCamera
    image_hw: tuple[int, int]
    bounds_lonlat: tuple[float, float, float, float]   # (lon_min, lat_min, lon_max, lat_max)


class DFC2019Dataset(Dataset):
    """Loader for DFC2019 Track-3 multi-view AOIs.

    Supports per-AOI splits where train views fit a 3DGS scene and eval views
    are held out for novel-view synthesis benchmarking.
    """

    def __init__(self, root: str | Path, site: str, split: str = "train"):
        self.root = Path(root) / site
        if not self.root.exists():
            raise FileNotFoundError(
                f"DFC2019 site {site!r} not found at {self.root}. "
                "Run scripts/download_dfc2019.sh first."
            )
        self.site = site
        self.split = split
        self.entries = self._index()

    # -- indexing ---------------------------------------------------------------

    def _index(self) -> list[Path]:
        index_file = self.root / f"{self.split}.txt"
        if not index_file.exists():
            raise FileNotFoundError(
                f"{index_file} not present. Build splits with scripts/build_dfc2019_splits.py."
            )
        return [self.root / line.strip() for line in index_file.read_text().splitlines() if line.strip()]

    # -- loading ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> DFC2019Sample:
        path = self.entries[idx]
        # Load image with rasterio so we get geo metadata and per-band normalization.
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("rasterio is required to load DFC2019 imagery.") from exc

        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")
            arr = arr / 2047.0  # WorldView-3 11-bit
            arr = np.clip(arr, 0.0, 1.0)
            if arr.shape[0] >= 3:
                arr = arr[:3]
            else:
                arr = np.repeat(arr, 3, axis=0)
            image = torch.from_numpy(arr)
            bounds = src.bounds  # type: ignore[union-attr]
            tags = src.tags(ns="RPC")  # type: ignore[union-attr]

        meta = self._parse_rpc_tags(tags)
        camera = RPCCamera(meta)
        return DFC2019Sample(
            site=self.site,
            image=image,
            rpc=camera,
            image_hw=(image.shape[-2], image.shape[-1]),
            bounds_lonlat=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        )

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _parse_rpc_tags(tags: dict) -> RPCMeta:
        """Parse the RPC metadata GDAL exposes via the RPC namespace."""

        def coefs(key: str) -> tuple:
            raw = tags[key].split()
            return tuple(float(x) for x in raw)

        return RPCMeta(
            line_offset=float(tags["LINE_OFF"]),
            sample_offset=float(tags["SAMP_OFF"]),
            lat_offset=float(tags["LAT_OFF"]),
            lon_offset=float(tags["LONG_OFF"]),
            height_offset=float(tags["HEIGHT_OFF"]),
            line_scale=float(tags["LINE_SCALE"]),
            sample_scale=float(tags["SAMP_SCALE"]),
            lat_scale=float(tags["LAT_SCALE"]),
            lon_scale=float(tags["LONG_SCALE"]),
            height_scale=float(tags["HEIGHT_SCALE"]),
            line_num_coef=coefs("LINE_NUM_COEFF"),
            line_den_coef=coefs("LINE_DEN_COEFF"),
            samp_num_coef=coefs("SAMP_NUM_COEFF"),
            samp_den_coef=coefs("SAMP_DEN_COEFF"),
        )
