from sat_splat.cameras.base import CameraModel, project, projection_jacobian
from sat_splat.cameras.equidist import EquidistantFisheye
from sat_splat.cameras.equirect import Equirectangular
from sat_splat.cameras.pushbroom import Pushbroom
from sat_splat.cameras.rpc import RPCCamera

__all__ = [
    "CameraModel",
    "Equirectangular",
    "EquidistantFisheye",
    "Pushbroom",
    "RPCCamera",
    "project",
    "projection_jacobian",
]


def from_name(name: str, **kwargs):
    """Dispatch a camera model by name. Used by Hydra configs."""
    table = {
        "rpc": RPCCamera,
        "pushbroom": Pushbroom,
        "equidist": EquidistantFisheye,
        "fisheye": EquidistantFisheye,
        "equirect": Equirectangular,
        "panorama": Equirectangular,
        "360": Equirectangular,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"unknown camera model {name!r}; expected one of {list(table)}")
    return table[key](**kwargs)
