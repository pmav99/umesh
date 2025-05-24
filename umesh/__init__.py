from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version(__name__)

from ._api import VTKCompressionType
from ._api import VTKDataModeType
from ._api import calc_mesh_quality
from ._api import calc_metric
from ._api import clip
from ._api import read
from ._api import read_vtk
from ._api import read_vtu
from ._api import reproject
from ._api import create_ugrid
from ._api import to_vtu
from ._api import write_vtk
from ._api import write_vtu


__all__ = (
    "VTKCompressionType",
    "VTKDataModeType",
    "calc_mesh_quality",
    "calc_metric",
    "clip",
    "read",
    "read_vtk",
    "read_vtu",
    "reproject",
    "create_ugrid",
    "to_vtu",
    "write_vtk",
    "write_vtu",
    "__version__",
)
