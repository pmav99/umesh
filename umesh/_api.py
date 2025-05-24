# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
from __future__ import annotations

import dataclasses
import enum
import functools
import os
import pathlib
import typing as T
from collections import abc

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import numpy_to_vtkIdTypeArray
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkCommonDataModel import VTK_TRIANGLE
from vtkmodules.vtkCommonDataModel import vtkBox
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkFiltersExtraction import vtkExtractGeometry
from vtkmodules.vtkFiltersVerdict import vtkMeshQuality
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from ._utils import parse_gr3
# from vtkmodules.vtkFiltersExtraction import vtkExtractUnstructuredGrid

if T.TYPE_CHECKING:
    import pandas as pd
    import numpy.typing as npt
    import pyproj

    # Types
    Path = os.PathLike[str] | str
    NPArrayF = npt.NDArray[np.float64]
    NPArrayI = npt.NDArray[np.int32]


class VTK_TRIANGLE_METRICS(enum.StrEnum):
    AREA = enum.auto()
    ASPECT_FROBENIUS = enum.auto()
    INVERSE_FROBENIUS = enum.auto()
    ASPECT_RATIO = enum.auto()
    INVERSE_ASPECT_RATIO = enum.auto()
    CONDITION = enum.auto()
    DISTORTION = enum.auto()
    EDGE_RATIO = enum.auto()
    INVERSE_EDGE_RATIO = enum.auto()
    EQUIANGLE_SKEW = enum.auto()
    MAX_ANGLE = enum.auto()
    MIN_ANGLE = enum.auto()
    NORMALIZED_INRADIUS = enum.auto()
    RADIUS_RATIO = enum.auto()
    RELATIVE_SIZE_SQUARED = enum.auto()
    SCALED_JACOBIAN = enum.auto()
    SHAPE = enum.auto()
    SHAPE_AND_SIZE = enum.auto()


@dataclasses.dataclass
class PlotInfo:
    title: str | None
    suptitle: str


_PLOT_DATA = {
    VTK_TRIANGLE_METRICS.EDGE_RATIO: PlotInfo(
        title="Hmax / Hmin where Hmax and Hmin are respectively the maximum and the minimum edge lengths",
        suptitle="Edge Ratio",
    ),
    VTK_TRIANGLE_METRICS.INVERSE_EDGE_RATIO: PlotInfo(
        title="Hmin / Hmax where Hmax and Hmin are respectively the maximum and the minimum edge lengths",
        suptitle="Inverse Edge Ratio",
    ),
    VTK_TRIANGLE_METRICS.ASPECT_RATIO: PlotInfo(
        title="Hmax / ( 2.0 * sqrt(3.0) * IR) where Hmax is the maximum edge length and IR is the inradius",
        suptitle="Aspect Ratio",
    ),
    VTK_TRIANGLE_METRICS.INVERSE_ASPECT_RATIO: PlotInfo(
        title="( 2.0 * sqrt(3.0) * IR) / Hmax where Hmax is the maximum edge length and IR is the inradius",
        suptitle="Inverse Aspect Ratio",
    ),
    VTK_TRIANGLE_METRICS.ASPECT_FROBENIUS: PlotInfo(
        title="The Frobenius condition number; i.e. the transformation matrix from an equilateral triangle to a triangle",
        suptitle="Frobenius Aspect",
    ),
    VTK_TRIANGLE_METRICS.INVERSE_FROBENIUS: PlotInfo(
        title="The inverse Frobenius condition number; i.e. the transformation matrix from a triangle to an equilateral triangle",
        suptitle="Inverse Frobenius Aspect",
    ),
}


_VTK_TRIANGLE_METRICS_MAPPING = {
    VTK_TRIANGLE_METRICS.EDGE_RATIO: 0,
    VTK_TRIANGLE_METRICS.INVERSE_EDGE_RATIO: 0,
    VTK_TRIANGLE_METRICS.ASPECT_RATIO: 1,
    VTK_TRIANGLE_METRICS.INVERSE_ASPECT_RATIO: 1,
    VTK_TRIANGLE_METRICS.RADIUS_RATIO: 2,
    VTK_TRIANGLE_METRICS.ASPECT_FROBENIUS: 3,
    VTK_TRIANGLE_METRICS.INVERSE_FROBENIUS: 3,
    # VTK_TRIANGLE_METRICS.MED_ASPECT_FROBENIUS: 4,
    # VTK_TRIANGLE_METRICS.MAX_ASPECT_FROBENIUS: 5,
    VTK_TRIANGLE_METRICS.MIN_ANGLE: 6,
    # VTK_TRIANGLE_METRICS.COLLAPSE_RATIO: 7,
    VTK_TRIANGLE_METRICS.MAX_ANGLE: 8,
    VTK_TRIANGLE_METRICS.CONDITION: 9,
    VTK_TRIANGLE_METRICS.SCALED_JACOBIAN: 10,
    # VTK_TRIANGLE_METRICS.SHEAR: 11,
    VTK_TRIANGLE_METRICS.RELATIVE_SIZE_SQUARED: 12,
    VTK_TRIANGLE_METRICS.SHAPE: 13,
    VTK_TRIANGLE_METRICS.SHAPE_AND_SIZE: 14,
    VTK_TRIANGLE_METRICS.DISTORTION: 15,
    # VTK_TRIANGLE_METRICS.MAX_EDGE_RATIO: 16,
    # VTK_TRIANGLE_METRICS.SKEW: 17,
    # VTK_TRIANGLE_METRICS.TAPER: 18,
    # VTK_TRIANGLE_METRICS.VOLUME: 19,
    # VTK_TRIANGLE_METRICS.STRETCH: 20,
    # VTK_TRIANGLE_METRICS.DIAGONAL: 21,
    # VTK_TRIANGLE_METRICS.DIMENSION: 22,
    # VTK_TRIANGLE_METRICS.ODDY: 23,
    # VTK_TRIANGLE_METRICS.SHEAR_AND_SIZE: 24,
    # VTK_TRIANGLE_METRICS.JACOBIAN: 25,
    # VTK_TRIANGLE_METRICS.WARPAGE: 26,
    # VTK_TRIANGLE_METRICS.ASPECT_GAMMA: 27,
    VTK_TRIANGLE_METRICS.AREA: 28,
    VTK_TRIANGLE_METRICS.EQUIANGLE_SKEW: 29,
    # VTK_TRIANGLE_METRICS.EQUIVOLUME_SKEW: 30,
    # VTK_TRIANGLE_METRICS.MAX_STRETCH: 31,
    # VTK_TRIANGLE_METRICS.MEAN_ASPECT_FROBENIUS: 32,
    # VTK_TRIANGLE_METRICS.MEAN_RATIO: 33,
    # VTK_TRIANGLE_METRICS.NODAL_JACOBIAN_RATIO: 34,
    VTK_TRIANGLE_METRICS.NORMALIZED_INRADIUS: 35,
    # VTK_TRIANGLE_METRICS.SQUISH_INDEX: 36,
}


class VTKCompressionType(enum.IntEnum):
    NONE = 0
    ZLIB = 1
    LZ4 = 2
    LZMA = 3


class VTKDataModeType(enum.IntEnum):
    ASCII = 0
    BINARY = 1
    APPENDED = 2


DEFAULT_COMPRESSION_TYPE = VTKCompressionType.ZLIB
DEFAULT_COMPRESSION_LEVEL = 1
DEFAULT_DATA_MODE = VTKDataModeType.BINARY
ALL_VTK_TRIANGLE_METRICS = tuple(VTK_TRIANGLE_METRICS)


def create_ugrid(
    points: NPArrayF,
    triangles: NPArrayI,
    point_data: dict[str, NPArrayF] | None = None,
    cell_data: dict[str, NPArrayF] | None = None,
    field_data: dict[str, T.Any] | None = None,
):
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))

    # If necessary, add cell type id to the triangles
    if triangles.shape[-1] == 3:
        triangles = np.c_[np.full(len(triangles), 3), triangles]

    vtk_cells = vtkCellArray()
    vtk_cells.SetCells(
        len(triangles),
        numpy_to_vtkIdTypeArray(triangles.flatten(), deep=False),
    )

    ugrid = vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(VTK_TRIANGLE, vtk_cells)

    if point_data is not None:
        for key, array in point_data.items():
            vtk_point_array = numpy_to_vtk(array)
            vtk_point_array.SetName(key)
            _ = ugrid.GetPointData().AddArray(vtk_point_array)

    if cell_data is not None:
        for key, array in cell_data.items():
            vtk_cell_array = numpy_to_vtk(array)
            vtk_cell_array.SetName(key)
            _ = ugrid.GetCellData().AddArray(vtk_cell_array)

    if field_data is not None:
        for key, value in field_data.items():
            ugrid.field_data[key] = value

    return ugrid


def read_vtk(filename: Path) -> vtkUnstructuredGrid:
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    ugrid = T.cast(vtkUnstructuredGrid, reader.GetOutput())
    return ugrid


def write_vtk(ugrid: vtkUnstructuredGrid, filename: Path) -> None:
    writer = vtkUnstructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(ugrid)
    _ = writer.Write()


def read_vtu(filename: Path) -> vtkUnstructuredGrid:
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    ugrid = T.cast(vtkUnstructuredGrid, reader.GetOutput())
    return ugrid


def write_vtu(
    ugrid: vtkUnstructuredGrid,
    filename: Path,
    compression_type: VTKCompressionType = DEFAULT_COMPRESSION_TYPE,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    data_mode: VTKDataModeType = DEFAULT_DATA_MODE,
) -> None:
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(ugrid)
    writer.SetCompressorType(compression_type.value)
    writer.SetCompressionLevel(compression_level)
    writer.SetDataMode(data_mode.value)
    _ = writer.Write()


def read(filename: Path) -> vtkUnstructuredGrid:
    extension = pathlib.Path(filename).suffix
    match extension:
        case ".vtk":
            ugrid = read_vtk(filename)
        case ".vtu":
            ugrid = read_vtu(filename)
        case _:
            raise ValueError("Only `vtk` and `vtu` extensions are supported, not: {extension}")
    return ugrid


def to_vtu(
    input: Path,
    output: Path | None = None,
    compression_type: VTKCompressionType = DEFAULT_COMPRESSION_TYPE,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    data_mode: VTKDataModeType = DEFAULT_DATA_MODE,
) -> None:
    input_path = pathlib.Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output is None:
        output_path = input_path.with_suffix(".vtu")
    else:
        output_path = pathlib.Path(output)

    ugrid = read(input_path)
    write_vtu(ugrid, output_path, compression_type, compression_level, data_mode)


# def clip2(
#     ugrid: vtkUnstructuredGrid,
#     bbox: abc.Sequence[float],
# ) -> vtkUnstructuredGrid:
#     # VTK is using a different ordering compared to what most python GeoSpatial
#     # libraries are using (e.g. shapely). More specifically VTK uses:
#     #       [x_min, x_max, y_min, y_max, z_min, z_max]w
#     # while shapely uses:
#     #       [x_min, y_min, x_max, y_max]
#     match len(bbox):
#         case 4:
#             vtk_bbox = [bbox[0], bbox[2], bbox[1], bbox[3], 0.0, 0.0]
#         case 6:
#             vtk_bbox = bbox
#         case _:
#             raise ValueError("Only sequences of 4 or 6 items are allowed. Please check the docs")
#
#     extractor = vtkExtractUnstructuredGrid()
#     extractor.SetInputData(ugrid)
#     extractor.SetExtent(*vtk_bbox)
#     extractor.Update()
#
#     extracted = extractor.GetOutput()
#     return extracted


def clip(
    ugrid: vtkUnstructuredGrid,
    bbox: abc.Sequence[float],
    *,
    include_boundary: bool = True,
) -> vtkUnstructuredGrid:
    """
    Extract cells from a VTK unstructured grid within a specified bounding box.

    Supports both 2D and 3D bounding boxes with automatic conversion between common GIS
    conventions and VTK's bounding box format. The extraction can include either:
    - Only cells fully contained within the box, or
    - Both contained cells and those intersecting the box boundaries

    Args:
        ugrid (vtkUnstructuredGrid): The input VTK unstructured grid to extract from.
        bbox (Sequence[float]): Bounding box coordinates. Two formats are supported:
            - 4 elements [x_min, y_min, x_max, y_max] (2D, GIS convention)
            - 6 elements [x_min, x_max, y_min, y_max, z_min, z_max] (3D, VTK convention)
        include_boundary (bool, optional): Whether to include cells intersecting the box
            boundaries. Defaults to True.

    Returns:
        vtkUnstructuredGrid: A new unstructured grid containing the extracted cells according
        to the specified boundary inclusion criteria.

    Raises:
        ValueError: If the bbox sequence doesn't have exactly 4 or 6 elements.

    Note:
        - Coordinate ordering: VTK uses [x_min, x_max, y_min, y_max, z_min, z_max] while
          most Python GIS libraries use [x_min, y_min, x_max, y_max]
        - For 2D extractions, Z bounds are automatically set to [0, 0]
        - The include_boundary parameter maps directly to VTK's ExtractBoundaryCells setting
    """
    # Convert input bbox to VTK format [x_min, x_max, y_min, y_max, z_min, z_max]
    match len(bbox):
        case 4:
            # GIS format [x_min, y_min, x_max, y_max] → VTK format with Z=0
            vtk_bbox = [bbox[0], bbox[2], bbox[1], bbox[3], 0, 0]
        case 6:
            # Already in VTK format
            vtk_bbox = list(bbox)
        case _:
            msg = f"Invalid bbox length: {len(bbox)}. Expected 4 (2D) or 6 (3D) elements."
            raise ValueError(msg)

    # Create implicit function for extraction
    box = vtkBox()
    box.SetBounds(vtk_bbox)

    extractor = vtkExtractGeometry()
    extractor.SetInputData(ugrid)
    extractor.SetImplicitFunction(box)
    extractor.ExtractInsideOn()
    extractor.SetExtractBoundaryCells(include_boundary)
    extractor.Update()

    return extractor.GetOutput()


def calc_metric(ugrid: vtkUnstructuredGrid, metric: VTK_TRIANGLE_METRICS) -> NPArrayF:
    """
    Compute the specified metric for all triangular cells in a VTK unstructured grid.

    Parameters:
    -----------
    unstructured_grid : vtkUnstructuredGrid
        Input unstructured grid containing triangular cells.

    Returns:
    --------
    np.ndarray
        Array containing the aspect Frobenius metric for each triangular cell.
    """

    # Create a mesh quality filter
    quality_filter = vtkMeshQuality()
    quality_filter.SetInputData(ugrid)
    quality_filter.SetTriangleQualityMeasure(_VTK_TRIANGLE_METRICS_MAPPING[metric])
    quality_filter.Update()

    output = quality_filter.GetOutput()
    quality_array = output.GetCellData().GetArray("Quality")
    quality_np = T.cast(NPArrayF, vtk_to_numpy(quality_array))  # pyright: ignore[reportUnknownArgumentType]

    return quality_np


def calc_mesh_quality(
    ugrid: vtkUnstructuredGrid,
    metrics: tuple[VTK_TRIANGLE_METRICS, ...] = ALL_VTK_TRIANGLE_METRICS,  # type: ignore[assignment]
    **kwargs: T.Any,
) -> pd.DataFrame:
    import multifutures as mf
    import pandas as pd

    results = mf.multithread(
        func=functools.partial(calc_metric, ugrid=ugrid),
        func_kwargs=[dict(metric=metric) for metric in metrics],
        check=True,
        include_kwargs=True,
        **kwargs,
    )
    data = {r.kwargs["metric"]: r.result for r in results if r.kwargs is not None}
    df = pd.DataFrame(data)[list(metrics)]
    return df.dropna()


def calc_mesh_quality_single(
    ugrid: vtkUnstructuredGrid,
    metrics: tuple[VTK_TRIANGLE_METRICS, ...] = ALL_VTK_TRIANGLE_METRICS,  # type: ignore[assignment]
) -> pd.DataFrame:
    import pandas as pd
    from tqdm.auto import tqdm

    data = {metric: calc_metric(ugrid, metric) for metric in tqdm(metrics)}
    df = pd.DataFrame(data)
    return df.dropna()


def reproject(
    ugrid: vtkUnstructuredGrid,
    from_crs: pyproj.CRS,
    to_crs: pyproj.CRS,
) -> vtkUnstructuredGrid:
    import pyproj

    transform = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True, only_best=True).transform
    # XXX this changes the values in place! Think if we need to create a new object
    ugrid.points = np.c_[transform(*ugrid.points.T, errcheck=True)]  # type: ignore[call-overload]  # pyright: ignore[reportUnknownArgumentType]
    # Save crs as field data
    string_data = vtkStringArray()
    string_data.SetName("crs")
    _ = string_data.InsertNextValue(to_crs.to_wkt())
    ugrid.field_data.AddArray(string_data)
    return ugrid


def gr3_to_vtu(
    filename: Path,
    output: Path,
    variable: str,
    include_boundaries: bool,
) -> None:
    parsed = parse_gr3(filename=filename, include_boundaries=include_boundaries)
    ugrid = create_ugrid(
        points=np.c_[parsed["nodes"][:, :2], np.zeros(len(parsed["nodes"]))],
        triangles=parsed["elements"],
        point_data={variable: parsed["nodes"][:, 2]},
    )
    write_vtu(ugrid=ugrid, filename=output)


def gr3_append_point_data(
    filename: Path,
    output: Path,
    variable: str,
) -> None:
    ugrid = read_vtu(output)
    parsed = parse_gr3(filename=filename, include_boundaries=False)

    vtk_point_array = numpy_to_vtk(parsed["nodes"][:, 2])
    vtk_point_array.SetName(variable)
    _ = ugrid.GetPointData().AddArray(vtk_point_array)

    write_vtu(ugrid, filename=output)


# def ugrid_histogram(
#    ugrid: vtkUnstructuredGrid,
#    metric: VTK_TRIANGLE_METRICS,
#    bins: int | str = "auto",
#    output: Path | None = None,
#    logarithmic: bool = False,
# ) -> None:  # pragma: no cover
#    import matplotlib.pyplot as plt
#
#    plot_info = _PLOT_DATA[VTK_TRIANGLE_METRICS(metric)]
#    if metric.startswith("inverse"):
#        data = 1 / calc_metric(ugrid, metric)
#    else:
#        data = calc_metric(ugrid, metric)
#    data = np.sort(data[~np.isnan(data)])
#    mean = data.mean()
#    fig, ax = plt.subplots()
#    #cumulative = np.cumsum(data)
#    #xdata = np.linspace(data[0], data[-1], len(data))
#    _ = ax.hist(data, bins=bins, log=logarithmic, alpha=0.3, fill=True)#, hatch="xxxx")
#    _ = ax.hist(data, bins=bins, cumulative=True, histtype="step", log=logarithmic)
#    #_ = ax.axvline(mean, color='green', linestyle='dashed', linewidth=1)
#    _ = ax.axvline(mean, color='green')
#    #ax2 = ax.twinx()
#    #_ = ax2.semilogy(np.linspace(data[0], data[-1], len(data)), cumulative, color="red")
#    #ax.axvline(metric, color='m', linestyle='dashed', linewidth=1)
#    fig.suptitle(plot_info.suptitle, size=14, weight="bold")
#    ax.set_title(plot_info.title)
#    ax.grid()
#
#    plt.show()
