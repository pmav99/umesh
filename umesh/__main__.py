# pyright: reportUnknownMemberType=false
from __future__ import annotations

import pathlib
from typing import Annotated

import cyclopts
import numpy as np
import pandas as pd
import pyproj

from umesh import __version__
from umesh import _api as api  # pyright: ignore[reportPrivateUsage]

app = cyclopts.App(version=__version__, help_format="rich")

MultiParam = cyclopts.Parameter(
    consume_multiple=True,
)


@app.command()
def to_vtu(
    input: pathlib.Path,
    output: pathlib.Path | None = None,
    compression_type: api.VTKCompressionType = api.DEFAULT_COMPRESSION_TYPE,
    compression_level: int = api.DEFAULT_COMPRESSION_LEVEL,
    data_mode: api.VTKDataModeType = api.DEFAULT_DATA_MODE,
) -> None:
    """
    Save an unstructured grid file in VTU format.

    Parameters
    ----------
    input:
        Path to the input file.
    output:
        Path to the output `.vtu` file.
    compression_type:
        Compression algorithm to use.
    compression_level:
        Compression level to use.
    data_mode:
        Data mode to use

    """
    api.to_vtu(input, output, compression_type, compression_level, data_mode)


@app.command()
def stats(
    input: pathlib.Path,
    metrics: Annotated[tuple[api.VTK_TRIANGLE_METRICS, ...], MultiParam] = api.ALL_VTK_TRIANGLE_METRICS,  # type: ignore[assignment]
    quantiles: Annotated[tuple[float, ...], MultiParam] = (0.01, 0.05, 0.25, 0.75, 0.95, 0.99),
    progress_bar: bool = False,
) -> None:  # fmt: skip
    """
    Print mesh quality statistics for an unstructured grid.

    Parameters
    ----------
    input:
        Path to the input unstructured grid file.
    metrics:
        A tuple of mesh quality metrics to calculate. Defaults to all available metrics.
    quantiles:
        A tuple of quantiles to include in the statistics. Defaults to common quantiles.
    progress_bar:
        Whether to display a progress bar during calculations.
    """
    ugrid = api.read(input)
    df = api.calc_mesh_quality(ugrid, metrics, progress_bar=progress_bar)
    for column in df.columns:
        if "inverse" in column:
            df[column] = 1 / df[column]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:_.6f}".format)
    print(df.describe(list(quantiles)))


@app.command()
def reproject(
    input: pathlib.Path,
    output: pathlib.Path,
    from_crs: str,
    to_crs: str,
) -> None:
    """
    Reproject an unstructured grid to a new Coordinate Reference System (CRS).

    Reads an unstructured grid from an input file, reprojects its coordinates
    from a source CRS to a target CRS, and writes the reprojected grid to an
    output file in VTU format.

    Some reminders:

    - Longitude/Latitude is "EPSG:4326"
    - North Polar Stereographic is "EPSG:3995"
    - Geocentric Cartesian is "EPSG:4978"

    Parameters
    ----------
    input : pathlib.Path
        Path to the input unstructured grid file.
    output : pathlib.Path
        Path to the output VTU file for the reprojected grid.
    from_crs : str
        The source CRS of the input grid, specified as a string that
        `pyproj.CRS.from_user_input()` can interpret (e.g., "EPSG:4326",
        "+proj=utm +zone=32 +north").
    to_crs : str
        The target CRS for the reprojection, specified as a string that
        `pyproj.CRS.from_user_input()` can interpret (e.g., "EPSG:4326",
        "+proj=utm +zone=32 +north").

    """
    ugrid = api.read(input)
    src_crs = pyproj.CRS.from_user_input(from_crs)
    tgt_crs = pyproj.CRS.from_user_input(to_crs)
    reprojected = api.reproject(ugrid, src_crs, tgt_crs)
    api.write_vtu(reprojected, output)


@app.command()
def clip(
    input: pathlib.Path,
    bbox: tuple[float, float, float, float],
    output: pathlib.Path,
) -> None:
    """
    Clip mesh to the specified bbox
    """
    ugrid = api.read(input)
    from_crs = pyproj.CRS(4326)
    to_crs = pyproj.CRS.from_user_input(ugrid.field_data["crs"].GetValue(0))
    print(to_crs)
    transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
    bbox = np.r_[transformer.transform(bbox[::2], bbox[1::2], [0, 0])]
    # (x0, x1), (y0, y1), (z0, z1) = transformer.transform(bbox[::2], bbox[1::2], [0, 0])
    # bbox = x0, x1, y0, y1, z0, z1
    print(bbox)
    clipped = api.clip(ugrid, bbox)
    api.write_vtu(clipped, output)


# @app.command()
# def hist(
#    input: pathlib.Path,
#    metric: api.VTK_TRIANGLE_METRICS,
#    bins: int | str = "auto",
#    output: pathlib.Path | None = None,
#    *,
#    logarithmic: bool = True,
# ) -> None:
#    ugrid = api.read(input)
#    api.ugrid_histogram(ugrid, metric, bins, output, logarithmic)
#
# if __name__ == "__main__":
#    app()
