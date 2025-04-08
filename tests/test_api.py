# pyright: basic
from __future__ import annotations

import pathlib

import numpy as np
import pyproj
import pytest
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

from umesh import calc_mesh_quality
from umesh import clip
from umesh import read
from umesh import read_vtk
from umesh import read_vtu
from umesh import reproject
from umesh import to_vtk_unstructured_grid
from umesh import to_vtu
from umesh import write_vtk
from umesh import write_vtu
from umesh._api import calc_mesh_quality_single


ROOT = pathlib.Path(__file__).parent.parent.resolve()
TESTS = ROOT / "tests"
DATA = TESTS / "data"
VTK_MESH = DATA / "mesh.vtk"
VTU_MESH = DATA / "mesh.vtu"


EXPECTED_VTK = """\
# vtk DataFile Version 5.1
vtk output
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0 1 0 0 1 1 0
0 1 0
CELLS 3 6
OFFSETS vtktypeint64
0 3 6
CONNECTIVITY vtktypeint64
0 1 2 2 3 0
CELL_TYPES 2
5
5
"""

EXPECTED_VTU = """\
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
<UnstructuredGrid>
<Piece NumberOfPoints="4" NumberOfCells="2">
<PointData>
</PointData>
<CellData>
</CellData>
<Points>
<DataArray type="Float32" Name="Points" NumberOfComponents="3" format="binary" RangeMin="0" RangeMax="1.4142135623730951">
AQAAAACAAAAwAAAAFQAAAA==eAFjYEAGDfYIHoiNzmdgAAA65AL9
<InformationKey name="L2_NORM_RANGE" location="vtkDataArray" length="2">
<Value index="0">
0
</Value>
<Value index="1">
1.4142135624
</Value>
</InformationKey>
</DataArray>
</Points>
<Cells>
<DataArray type="Int64" Name="connectivity" format="binary" RangeMin="0" RangeMax="3">
AQAAAACAAAAwAAAAFQAAAA==eAFjYIAARijNhEYzQ/kwCgAA+AAJ
</DataArray>
<DataArray type="Int64" Name="offsets" format="binary" RangeMin="3" RangeMax="6">
AQAAAACAAAAQAAAADgAAAA==eAFjZoAANigNAABwAAo=
</DataArray>
<DataArray type="UInt8" Name="types" format="binary" RangeMin="5" RangeMax="5">
AQAAAACAAAACAAAACgAAAA==eAFjZQUAABEACw==
</DataArray>
</Cells>
</Piece>
</UnstructuredGrid>
</VTKFile>
""".strip()

# Define points and triangles
points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
triangles = np.array(
    [
        [3, 0, 1, 2],  # Triangle with vertices at indices 0, 1, and 2
        [3, 2, 3, 0],
    ],
)


# TEST_PARAMS = pytest.mark.parametrize(
#     "func,output,expected",
#     [
#         pytest.param(to_vtk, "output.vtk", EXPECTED_VTK, id="vtk"),
#         pytest.param(to_vtu, "output.vtu", EXPECTED_VTU, id="vtu"),
#     ],
# )


@pytest.mark.parametrize(
    "triangles",
    [
        pytest.param(triangles, id="triangles_with_type_id"),
        pytest.param(triangles[:, 1:], id="triangles_without_type_id"),
    ],
)
def test_to_vtk_triangles_without_type_id(triangles):
    ugrid = to_vtk_unstructured_grid(points, triangles)
    assert isinstance(ugrid, vtkUnstructuredGrid)
    assert len(ugrid.points) == 4
    assert len(ugrid.cells["cell_types"]) == 2
    assert (ugrid.cells["cell_types"] == 5).all()


def test_read_vtk():
    ugrid = read_vtk(VTK_MESH)
    assert isinstance(ugrid, vtkUnstructuredGrid)
    assert len(ugrid.points) == 146
    assert len(ugrid.cells["connectivity"]) == 828
    assert len(ugrid.cells["offsets"]) == 293
    assert len(ugrid.cells["cell_types"]) == 292


def test_read_vtu():
    ugrid = read_vtu(VTU_MESH)
    assert isinstance(ugrid, vtkUnstructuredGrid)
    assert len(ugrid.points) == 146
    assert len(ugrid.cells["connectivity"]) == 828
    assert len(ugrid.cells["offsets"]) == 293
    assert len(ugrid.cells["cell_types"]) == 292


@pytest.mark.parametrize("input_file", ["tests/data/mesh.vtk", "tests/data/mesh.vtu"])
def test_read(input_file):
    ugrid = read(input_file)
    assert isinstance(ugrid, vtkUnstructuredGrid)
    assert len(ugrid.points) == 146
    assert len(ugrid.cells["connectivity"]) == 828
    assert len(ugrid.cells["offsets"]) == 293
    assert len(ugrid.cells["cell_types"]) == 292


def test_read_write_vtk(tmp_path: pathlib.Path):
    original_ugrid = read_vtk(VTK_MESH)
    output_path = tmp_path / "out.vtk"
    write_vtk(original_ugrid, output_path)
    read_ugrid = read_vtk(output_path)
    assert len(original_ugrid.points) == len(read_ugrid.points)
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["offsets"]) == len(read_ugrid.cells["offsets"])
    assert len(original_ugrid.cells["cell_types"]) == len(read_ugrid.cells["cell_types"])


def test_read_write_vtu(tmp_path: pathlib.Path):
    original_ugrid = read_vtu(VTU_MESH)
    output_path = tmp_path / "out.vtu"
    write_vtu(original_ugrid, output_path)
    read_ugrid = read_vtu(output_path)
    assert len(original_ugrid.points) == len(read_ugrid.points)
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["offsets"]) == len(read_ugrid.cells["offsets"])
    assert len(original_ugrid.cells["cell_types"]) == len(read_ugrid.cells["cell_types"])


def test_to_vtu(tmp_path: pathlib.Path):
    original_ugrid = read_vtk(VTK_MESH)
    output_path = tmp_path / "out.vtu"
    to_vtu(VTK_MESH, output_path)
    read_ugrid = read_vtu(output_path)
    assert len(original_ugrid.points) == len(read_ugrid.points)
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["connectivity"]) == len(read_ugrid.cells["connectivity"])
    assert len(original_ugrid.cells["offsets"]) == len(read_ugrid.cells["offsets"])


def test_clip():
    original = read_vtu(VTU_MESH)
    assert original.bounds == pytest.approx((0, 1, 0, 1, 0, 0))
    extracted = clip(original, [0.5, 0.5, 1.1, 1.1])
    assert extracted.bounds == pytest.approx((0.4, 1.0, 0.4, 1.0, 0, 0), abs=0.1)


def test_clip_vtk_conventions_for_bbox():
    original = read_vtu(VTU_MESH)
    assert original.bounds == pytest.approx((0, 1, 0, 1, 0, 0))
    extracted = clip(original, [0.5, 1.1, 0.5, 1.1, -1, 1])
    assert extracted.bounds == pytest.approx((0.4, 1.0, 0.4, 1.0, 0, 0), abs=0.1)


def test_mesh_quality():
    ugrid = read_vtk(VTK_MESH)
    df_single = calc_mesh_quality_single(ugrid)
    df_multi = calc_mesh_quality(ugrid)
    assert df_single.equals(df_multi)


def test_reproject_roundtrip():
    original = read_vtk(VTK_MESH)
    intermediate = reproject(original, pyproj.CRS(4326), pyproj.CRS(3857))
    final = reproject(intermediate, pyproj.CRS(3857), pyproj.CRS(4326))
    assert np.array_equal(original.points, final.points)
    assert np.array_equal(original.cells["connectivity"], final.cells["connectivity"])
    assert np.array_equal(original.cells["offsets"], final.cells["offsets"])
    assert np.array_equal(original.cells["cell_types"], final.cells["cell_types"])
