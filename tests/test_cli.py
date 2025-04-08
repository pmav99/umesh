from __future__ import annotations

import pathlib
import subprocess

import pytest


_VTK_SAMPLE_CONTENTS = """\
# vtk DataFile Version 2.0
Unstructured Grid Example
ASCII
DATASET UNSTRUCTURED_GRID

POINTS 27 float
0 0 0  1 0 0  2 0 0  0 1 0  1 1 0  2 1 0
0 0 1  1 0 1  2 0 1  0 1 1  1 1 1  2 1 1
0 1 2  1 1 2  2 1 2  0 1 3  1 1 3  2 1 3
0 1 4  1 1 4  2 1 4  0 1 5  1 1 5  2 1 5
0 1 6  1 1 6  2 1 6

CELLS 11 60
8 0 1 4 3 6 7 10 9
8 1 2 4 5 7 8 10 11
4 6 10 9 12
4 11 14 10 13
6 15 16 17 14 13 12
6 18 15 19 16 20 17
4 22 23 20 19
3 21 22 18
3 22 19 18
2 26 25
1 24

CELL_TYPES 11
12
11
10
8
7
6
9
5
4
3
1

POINT_DATA 27
SCALARS scalars float 1
LOOKUP_TABLE default
0.0 1.0 2.0 3.0 4.0 5.0
6.0 7.0 8.0 9.0 10.0 11.0
12.0 13.0 14.0 15.0 16.0 17.0
18.0 19.0 20.0 21.0 22.0 23.0
24.0 25.0 26.0

VECTORS vectors float
1 0 0  1 1 0  0 2 0  1 0 0  1 1 0  0 2 0
1 0 0  1 1 0  0 2 0  1 0 0  1 1 0  0 2 0
0 0 1  0 0 1  0 0 1  0 0 1  0 0 1  0 0 1
0 0 1  0 0 1  0 0 1  0 0 1  0 0 1  0 0 1
0 0 1  0 0 1  0 0 1

CELL_DATA 11
SCALARS scalars float 1
LOOKUP_TABLE CellColors
0.0 1.0 2.0 3.0 4.0 5.0
6.0 7.0 8.0 9.0 10.0

LOOKUP_TABLE CellColors 11
.4 .4 1 1
.4 1 .4 1
.4 1 1 1
1 .4 .4 1
1 .4 1 1
1 1 .4 1
1 1 1 1
1 .5 .5 1
.5 1 .5 1
.5 .5 .5 1
1 .5 .4 1
"""


def _run_cmd(subcommand: str, args: list[str | pathlib.Path]) -> subprocess.CompletedProcess[str]:
    """Helper function to run the CLI with given arguments."""
    command = ["umesh", subcommand]
    command.extend(str(arg) for arg in args)
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    return result


@pytest.fixture
def input_vtk(tmp_path: pathlib.Path):
    """Fixture to create a dummy input .vtk file."""
    input_file = tmp_path / "input.vtk"
    with open(input_file, "w") as f:
        _ = f.write(_VTK_SAMPLE_CONTENTS)
    return input_file


def test_cli_to_vtu(input_vtk: pathlib.Path):
    output_vtu = input_vtk.with_suffix(".vtu")
    result = _run_cmd("to-vtu", ["--input", input_vtk, "--output", output_vtu])
    assert result.returncode == 0
    assert output_vtu.exists()


# def test_output_specified(input_vtk: pathlib.Path, tmp_path: pathlib.Path):
#     output_vtu = tmp_path / "output.vtu"
#     result = _run_cmd("to-vtu", [input_vtk, "--output", output_vtu])
#     assert result.returncode == 0
#     assert output_vtu.exists()
#     assert input_vtk.with_suffix(".vtu").exists() is False
#
#
# def test_invalid_input_path(tmp_path: pathlib.Path):
#     """Test running the CLI with a non-existent input file."""
#     invalid_input = tmp_path / "nonexistent.vtk"
#     result = _run_cmd("to-vtu", [invalid_input])
#     assert result.returncode != 0
#
#
# def test_help_message():
#     """Test that the CLI displays the help message."""
#     result_long = _run_cmd("to-vtu", ["--help"])
#     assert result_long.returncode == 0
#     assert "Convert a VTK unstructured grid file (.vtk) to a VTU file (.vtu)." in result_long.stdout
#     assert "input" in result_long.stdout
#     assert "-o" in result_long.stdout
#     assert "--output" in result_long.stdout
#
#     result_short = _run_cmd("to-vtu", ["-h"])
#     assert result_short.returncode == 0
#     assert "Convert a VTK unstructured grid file (.vtk) to a VTU file (.vtu)." in result_short.stdout
#     assert "input" in result_short.stdout
#     assert "-o" in result_long.stdout
#     assert "--output" in result_long.stdout
