# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from __future__ import annotations

import collections
import io
import itertools
import os
import typing as T

import numpy as np


def _readline(fd: io.BufferedReader) -> bytes:
    return fd.readline().split(b"=")[0].split(b"!")[0].strip()


# To test this:
#    for path in sorted(pathlib.Path("/path/to/schism_verification_tests/").glob("**/hgrid.gr3")):
#         path
#         _ = parse_gr3(path)
def parse_gr3(
    filename: os.PathLike[str] | str,
    include_boundaries: bool = False,
    sep: str | None = None,
) -> dict[str, T.Any]:
    """
    Parse an *.gr3 file.

    The function is also able to handle fort.14 files, too, (i.e. ADCIRC)
    but the boundary parsing is not keeping all the available information.
    """
    rvalue: dict[str, T.Any] = {}
    with open(filename, "rb") as fd:
        _ = fd.readline()  # skip line
        no_elements, no_points = map(int, fd.readline().strip().split(b"!")[0].split())
        nodes_buffer = io.BytesIO(b"\n".join(itertools.islice(fd, 0, no_points)))
        nodes = np.loadtxt(nodes_buffer, delimiter=sep, usecols=(1, 2, 3))
        elements_buffer = io.BytesIO(b"\n".join(itertools.islice(fd, 0, no_elements)))
        elements = np.loadtxt(elements_buffer, delimiter=sep, usecols=(2, 3, 4), dtype=int)
        elements -= 1  # 0-based index for the nodes
        rvalue["nodes"] = nodes
        rvalue["elements"] = elements
        # boundaries
        if include_boundaries:
            boundaries: dict[str | int, list[T.Any]] = collections.defaultdict(list)
            no_open_boundaries = int(_readline(fd))
            total_open_boundary_nodes = int(_readline(fd))  # noqa: F841
            for _ in range(no_open_boundaries):
                no_nodes_in_boundary = int(_readline(fd))
                boundary_nodes = np.loadtxt(fd, delimiter=sep, usecols=(0,), dtype=int)
                boundaries["open"].append(boundary_nodes - 1)  # 0-based index
            # closed boundaries
            no_closed_boundaries = int(_readline(fd))
            total_closed_boundary_nodes = int(_readline(fd))  # noqa: F841
            for _ in range(no_closed_boundaries):
                # Sometimes it seems that the closed boundaries don't have a "type indicator"
                # For example: Test_COSINE_SFBay/hgrid.gr3
                # In this cases we assume that boundary type is 0 (i.e. land in schism)
                # XXX Maybe check the source code?
                parsed = _readline(fd).split(b" ")
                if len(parsed) == 1:
                    no_nodes_in_boundary = int(parsed[0])
                    boundary_type = 0
                else:
                    no_nodes_in_boundary, boundary_type = map(int, (p for p in parsed if p))
                boundary_nodes = np.genfromtxt(
                    fd,
                    delimiter=sep,
                    usecols=(0,),
                    max_rows=no_nodes_in_boundary,
                    dtype=int,
                )
                boundaries[boundary_type].append(boundary_nodes - 1)  # 0-based-index
            rvalue["boundaries"] = boundaries
    return rvalue
