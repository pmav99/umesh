# umesh

Some utilities for reading/writing VTK/VTU files

### API

```python
import umesh

# Read vtk/vtu files from disk
ugrid = umesh.read("/path/to/file.vtk")

# Write vtkUnstructuredGrid instances to disk
umesh.write_vtu(ugrid, "/path/to/output.vtu")
```

### CLI

```bash
➜ umesh --help
Usage: umesh COMMAND

╭─ Commands ─────────────────────────────────────────────────────────────────────────────╮
│ clip       Clip mesh to the specified bbox                                             │
│ reproject  Reproject an unstructured grid to a new Coordinate Reference System (CRS).  │
│ stats      Print mesh quality statistics for an unstructured grid.                     │
│ to-vtu     Save an unstructured grid file in VTU format.                               │
│ --help -h  Display this message and exit.                                              │
│ --version  Display application version.                                                │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```
