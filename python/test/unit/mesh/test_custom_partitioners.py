# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os

import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx.cpp.mesh import CellType, GhostMode, partition_cells_graph
from dolfinx.io import XDMFFile
from dolfinx_utils.test.fixtures import tempdir
from mpi4py import MPI

assert (tempdir)


@pytest.mark.parametrize("partitioner", [partition_cells_graph])
@pytest.mark.parametrize("Nx", [2, 5, 10])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_partition_box_mesh(partitioner, Nx, cell_type):
    mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, [np.array([0, 0, 0]),
                                            np.array([1, 1, 1])], [Nx, Nx, Nx], cell_type,
                           GhostMode.shared_facet, partitioner)
    tdim = mesh.topology.dim

    c = 6 if cell_type == CellType.tetrahedron else 1
    assert mesh.topology.index_map(tdim).size_global == Nx**3 * c
    assert mesh.topology.index_map(tdim).size_local != 0
    assert mesh.topology.index_map(0).size_global == (Nx + 1)**3


@pytest.mark.parametrize("Nx", [3, 10, 13])
@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron])
def test_custom_partitioner(tempdir, Nx, cell_type):
    mpi_comm = MPI.COMM_WORLD

    Lx = mpi_comm.size
    points = [np.array([0, 0, 0]), np.array([Lx, Lx, Lx])]
    mesh = dolfinx.BoxMesh(
        mpi_comm, points, [Nx, Nx, Nx], cell_type, GhostMode.shared_facet)

    filename = os.path.join(tempdir, "u1_.xdmf")
    with XDMFFile(mpi_comm, filename, "w") as file:
        file.write_mesh(mesh)

    # Read all geometry data on all processes
    with XDMFFile(MPI.COMM_SELF, filename, "r") as file:
        x_global = file.read_geometry_data()

    # Read topology data
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        x = file.read_geometry_data()
        topo = file.read_topology_data()

    num_local_coor = x.shape[0]
    all_sizes = mpi_comm.allgather(num_local_coor)
    all_sizes.insert(0, 0)
    all_ranges = np.cumsum(all_sizes)

    # Testing the premise: coordinates are read contiguously in chunks
    rank = mpi_comm.rank
    assert (np.all(x_global[all_ranges[rank]:all_ranges[rank + 1]] == x))

    cell = ufl.Cell(dolfinx.cpp.mesh.to_string(cell_shape))
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, cell_degree))

    # Partition mesh in layers, capture geometrical data and topological
    # data from outer scope
    def partitioner(*args):
        midpoints = np.mean(x_global[topo], axis=1)
        dest = np.floor(midpoints[:, 0] % mpi_comm.size).astype(np.int32)
        return dolfinx.cpp.graph.AdjacencyList_int32(dest)

    ghost_mode = GhostMode.none
    new_mesh = dolfinx.mesh.create_mesh(mpi_comm, topo, x, domain, ghost_mode, partitioner)
    new_mesh.topology.create_connectivity_all()

    tdim = new_mesh.topology.dim
    assert mesh.topology.index_map(tdim).size_global == new_mesh.topology.index_map(tdim).size_global
    num_cells = new_mesh.topology.index_map(tdim).size_local
    cell_midpoints = dolfinx.cpp.mesh.midpoints(new_mesh, tdim, range(num_cells))
    assert num_cells > 0
    assert np.all(cell_midpoints[:, 0] >= mpi_comm.rank)
    assert np.all(cell_midpoints[:, 0] <= mpi_comm.rank + 1)
