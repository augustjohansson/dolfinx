# Copyright (C) 2017-2020 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Creation, refining and marking of meshes"""

import types

import numpy
import ufl
from dolfinx import cpp
from dolfinx.cpp.mesh import create_meshtags

__all__ = [
    "locate_entities", "locate_entities_boundary", "refine", "create_mesh", "create_meshtags", "MeshTags"
]


def locate_entities(mesh: cpp.mesh.Mesh,
                    dim: int,
                    marker: types.FunctionType):
    """Compute list of mesh entities satisfying a geometric marking function.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to consider
    marker
        A function that takes an array of points `x` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to `True` for entities to be located.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return cpp.mesh.locate_entities(mesh, dim, marker)


def locate_entities_boundary(mesh: cpp.mesh.Mesh,
                             dim: int,
                             marker: types.FunctionType):
    """Compute list of mesh entities that are attached to an owned boundary facet
    and satisfy a geometric marking function.

    For vertices and edges, in parallel this function will not necessarily
    mark all entities that are on the exterior boundary. For example, it is
    possible for a process to have a vertex that lies on the boundary without
    any of the attached facets being a boundary facet. When used to find
    degrees-of-freedom, e.g. using fem.locate_dofs_topological, the function
    that uses the data returned by this function must typically perform some
    parallel communication.

    Parameters
    ----------
    mesh
        The mesh
    dim
        The topological dimension of the mesh entities to
        consider
    marker
        A function that takes an array of points `x` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to `True` for entities to be located.

    Returns
    -------
    numpy.ndarray
        Indices (local to the process) of marked mesh entities.

    """

    return cpp.mesh.locate_entities_boundary(mesh, dim, marker)


_uflcell_to_dolfinxcell = {
    "interval": cpp.mesh.CellType.interval,
    "triangle": cpp.mesh.CellType.triangle,
    "quadrilateral": cpp.mesh.CellType.quadrilateral,
    "tetrahedron": cpp.mesh.CellType.tetrahedron,
    "hexahedron": cpp.mesh.CellType.hexahedron
}

_meshtags_types = {
    numpy.int8: cpp.mesh.MeshTags_int8,
    numpy.int32: cpp.mesh.MeshTags_int32,
    numpy.int64: cpp.mesh.MeshTags_int64,
    numpy.double: cpp.mesh.MeshTags_double
}


def refine(mesh, cell_markers=None, redistribute=True):
    """Refine a mesh"""
    if cell_markers is None:
        mesh_refined = cpp.refinement.refine(mesh, redistribute)
    else:
        mesh_refined = cpp.refinement.refine(mesh, cell_markers, redistribute)

    domain = mesh._ufl_domain
    domain._ufl_cargo = mesh_refined
    mesh_refined._ufl_domain = domain
    return mesh_refined


def create_mesh(comm, cells, x, domain,
                ghost_mode=cpp.mesh.GhostMode.shared_facet,
                partitioner=cpp.mesh.partition_cells_graph):
    """Create a mesh from topology and geometry data"""
    ufl_element = domain.ufl_coordinate_element()
    cell_shape = ufl_element.cell().cellname()
    cell_degree = ufl_element.degree()
    cmap = cpp.fem.CoordinateElement(_uflcell_to_dolfinxcell[cell_shape], cell_degree)
    try:
        mesh = cpp.mesh.create_mesh(comm, cells, cmap, x, ghost_mode, partitioner)
    except TypeError:
        mesh = cpp.mesh.create_mesh(comm, cpp.graph.AdjacencyList_int64(numpy.cast['int64'](cells)),
                                    cmap, x, ghost_mode, partitioner)

    # Attach UFL data (used when passing a mesh into UFL functions)
    domain._ufl_cargo = mesh
    mesh._ufl_domain = domain
    return mesh


def MeshTags(mesh, dim, indices, values):

    if isinstance(values, int):
        values = numpy.full(indices.shape, values, dtype=numpy.int32)
    elif isinstance(values, float):
        values = numpy.full(indices.shape, values, dtype=numpy.double)

    dtype = values.dtype.type
    if dtype not in _meshtags_types.keys():
        raise KeyError("Datatype {} of values array not recognised".format(dtype))

    fn = _meshtags_types[dtype]
    return fn(mesh, dim, indices.astype(numpy.int32), values)


# Functions to extend cpp.mesh.Mesh with


def ufl_cell(self):
    return ufl.Cell(self.topology.cell_name(), geometric_dimension=self.geometry.dim)


def ufl_domain(self):
    """Return the ufl domain corresponding to the mesh."""
    return self._ufl_domain


# Extend cpp.mesh.Mesh class, and clean-up
cpp.mesh.Mesh.ufl_cell = ufl_cell
cpp.mesh.Mesh.ufl_domain = ufl_domain

del ufl_cell
del ufl_domain
