# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a ufc_dofmap on a specific mesh.
    """

    def __init__(self, dofmap: cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        """ Returns the Local-global dof map for the cell (using process-local indices)

        Parameters
        ----------
        cell
          The cell index
        """
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def bs(self):
        """ Returns the block size of the dofmap """
        return self._cpp_object.bs

    @property
    def dof_layout(self):
        """ Returns the layout of dofs on an element """
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        """ Returns the index map that described the parallel distribution of the dofmap """
        return self._cpp_object.index_map

    @property
    def index_map_bs(self):
        """ Returns the block size of the index map """
        return self._cpp_object.index_map_bs

    @property
    def list(self):
        """ Returns the adjacency list with dof indices for each cell """
        return self._cpp_object.list()
