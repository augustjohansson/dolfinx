# Copyright (C) 2020-2021 Joseph P. Dean, Massimiliano Leoni
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx, inner


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical, when passed two function
    spaces, returns the correct degrees of freedom in each space.
    """
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = dolfinx.fem.FunctionSpace(mesh, P0 * P1)
    V = W.sub(0).collapse()

    dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(0), V), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    # Collect dofs (global indices) from all processes
    dofs0_global = W.sub(0).dofmap.index_map.local_to_global(dofs[0])
    dofs1_global = V.dofmap.index_map.local_to_global(dofs[1])
    all_dofs0 = set(np.concatenate(MPI.COMM_WORLD.allgather(dofs0_global)))
    all_dofs1 = set(np.concatenate(MPI.COMM_WORLD.allgather(dofs1_global)))

    # Check only one dof pair is found globally
    assert len(all_dofs0) == 1
    assert len(all_dofs1) == 1

    # On process with the dof pair
    if len(dofs) == 1:
        # Check correct dof returned in W
        coords_W = W.tabulate_dof_coordinates()
        assert np.isclose(coords_W[dofs[0][0]], [0, 0, 0]).all()
        # Check correct dof returned in V
        coords_V = V.tabulate_dof_coordinates()
        assert np.isclose(coords_V[dofs[0][1]], [0, 0, 0]).all()


def xtest_overlapping_bcs():
    """Test that, when boundaries condition overlap, the last provided
    boundary condition is applied.
    """
    n = 123
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, n, n)
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = inner(u, v) * dx
    L = inner(1, v) * dx

    dofsLeft = dolfinx.fem.locate_dofs_geometrical(V, lambda x: x[0] < 1.0 / (2.0 * n))
    dofsTop = dolfinx.fem.locate_dofs_geometrical(V, lambda x: x[1] > 1.0 - 1.0 / (2.0 * n))
    dofCorner = list(set(dofsLeft).intersection(set(dofsTop)))

    # Check only one dof pair is found globally
    # assert len(dofCorner) == 1
    print("******: ", len(dofCorner))

    u0 = dolfinx.Function(V)
    with u0.vector.localForm() as u0_loc:
        u0_loc.set(0)
    u1 = dolfinx.Function(V)
    with u1.vector.localForm() as u1_loc:
        u1_loc.set(123.456)
    bcs = [dolfinx.DirichletBC(u0, dofsLeft), dolfinx.DirichletBC(u1, dofsTop)]

    A = dolfinx.fem.create_matrix(a)
    b = dolfinx.fem.create_vector(L)
    dolfinx.fem.assemble_matrix(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    dolfinx.fem.assemble_vector(b, L)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    if len(dofCorner) > 0:
        with b.localForm() as b_loc:
            assert b[dofCorner[0]] == 123.456
            # b_loc.set(0)
    # assert b[dofCorner[0]] == 123.456
    # assert A.getDiagonal()[dofCorner[0]] == 1
