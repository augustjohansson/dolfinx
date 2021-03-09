import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace
from dolfinx.cpp.mesh import CellType
import ufl
from ufl import FiniteElement, MixedElement, VectorElement
from ufl import dx, inner, split
import pytest


@pytest.mark.parametrize("element", ["Lagrange", "N1curl"])
def test_splitting(element):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 1, 1, CellType.triangle,
                          dolfinx.cpp.mesh.GhostMode.shared_facet)

    U_el = FiniteElement(element, ufl.triangle, 1)
    U = FunctionSpace(mesh, U_el)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)
    a = inner(u, v) * dx
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()

    result0 = A.convert('dense').getDenseArray()
    dofs0 = list(U.dofmap.cell_dofs(0))

    U_el = MixedElement([FiniteElement(element, ufl.triangle, 1), FiniteElement(element, ufl.triangle, 1)])
    U = FunctionSpace(mesh, U_el)
    u = ufl.TrialFunction(U)
    v = ufl.TestFunction(U)
    a = inner(split(u)[0], split(v)[0]) * dx
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    result1 = A.convert('dense').getDenseArray()
    dofs1 = list(U.dofmap.cell_dofs(0)[:len(dofs0)])

    assert np.allclose(
        [[result0[i, j] for i in dofs0] for j in dofs0],
        [[result1[i, j] for i in dofs1] for j in dofs1])
