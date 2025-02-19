# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly"""

import math

import dolfinx
import numpy
import scipy.sparse
import pytest
import ufl
from dolfinx import fem
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import create_mesh
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI
from petsc4py import PETSc
from ufl import derivative, ds, dx, inner


def nest_matrix_norm(A):
    """Return norm of a MatNest matrix"""
    assert A.getType() == "nest"
    norm = 0.0
    nrows, ncols = A.getNestSize()
    for row in range(nrows):
        for col in range(ncols):
            A_sub = A.getNestSubMatrix(row, col)
            if A_sub:
                _norm = A_sub.norm()
                norm += _norm * _norm
    return math.sqrt(norm)


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_assemble_functional_dx(mode):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    M = 1.0 * dx(domain=mesh)
    value = dolfinx.fem.assemble_scalar(M)
    value = mesh.mpi_comm().allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(1.0, 1e-12)
    x = ufl.SpatialCoordinate(mesh)
    M = x[0] * dx(domain=mesh)
    value = dolfinx.fem.assemble_scalar(M)
    value = mesh.mpi_comm().allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(0.5, 1e-12)


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_assemble_functional_ds(mode):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    M = 1.0 * ds(domain=mesh)
    value = dolfinx.fem.assemble_scalar(M)
    value = mesh.mpi_comm().allreduce(value, op=MPI.SUM)
    assert value == pytest.approx(4.0, 1e-12)


def test_assemble_derivatives():
    """This test checks the original_coefficient_positions, which may change
    under differentiation (some coefficients and constants are
    eliminated)"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12)
    Q = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u = dolfinx.Function(Q)
    v = ufl.TestFunction(Q)
    du = ufl.TrialFunction(Q)
    b = dolfinx.Function(Q)
    c1 = fem.Constant(mesh, [[1.0, 0.0], [3.0, 4.0]])
    c2 = fem.Constant(mesh, 2.0)

    with b.vector.localForm() as b_local:
        b_local.set(2.0)

    # derivative eliminates 'u' and 'c1'
    L = ufl.inner(c1, c1) * v * dx + c2 * b * inner(u, v) * dx
    a = derivative(L, u, du)

    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()
    a = c2 * b * inner(du, v) * dx
    A2 = dolfinx.fem.assemble_matrix(a)
    A2.assemble()
    assert (A1 - A2).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_basic_assembly(mode):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    f = dolfinx.Function(V)
    with f.vector.localForm() as f_local:
        f_local.set(10.0)
    a = inner(f * u, v) * dx + inner(u, v) * ds
    L = inner(f, v) * dx + inner(2.0, v) * ds

    # Initial assembly
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)

    # Second assembly
    normA = A.norm()
    A.zeroEntries()
    A = dolfinx.fem.assemble_matrix(A, a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    assert normA == pytest.approx(A.norm())
    normb = b.norm()
    with b.localForm() as b_local:
        b_local.set(0.0)
    b = dolfinx.fem.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert isinstance(b, PETSc.Vec)
    assert normb == pytest.approx(b.norm())

    # Vector re-assembly - no zeroing (but need to zero ghost entries)
    with b.localForm() as b_local:
        b_local.array[b.local_size:] = 0.0
    dolfinx.fem.assemble_vector(b, L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    assert 2.0 * normb == pytest.approx(b.norm())

    # Matrix re-assembly (no zeroing)
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()
    assert 2.0 * normA == pytest.approx(A.norm())


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_assembly_bcs(mode):
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(u, v) * dx + inner(u, v) * ds
    L = inner(1.0, v) * dx

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    bdofsV = dolfinx.fem.locate_dofs_geometrical(V, boundary)

    u_bc = dolfinx.fem.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(1.0)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsV)

    # Assemble and apply 'global' lifting of bcs
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    g = b.duplicate()
    with g.localForm() as g_local:
        g_local.set(0.0)
    dolfinx.fem.set_bc(g, [bc])
    f = b - A * g
    dolfinx.fem.set_bc(f, [bc])

    # Assemble vector and apply lifting of bcs during assembly
    b = dolfinx.fem.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_bc = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b_bc, [a], [[bc]])
    b_bc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_bc, [bc])

    assert (f - b_bc).norm() == pytest.approx(0.0, rel=1e-12, abs=1e-12)


@skip_in_parallel
def test_assemble_manifold():
    """Test assembly of poisson problem on a mesh with topological dimension 1
    but embedded in 2D (gdim=2).
    """
    points = numpy.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0],
                          [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]], dtype=numpy.float64)
    cells = numpy.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=numpy.int32)
    cell = ufl.Cell("interval", geometric_dimension=points.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 1

    U = dolfinx.FunctionSpace(mesh, ("P", 1))

    u, v = ufl.TrialFunction(U), ufl.TestFunction(U)
    w = dolfinx.Function(U)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(mesh)
    L = ufl.inner(1.0, v) * ufl.dx(mesh)

    bcdofs = dolfinx.fem.locate_dofs_geometrical(U, lambda x: numpy.isclose(x[0], 0.0))
    bcs = [dolfinx.DirichletBC(w, bcdofs)]
    A = dolfinx.fem.assemble_matrix(a, bcs)
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    dolfinx.fem.set_bc(b, bcs)

    assert numpy.isclose(b.norm(), 0.41231)
    assert numpy.isclose(A.norm(), 25.0199)


@pytest.mark.parametrize("mode",
                         [dolfinx.cpp.mesh.GhostMode.none,
                          dolfinx.cpp.mesh.GhostMode.shared_facet,
                          ])
def test_matrix_assembly_block(mode):
    """Test assembly of block matrices and vectors into (a) monolithic
    blocked structures, PETSc Nest structures, and monolithic structures.
    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 8, ghost_mode=mode)

    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    V0 = dolfinx.fem.FunctionSpace(mesh, P0)
    V1 = dolfinx.fem.FunctionSpace(mesh, P1)

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = dolfinx.mesh.locate_entities_boundary(mesh, facetdim, boundary)

    bdofsV1 = dolfinx.fem.locate_dofs_topological(V1, facetdim, bndry_facets)

    u_bc = dolfinx.fem.Function(V1)
    with u_bc.vector.localForm() as u_local:
        u_local.set(50.0)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsV1)

    # Define variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfinx.Function(V0)

    a00 = inner(u, v) * dx
    a01 = inner(p, v) * dx
    a10 = inner(u, q) * dx
    a11 = inner(p, q) * dx

    L0 = zero * inner(f, v) * dx
    L1 = inner(g, q) * dx

    a_block = [[a00, a01], [a10, a11]]
    L_block = [L0, L1]

    # Monolithic blocked
    A0 = dolfinx.fem.assemble_matrix_block(a_block, [bc])
    A0.assemble()
    b0 = dolfinx.fem.assemble_vector_block(L_block, a_block, [bc])
    assert A0.getType() != "nest"
    Anorm0 = A0.norm()
    bnorm0 = b0.norm()

    # Nested (MatNest)
    A1 = dolfinx.fem.assemble_matrix_nest(a_block, [bc], [["baij", "aij"], ["aij", ""]])
    A1.assemble()
    Anorm1 = nest_matrix_norm(A1)
    assert Anorm0 == pytest.approx(Anorm1, 1.0e-12)

    b1 = dolfinx.fem.assemble_vector_nest(L_block)
    dolfinx.fem.apply_lifting_nest(b1, a_block, [bc])
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(L_block), [bc])
    dolfinx.fem.set_bc_nest(b1, bcs0)
    b1.assemble()

    bnorm1 = math.sqrt(sum([x.norm()**2 for x in b1.getNestSubVecs()]))
    assert bnorm0 == pytest.approx(bnorm1, 1.0e-12)

    # Monolithic version
    E = P0 * P1
    W = dolfinx.fem.FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx + inner(u0, v1) * dx + inner(
        u1, v0) * dx
    L = zero * inner(f, v0) * ufl.dx + inner(g, v1) * dx

    bdofsW_V1 = dolfinx.fem.locate_dofs_topological((W.sub(1), V1), mesh.topology.dim - 1, bndry_facets)
    bc = dolfinx.fem.dirichletbc.DirichletBC(u_bc, bdofsW_V1, W.sub(1))
    A2 = dolfinx.fem.assemble_matrix(a, [bc])
    A2.assemble()
    b2 = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b2, [a], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, [bc])
    assert A2.getType() != "nest"
    assert A2.norm() == pytest.approx(Anorm0, 1.0e-9)
    assert b2.norm() == pytest.approx(bnorm0, 1.0e-9)


@pytest.mark.parametrize("mode", [
    dolfinx.cpp.mesh.GhostMode.none,
    dolfinx.cpp.mesh.GhostMode.shared_facet,
])
def test_assembly_solve_block(mode):
    """Solve a two-field mass-matrix like problem with block matrix approaches
    and test that solution is the same.
    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 31, ghost_mode=mode)
    P = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V0 = dolfinx.fem.FunctionSpace(mesh, P)
    V1 = V0.clone()

    def boundary(x):
        return numpy.logical_or(x[0] < 1.0e-6, x[0] > 1.0 - 1.0e-6)

    # Locate facets on boundary
    facetdim = mesh.topology.dim - 1
    bndry_facets = dolfinx.mesh.locate_entities_boundary(mesh, facetdim, boundary)

    bdofsV0 = dolfinx.fem.locate_dofs_topological(V0, facetdim, bndry_facets)
    bdofsV1 = dolfinx.fem.locate_dofs_topological(V1, facetdim, bndry_facets)

    u_bc0 = dolfinx.fem.Function(V0)
    u_bc0.vector.set(50.0)
    u_bc0.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u_bc1 = dolfinx.fem.Function(V1)
    u_bc1.vector.set(20.0)
    u_bc1.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bcs = [
        dolfinx.fem.dirichletbc.DirichletBC(u_bc0, bdofsV0),
        dolfinx.fem.dirichletbc.DirichletBC(u_bc1, bdofsV1)
    ]

    # Variational problem
    u, p = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v, q = ufl.TestFunction(V0), ufl.TestFunction(V1)
    f = 1.0
    g = -3.0
    zero = dolfinx.Function(V0)

    a00 = inner(u, v) * dx
    a01 = zero * inner(p, v) * dx
    a10 = zero * inner(u, q) * dx
    a11 = inner(p, q) * dx
    L0 = inner(f, v) * dx
    L1 = inner(g, q) * dx

    def monitor(ksp, its, rnorm):
        pass
        # print("Norm:", its, rnorm)

    A0 = dolfinx.fem.assemble_matrix_block([[a00, a01], [a10, a11]], bcs)
    b0 = dolfinx.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
                                           bcs)
    A0.assemble()
    A0norm = A0.norm()
    b0norm = b0.norm()
    x0 = A0.createVecLeft()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setOperators(A0)
    ksp.setMonitor(monitor)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-14)
    ksp.setFromOptions()
    ksp.solve(b0, x0)
    x0norm = x0.norm()

    # Nested (MatNest)
    A1 = dolfinx.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], bcs, diagonal=1.0)
    A1.assemble()
    b1 = dolfinx.fem.assemble_vector_nest([L0, L1])
    dolfinx.fem.apply_lifting_nest(b1, [[a00, a01], [a10, a11]], bcs)
    for b_sub in b1.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form([L0, L1]), bcs)
    dolfinx.fem.set_bc_nest(b1, bcs0)
    b1.assemble()

    b1norm = b1.norm()
    assert b1norm == pytest.approx(b0norm, 1.0e-12)
    A1norm = nest_matrix_norm(A1)
    assert A0norm == pytest.approx(A1norm, 1.0e-12)

    x1 = b1.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A1)
    ksp.setType('cg')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b1, x1)
    x1norm = x1.norm()
    assert x1norm == pytest.approx(x0norm, rel=1.0e-12)

    # Monolithic version
    E = P * P
    W = dolfinx.fem.FunctionSpace(mesh, E)
    u0, u1 = ufl.TrialFunctions(W)
    v0, v1 = ufl.TestFunctions(W)
    a = inner(u0, v0) * dx + inner(u1, v1) * dx
    L = inner(f, v0) * ufl.dx + inner(g, v1) * dx

    u0_bc = dolfinx.fem.Function(V0)
    u0_bc.vector.set(50.0)
    u0_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    u1_bc = dolfinx.fem.Function(V1)
    u1_bc.vector.set(20.0)
    u1_bc.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    bdofsW0_V0 = dolfinx.fem.locate_dofs_topological((W.sub(0), V0), facetdim, bndry_facets)
    bdofsW1_V1 = dolfinx.fem.locate_dofs_topological((W.sub(1), V1), facetdim, bndry_facets)

    bcs = [
        dolfinx.fem.dirichletbc.DirichletBC(u0_bc, bdofsW0_V0, W.sub(0)),
        dolfinx.fem.dirichletbc.DirichletBC(u1_bc, bdofsW1_V1, W.sub(1))
    ]

    A2 = dolfinx.fem.assemble_matrix(a, bcs)
    A2.assemble()
    b2 = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b2, [a], [bcs])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, bcs)
    A2norm = A2.norm()
    b2norm = b2.norm()
    assert A2norm == pytest.approx(A0norm, 1.0e-12)
    assert b2norm == pytest.approx(b0norm, 1.0e-12)

    x2 = b2.copy()
    ksp = PETSc.KSP()
    ksp.create(mesh.mpi_comm())
    ksp.setMonitor(monitor)
    ksp.setOperators(A2)
    ksp.setType('cg')
    ksp.getPC().setType('jacobi')
    ksp.setTolerances(rtol=1.0e-12)
    ksp.setFromOptions()
    ksp.solve(b2, x2)
    x2norm = x2.norm()
    assert x2norm == pytest.approx(x0norm, 1.0e-10)


@pytest.mark.parametrize("mesh", [
    UnitSquareMesh(MPI.COMM_WORLD, 12, 11, ghost_mode=dolfinx.cpp.mesh.GhostMode.none),
    UnitSquareMesh(MPI.COMM_WORLD, 12, 11, ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet),
    UnitCubeMesh(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=dolfinx.cpp.mesh.GhostMode.none),
    UnitCubeMesh(MPI.COMM_WORLD, 3, 7, 3, ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)
])
def test_assembly_solve_taylor_hood(mesh):
    """Assemble Stokes problem with Taylor-Hood elements and solve."""
    P2 = fem.VectorFunctionSpace(mesh, ("Lagrange", 2))
    P1 = fem.FunctionSpace(mesh, ("Lagrange", 1))

    def boundary0(x):
        """Define boundary x = 0"""
        return x[0] < 10 * numpy.finfo(float).eps

    def boundary1(x):
        """Define boundary x = 1"""
        return x[0] > (1.0 - 10 * numpy.finfo(float).eps)

    # Locate facets on boundaries
    facetdim = mesh.topology.dim - 1
    bndry_facets0 = dolfinx.mesh.locate_entities_boundary(mesh, facetdim, boundary0)
    bndry_facets1 = dolfinx.mesh.locate_entities_boundary(mesh, facetdim, boundary1)

    bdofs0 = dolfinx.fem.locate_dofs_topological(P2, facetdim, bndry_facets0)
    bdofs1 = dolfinx.fem.locate_dofs_topological(P2, facetdim, bndry_facets1)

    u0 = dolfinx.Function(P2)
    u0.vector.set(1.0)
    u0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    bc0 = dolfinx.DirichletBC(u0, bdofs0)
    bc1 = dolfinx.DirichletBC(u0, bdofs1)

    u, p = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    a00 = inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(p, ufl.div(v)) * dx
    a10 = ufl.inner(ufl.div(u), q) * dx
    a11 = None

    p00 = a00
    p01, p10 = None, None
    p11 = inner(p, q) * dx

    # FIXME
    # We need zero function for the 'zero' part of L
    p_zero = dolfinx.Function(P1)
    f = dolfinx.Function(P2)
    L0 = ufl.inner(f, v) * dx
    L1 = ufl.inner(p_zero, q) * dx

    def nested_solve():
        """Nested solver"""
        A = dolfinx.fem.assemble_matrix_nest([[a00, a01], [a10, a11]], [bc0, bc1], [["baij", "aij"], ["aij", ""]])
        A.assemble()
        P = dolfinx.fem.assemble_matrix_nest([[p00, p01], [p10, p11]], [bc0, bc1], [["aij", "aij"], ["aij", ""]])
        P.assemble()
        b = dolfinx.fem.assemble_vector_nest([L0, L1])
        dolfinx.fem.apply_lifting_nest(b, [[a00, a01], [a10, a11]], [bc0, bc1])
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        bcs = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form([L0, L1]), [bc0, bc1])
        dolfinx.fem.set_bc_nest(b, bcs)
        b.assemble()

        ksp = PETSc.KSP()
        ksp.create(mesh.mpi_comm())
        ksp.setOperators(A, P)
        nested_IS = P.getNestISs()
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])
        ksp_u, ksp_p = pc.getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType('lu')
        ksp_p.setType("preonly")

        def monitor(ksp, its, rnorm):
            # print("Num it, rnorm:", its, rnorm)
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = b.copy()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), nest_matrix_norm(A), nest_matrix_norm(P)

    def blocked_solve():
        """Blocked (monolithic) solver"""
        A = dolfinx.fem.assemble_matrix_block([[a00, a01], [a10, a11]], [bc0, bc1])
        A.assemble()
        P = dolfinx.fem.assemble_matrix_block([[p00, p01], [p10, p11]], [bc0, bc1])
        P.assemble()
        b = dolfinx.fem.assemble_vector_block([L0, L1], [[a00, a01], [a10, a11]],
                                              [bc0, bc1])

        ksp = PETSc.KSP()
        ksp.create(mesh.mpi_comm())
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType('lu')
        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), A.norm(), P.norm()

    def monolithic_solve():
        """Monolithic (interleaved) solver"""
        P2_el = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        TH = P2_el * P1_el
        W = dolfinx.FunctionSpace(mesh, TH)
        (u, p) = ufl.TrialFunctions(W)
        (v, q) = ufl.TestFunctions(W)
        a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        a01 = ufl.inner(p, ufl.div(v)) * dx
        a10 = ufl.inner(ufl.div(u), q) * dx
        a = a00 + a01 + a10

        p00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        p11 = ufl.inner(p, q) * dx
        p_form = p00 + p11

        f = dolfinx.Function(W.sub(0).collapse())
        p_zero = dolfinx.Function(W.sub(1).collapse())
        L0 = inner(f, v) * dx
        L1 = inner(p_zero, q) * dx
        L = L0 + L1

        bdofsW0_P2_0 = dolfinx.fem.locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets0)
        bdofsW0_P2_1 = dolfinx.fem.locate_dofs_topological((W.sub(0), P2), facetdim, bndry_facets1)

        bc0 = dolfinx.DirichletBC(u0, bdofsW0_P2_0, W.sub(0))
        bc1 = dolfinx.DirichletBC(u0, bdofsW0_P2_1, W.sub(0))

        A = dolfinx.fem.assemble_matrix(a, [bc0, bc1])
        A.assemble()
        P = dolfinx.fem.assemble_matrix(p_form, [bc0, bc1])
        P.assemble()

        b = dolfinx.fem.assemble_vector(L)
        dolfinx.fem.apply_lifting(b, [a], [[bc0, bc1]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, [bc0, bc1])

        ksp = PETSc.KSP()
        ksp.create(mesh.mpi_comm())
        ksp.setOperators(A, P)
        ksp.setType("minres")
        pc = ksp.getPC()
        pc.setType('lu')

        def monitor(ksp, its, rnorm):
            # print("Num it, rnorm:", its, rnorm)
            pass

        ksp.setTolerances(rtol=1.0e-8, max_it=50)
        ksp.setMonitor(monitor)
        ksp.setFromOptions()
        x = A.createVecRight()
        ksp.solve(b, x)
        assert ksp.getConvergedReason() > 0
        return b.norm(), x.norm(), A.norm(), P.norm()

    bnorm0, xnorm0, Anorm0, Pnorm0 = nested_solve()
    bnorm1, xnorm1, Anorm1, Pnorm1 = blocked_solve()
    bnorm2, xnorm2, Anorm2, Pnorm2 = monolithic_solve()

    assert bnorm1 == pytest.approx(bnorm0, 1.0e-12)
    assert xnorm1 == pytest.approx(xnorm0, 1.0e-8)
    assert Anorm1 == pytest.approx(Anorm0, 1.0e-12)
    assert Pnorm1 == pytest.approx(Pnorm0, 1.0e-12)

    assert bnorm2 == pytest.approx(bnorm0, 1.0e-12)
    assert xnorm2 == pytest.approx(xnorm0, 1.0e-8)
    assert Anorm2 == pytest.approx(Anorm0, 1.0e-12)
    assert Pnorm2 == pytest.approx(Pnorm0, 1.0e-12)


def test_basic_interior_facet_assembly():
    mesh = dolfinx.RectangleMesh(MPI.COMM_WORLD,
                                 [numpy.array([0.0, 0.0, 0.0]),
                                  numpy.array([1.0, 1.0, 0.0])],
                                 [5, 5],
                                 cell_type=dolfinx.cpp.mesh.CellType.triangle,
                                 ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet)

    V = fem.FunctionSpace(mesh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = ufl.inner(ufl.avg(u), ufl.avg(v)) * ufl.dS

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    assert isinstance(A, PETSc.Mat)

    L = ufl.conj(ufl.avg(v)) * ufl.dS
    b = dolfinx.fem.assemble_vector(L)
    b.assemble()
    assert isinstance(b, PETSc.Vec)


@pytest.mark.parametrize("mode", [dolfinx.cpp.mesh.GhostMode.none, dolfinx.cpp.mesh.GhostMode.shared_facet])
def test_basic_assembly_constant(mode):
    """Tests assembly with Constant

    The following test should be sensitive to order of flattening the
    matrix-valued constant.

    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5, ghost_mode=mode)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    c = fem.Constant(mesh, [[1.0, 2.0], [5.0, 3.0]])

    a = inner(c[1, 0] * u, v) * dx + inner(c[1, 0] * u, v) * ds
    L = inner(c[1, 0], v) * dx + inner(c[1, 0], v) * ds

    # Initial assembly
    A1 = dolfinx.fem.assemble_matrix(a)
    A1.assemble()

    b1 = dolfinx.fem.assemble_vector(L)
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    c.value = [[1.0, 2.0], [3.0, 4.0]]

    A2 = dolfinx.fem.assemble_matrix(a)
    A2.assemble()

    b2 = dolfinx.fem.assemble_vector(L)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    assert (A1 * 3.0 - A2 * 5.0).norm() == pytest.approx(0.0)
    assert (b1 * 3.0 - b2 * 5.0).norm() == pytest.approx(0.0)


def test_lambda_assembler():
    """Tests assembly with a lambda function

    """
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = inner(u, v) * dx

    # Initial assembly
    a_form = fem.Form(a)

    rdata = []
    cdata = []
    vdata = []

    def mat_insert(rows, cols, vals):
        vdata.append(vals)
        rdata.append(numpy.repeat(rows, len(cols)))
        cdata.append(numpy.tile(cols, len(rows)))
        return 0

    dolfinx.cpp.fem.assemble_matrix(mat_insert, a_form._cpp_object, [])
    vdata = numpy.array(vdata).flatten()
    cdata = numpy.array(cdata).flatten()
    rdata = numpy.array(rdata).flatten()
    mat = scipy.sparse.coo_matrix((vdata, (rdata, cdata)))
    v = numpy.ones(mat.shape[1])
    s = MPI.COMM_WORLD.allreduce(mat.dot(v).sum(), MPI.SUM)
    assert numpy.isclose(s, 1.0)
