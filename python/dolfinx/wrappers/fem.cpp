// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <array>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/interpolate.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>
#include <ufc.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace py = pybind11;

namespace dolfinx_wrappers
{
void fem(py::module& m)
{
  // utils
  m.def("create_vector_block", &dolfinx::fem::create_vector_block,
        py::return_value_policy::take_ownership,
        "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def("create_vector_nest", &dolfinx::fem::create_vector_nest,
        py::return_value_policy::take_ownership,
        "Create nested vector for multiple (stacked) linear forms.");

  m.def("create_sparsity_pattern",
        &dolfinx::fem::create_sparsity_pattern<PetscScalar>,
        "Create a sparsity pattern for bilinear form.");
  m.def(
      "pack_coefficients",
      [](dolfinx::fem::Form<PetscScalar>& form) {
        return as_pyarray2d(dolfinx::fem::pack_coefficients(form));
      },
      "Pack coefficients for a Form.");
  m.def(
      "pack_coefficients",
      [](dolfinx::fem::Expression<PetscScalar>& expr) {
        return as_pyarray2d(dolfinx::fem::pack_coefficients(expr));
      },
      "Pack coefficients for an Expression.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Form<PetscScalar>& form) {
        return as_pyarray(dolfinx::fem::pack_constants(form));
      },
      "Pack constants for a Form.");
  m.def(
      "pack_constants",
      [](const dolfinx::fem::Expression<PetscScalar>& expression) {
        return as_pyarray(dolfinx::fem::pack_constants(expression));
      },
      "Pack constants for an Expression.");
  m.def("create_matrix", dolfinx::fem::create_matrix,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block", &dolfinx::fem::create_matrix_block,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest", &dolfinx::fem::create_matrix_nest,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");
  m.def(
      "create_element_dof_layout",
      [](const std::uintptr_t dofmap, const dolfinx::mesh::CellType cell_type,
         const std::vector<int>& parent_map) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_element_dof_layout(*p, cell_type,
                                                       parent_map);
      },
      "Create ElementDofLayout object from a ufc dofmap.");
  m.def(
      "create_dofmap",
      [](const MPICommWrapper comm, const std::uintptr_t dofmap,
         dolfinx::mesh::Topology& topology) {
        const ufc_dofmap* p = reinterpret_cast<const ufc_dofmap*>(dofmap);
        return dolfinx::fem::create_dofmap(comm.get(), *p, topology);
      },
      "Create DofMap object from a pointer to ufc_dofmap.");
  m.def(
      "create_form",
      [](const std::uintptr_t form,
         const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace>>&
             spaces,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Function<PetscScalar>>>& coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::Constant<PetscScalar>>>& constants,
         const std::map<dolfinx::fem::IntegralType,
                        const dolfinx::mesh::MeshTags<int>*>& subdomains,
         const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
        const ufc_form* p = reinterpret_cast<const ufc_form*>(form);
        return dolfinx::fem::create_form<PetscScalar>(
            *p, spaces, coefficients, constants, subdomains, mesh);
      },
      "Create Form from a pointer to ufc_form.");
  m.def(
      "build_dofmap",
      [](const MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& element_dof_layout) {
        auto [map, bs, dofmap] = dolfinx::fem::build_dofmap_data(
            comm.get(), topology, element_dof_layout);
        return std::tuple(map, bs, std::move(dofmap));
      },
      "Build and dofmap on a mesh.");
  m.def("transpose_dofmap", &dolfinx::fem::transpose_dofmap,
        "Build the index to (cell, local index) map from a "
        "dofmap ((cell, local index ) -> index).");

  // dolfinx::fem::FiniteElement
  py::class_<dolfinx::fem::FiniteElement,
             std::shared_ptr<dolfinx::fem::FiniteElement>>(
      m, "FiniteElement", "Finite element object")
      .def(py::init([](const std::uintptr_t ufc_element) {
        const ufc_finite_element* p
            = reinterpret_cast<const ufc_finite_element*>(ufc_element);
        return dolfinx::fem::FiniteElement(*p);
      }))
      .def("num_sub_elements", &dolfinx::fem::FiniteElement::num_sub_elements)
      .def("interpolation_points",
           [](const dolfinx::fem::FiniteElement& self) {
             const xt::xtensor<double, 2>& x = self.interpolation_points();

             // FIXME: Set read-only flag and return wrapper
             return py::array_t<double>(x.shape(), x.data());
             //  return py::array_t<double>(x.shape(), x.data(),
             //  py::cast(self));
           })
      .def_property_readonly("interpolation_ident",
                             &dolfinx::fem::FiniteElement::interpolation_ident)
      .def_property_readonly("value_rank",
                             &dolfinx::fem::FiniteElement::value_rank)
      .def("space_dimension", &dolfinx::fem::FiniteElement::space_dimension)
      .def("value_dimension", &dolfinx::fem::FiniteElement::value_dimension)
      .def("apply_dof_transformation",
           [](const dolfinx::fem::FiniteElement& self,
              py::array_t<double, py::array::c_style>& x,
              std::uint32_t cell_permutation, int dim) {
             self.apply_dof_transformation(
                 xtl::span(x.mutable_data(), x.size()), cell_permutation, dim);
           })
      .def("signature", &dolfinx::fem::FiniteElement::signature);

  // dolfinx::fem::ElementDofLayout
  py::class_<dolfinx::fem::ElementDofLayout,
             std::shared_ptr<dolfinx::fem::ElementDofLayout>>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(py::init<int, const std::vector<std::vector<std::set<int>>>&,
                    const std::vector<int>&,
                    const std::vector<
                        std::shared_ptr<const dolfinx::fem::ElementDofLayout>>,
                    const dolfinx::mesh::CellType>())
      .def_property_readonly("num_dofs",
                             &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("num_entity_dofs", &dolfinx::fem::ElementDofLayout::num_entity_dofs)
      .def("num_entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::num_entity_closure_dofs)
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs)
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs)
      .def("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  py::class_<dolfinx::fem::DofMap, std::shared_ptr<dolfinx::fem::DofMap>>(
      m, "DofMap", "DofMap object")
      .def(py::init<std::shared_ptr<const dolfinx::fem::ElementDofLayout>,
                    std::shared_ptr<const dolfinx::common::IndexMap>, int,
                    dolfinx::graph::AdjacencyList<std::int32_t>&, int>(),
           py::arg("element_dof_layout"), py::arg("index_map"),
           py::arg("index_map_bs"), py::arg("dofmap"), py::arg("bs"))
      .def_readonly("index_map", &dolfinx::fem::DofMap::index_map)
      .def_property_readonly("index_map_bs",
                             &dolfinx::fem::DofMap::index_map_bs)
      .def_readonly("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def("cell_dofs",
           [](const dolfinx::fem::DofMap& self, int cell) {
             xtl::span<const std::int32_t> dofs = self.cell_dofs(cell);
             return py::array_t<std::int32_t>(dofs.size(), dofs.data(),
                                              py::cast(self));
           })
      .def_property_readonly("bs", &dolfinx::fem::DofMap::bs)
      .def("list", &dolfinx::fem::DofMap::list,
           py::return_value_policy::reference_internal);

  // dolfinx::fem::CoordinateElement
  py::class_<dolfinx::fem::CoordinateElement,
             std::shared_ptr<dolfinx::fem::CoordinateElement>>(
      m, "CoordinateElement", "Coordinate map element")
      .def(py::init<dolfinx::mesh::CellType, int>(), py::arg("celltype"),
           py::arg("degree"))
      .def_property_readonly("dof_layout",
                             &dolfinx::fem::CoordinateElement::dof_layout)
      .def("push_forward",
           [](const dolfinx::fem::CoordinateElement& self,
              const py::array_t<double, py::array::c_style>& X,
              const py::array_t<double, py::array::c_style>& cell_geometry) {
             std::array<std::size_t, 2> s_x
                 = {static_cast<std::size_t>(X.shape(0)),
                    static_cast<std::size_t>(X.shape(1))};
             auto _X = xt::adapt(X.data(), X.size(), xt::no_ownership(), s_x);
             std::array<std::size_t, 2> s_g
                 = {static_cast<std::size_t>(cell_geometry.shape(0)),
                    static_cast<std::size_t>(cell_geometry.shape(1))};
             auto g = xt::adapt(cell_geometry.data(), cell_geometry.size(),
                                xt::no_ownership(), s_g);

             xt::xtensor<double, 2> x = xt::empty<double>(
                 {_X.shape(0), std::size_t(cell_geometry.shape(1))});
             const xt::xtensor<double, 2> phi
                 = xt::view(self.tabulate(0, _X), 0, xt::all(), xt::all(), 0);

             self.push_forward(x, g, phi);
             return xt_as_pyarray(std::move(x));
           })
      .def_readwrite("non_affine_atol",
                     &dolfinx::fem::CoordinateElement::non_affine_atol)
      .def_readwrite("non_affine_max_its",
                     &dolfinx::fem::CoordinateElement::non_affine_max_its);

  // dolfinx::fem::DirichletBC
  py::class_<dolfinx::fem::DirichletBC<PetscScalar>,
             std::shared_ptr<dolfinx::fem::DirichletBC<PetscScalar>>>
      dirichletbc(m, "DirichletBC",
                  "Object for representing Dirichlet (essential) boundary "
                  "conditions");

  dirichletbc
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>&
                 g,
             const py::array_t<std::int32_t, py::array::c_style>& dofs) {
            return dolfinx::fem::DirichletBC<PetscScalar>(
                g, std::vector<std::int32_t>(dofs.data(),
                                             dofs.data() + dofs.size()));
          }))
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>&
                 g,
             const std::array<py::array_t<std::int32_t, py::array::c_style>, 2>&
                 V_g_dofs,
             const std::shared_ptr<const dolfinx::fem::FunctionSpace>& V) {
            std::array dofs = {std::vector<std::int32_t>(
                                   V_g_dofs[0].data(),
                                   V_g_dofs[0].data() + V_g_dofs[0].size()),
                               std::vector<std::int32_t>(
                                   V_g_dofs[1].data(),
                                   V_g_dofs[1].data() + V_g_dofs[1].size())};
            return dolfinx::fem::DirichletBC(g, std::move(dofs), V);
          }))
      .def("dof_indices",
           [](const dolfinx::fem::DirichletBC<PetscScalar>& self) {
             auto [dofs, owned] = self.dof_indices();
             return std::pair(py::array_t<std::int32_t>(
                                  dofs.size(), dofs.data(), py::cast(self)),
                              owned);
           })
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::DirichletBC<PetscScalar>::function_space)
      .def_property_readonly("value",
                             &dolfinx::fem::DirichletBC<PetscScalar>::value);

  // dolfinx::fem::assemble
  // Functional
  m.def("assemble_scalar", &dolfinx::fem::assemble_scalar<PetscScalar>,
        "Assemble functional over mesh");
  // Vector
  m.def(
      "assemble_vector",
      [](py::array_t<PetscScalar, py::array::c_style> b,
         const dolfinx::fem::Form<PetscScalar>& L) {
        dolfinx::fem::assemble_vector<PetscScalar>(
            xtl::span(b.mutable_data(), b.size()), L);
      },
      py::arg("b"), py::arg("L"),
      "Assemble linear form into an existing vector");
  // Matrices
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), a, bcs);
        });
  m.def("assemble_matrix_petsc",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<bool>& rows0, const std::vector<bool>& rows1) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), a, rows0,
              rows1);
        });
  m.def("assemble_matrix_petsc_unrolled",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_expand_fn(
                  A, a.function_spaces()[0]->dofmap()->bs(),
                  a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES),
              a, bcs);
        });
  m.def("assemble_matrix_petsc_unrolled",
        [](Mat A, const dolfinx::fem::Form<PetscScalar>& a,
           const std::vector<bool>& rows0, const std::vector<bool>& rows1) {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_expand_fn(
                  A, a.function_spaces()[0]->dofmap()->bs(),
                  a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES),
              a, rows0, rows1);
        });
  m.def("insert_diagonal",
        [](Mat A, const dolfinx::fem::FunctionSpace& V,
           const std::vector<std::shared_ptr<
               const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           PetscScalar diagonal) {
          dolfinx::fem::set_diagonal(
              dolfinx::la::PETScMatrix::set_fn(A, INSERT_VALUES), V, bcs,
              diagonal);
        });
  m.def(
      "assemble_matrix",
      [](const std::function<int(const py::array_t<std::int32_t>&,
                                 const py::array_t<std::int32_t>&,
                                 const py::array_t<PetscScalar>&)>& fin,
         const dolfinx::fem::Form<PetscScalar>& form,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs) {
        std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                          const std::int32_t*, const PetscScalar*)>
            f = [&fin](int nr, const int* rows, int nc, const int* cols,
                       const PetscScalar* data) {
              return fin(py::array(nr, rows), py::array(nc, cols),
                         py::array(nr * nc, data));
            };
        dolfinx::fem::assemble_matrix<PetscScalar>(f, form, bcs);
      },
      "Experimental assembly with Python insertion function. This will be "
      "slow. Testing use only.");

  // BC modifiers
  m.def(
      "apply_lifting",
      [](py::array_t<PetscScalar, py::array::c_style> b,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::Form<PetscScalar>>>& a,
         const std::vector<std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>>& bcs1,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x0,
         double scale) {
        std::vector<xtl::span<const PetscScalar>> _x0;
        for (const auto& x : x0)
          _x0.emplace_back(x.data(), x.size());
        dolfinx::fem::apply_lifting<PetscScalar>(
            xtl::span(b.mutable_data(), b.size()), a, bcs1, _x0, scale);
      },
      "Modify vector for lifted boundary conditions");
  m.def(
      "set_bc",
      [](py::array_t<PetscScalar, py::array::c_style> b,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
         const py::array_t<PetscScalar, py::array::c_style>& x0, double scale) {
        if (x0.ndim() == 0)
        {
          dolfinx::fem::set_bc<PetscScalar>(
              xtl::span(b.mutable_data(), b.size()), bcs, scale);
        }
        else if (x0.ndim() == 1)
        {
          dolfinx::fem::set_bc<PetscScalar>(
              xtl::span(b.mutable_data(), b.size()), bcs,
              xtl::span(x0.data(), x0.shape(0)), scale);
        }
        else
          throw std::runtime_error("Wrong array dimension.");
      },
      py::arg("b"), py::arg("bcs"), py::arg("x0") = py::none(),
      py::arg("scale") = 1.0);
  // Tools
  m.def("bcs_rows", &dolfinx::fem::bcs_rows<PetscScalar>);
  m.def("bcs_cols", &dolfinx::fem::bcs_cols<PetscScalar>);

  m.def(
      "create_discrete_gradient",
      [](const dolfinx::fem::FunctionSpace& V0,
         const dolfinx::fem::FunctionSpace& V1) {
        dolfinx::la::SparsityPattern sp
            = dolfinx::fem::create_sparsity_discrete_gradient(V0, V1);
        Mat A = dolfinx::la::create_petsc_matrix(MPI_COMM_WORLD, sp);
        dolfinx::fem::assemble_discrete_gradient<PetscScalar>(
            dolfinx::la::PETScMatrix::set_fn(A, ADD_VALUES), V0, V1);
        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        return A;
      },
      py::return_value_policy::take_ownership);

  py::enum_<dolfinx::fem::IntegralType>(m, "IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell)
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet)
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet)
      .value("vertex", dolfinx::fem::IntegralType::vertex);

  // dolfinx::fem::Form
  py::class_<dolfinx::fem::Form<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Form<PetscScalar>>>(
      m, "Form", "Variational form object")
      .def(
          py::init(
              [](const std::vector<std::shared_ptr<
                     const dolfinx::fem::FunctionSpace>>& spaces,
                 const std::map<
                     dolfinx::fem::IntegralType,
                     std::pair<std::vector<std::pair<int, py::object>>,
                               const dolfinx::mesh::MeshTags<int>*>>& integrals,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Function<PetscScalar>>>& coefficients,
                 const std::vector<std::shared_ptr<
                     const dolfinx::fem::Constant<PetscScalar>>>& constants,
                 bool needs_permutation_data,
                 const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh) {
                using kern = std::function<void(
                    PetscScalar*, const PetscScalar*, const PetscScalar*,
                    const double*, const int*, const std::uint8_t*,
                    const std::uint32_t)>;
                std::map<dolfinx::fem::IntegralType,
                         std::pair<std::vector<std::pair<int, kern>>,
                                   const dolfinx::mesh::MeshTags<int>*>>
                    _integrals;

                // Loop over kernel for each entity type
                for (auto& kernel_type : integrals)
                {
                  // Set subdomain markers
                  _integrals[kernel_type.first].second = nullptr;

                  // Loop over each domain kernel
                  for (auto& kernel : kernel_type.second.first)
                  {
                    auto tabulate_tensor_ptr
                        = (void (*)(PetscScalar*, const PetscScalar*,
                                    const PetscScalar*, const double*,
                                    const int*, const std::uint8_t*,
                                    const std::uint32_t))
                              kernel.second.cast<std::uintptr_t>();
                    _integrals[kernel_type.first].first.push_back(
                        {kernel.first, tabulate_tensor_ptr});
                  }
                }
                return dolfinx::fem::Form<PetscScalar>(
                    spaces, _integrals, coefficients, constants,
                    needs_permutation_data, mesh);
              }),
          py::arg("spaces"), py::arg("integrals"), py::arg("coefficients"),
          py::arg("constants"), py::arg("need_permutation_data"),
          py::arg("mesh") = py::none())
      .def_property_readonly("coefficients",
                             &dolfinx::fem::Form<PetscScalar>::coefficients)
      .def_property_readonly("rank", &dolfinx::fem::Form<PetscScalar>::rank)
      .def_property_readonly("mesh", &dolfinx::fem::Form<PetscScalar>::mesh)
      .def_property_readonly("function_spaces",
                             &dolfinx::fem::Form<PetscScalar>::function_spaces)
      .def("integral_ids", &dolfinx::fem::Form<PetscScalar>::integral_ids)
      .def("domains", [](const dolfinx::fem::Form<PetscScalar>& self,
                         dolfinx::fem::IntegralType type, int i) {
        const std::vector<std::int32_t>& domains = self.domains(type, i);
        return py::array_t<std::int32_t>(domains.size(), domains.data(),
                                         py::cast(self));
      });
  m.def(
      "locate_dofs_topological",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         const int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote) -> std::array<py::array, 2> {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");
        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_topological(
                {V[0], V[1]}, dim, xtl::span(entities.data(), entities.size()),
                remote);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_topological",
      [](const dolfinx::fem::FunctionSpace& V, const int dim,
         const py::array_t<std::int32_t, py::array::c_style>& entities,
         bool remote) {
        return as_pyarray(dolfinx::fem::locate_dofs_topological(
            V, dim, xtl::span(entities.data(), entities.size()), remote));
      },
      py::arg("V"), py::arg("dim"), py::arg("entities"),
      py::arg("remote") = true);
  m.def(
      "locate_dofs_geometrical",
      [](const std::vector<
             std::reference_wrapper<const dolfinx::fem::FunctionSpace>>& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker) -> std::array<py::array, 2> {
        if (V.size() != 2)
          throw std::runtime_error("Expected two function spaces.");

        auto _marker =
            [&marker](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
          auto strides = x.strides();
          std::transform(strides.begin(), strides.end(), strides.begin(),
                         [](auto s) { return s * sizeof(double); });
          py::array_t _x(x.shape(), strides, x.data(), py::none());
          py::array_t m = marker(_x);
          std::vector<std::size_t> s(m.shape(), m.shape() + m.ndim());
          return xt::adapt(m.data(), m.size(), xt::no_ownership(), s);
        };

        std::array<std::vector<std::int32_t>, 2> dofs
            = dolfinx::fem::locate_dofs_geometrical({V[0], V[1]}, _marker);
        return {as_pyarray(std::move(dofs[0])), as_pyarray(std::move(dofs[1]))};
      },
      py::arg("V"), py::arg("marker"));
  m.def(
      "locate_dofs_geometrical",
      [](const dolfinx::fem::FunctionSpace& V,
         const std::function<py::array_t<bool>(const py::array_t<double>&)>&
             marker) {
        auto _marker =
            [&marker](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
          auto strides = x.strides();
          std::transform(strides.begin(), strides.end(), strides.begin(),
                         [](auto s) { return s * sizeof(double); });
          py::array_t _x(x.shape(), strides, x.data(), py::none());
          py::array_t m = marker(_x);
          std::vector<std::size_t> s(m.shape(), m.shape() + m.ndim());
          return xt::adapt(m.data(), m.size(), xt::no_ownership(), s);
        };
        return as_pyarray(dolfinx::fem::locate_dofs_geometrical(V, _marker));
      },
      py::arg("V"), py::arg("marker"));

  // dolfinx::fem::Function
  py::class_<dolfinx::fem::Function<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>(
      m, "Function", "A finite element function")
      .def(py::init<std::shared_ptr<const dolfinx::fem::FunctionSpace>>(),
           "Create a function on the given function space")
      .def(py::init<std::shared_ptr<dolfinx::fem::FunctionSpace>,
                    std::shared_ptr<dolfinx::la::Vector<PetscScalar>>>())
      .def_readwrite("name", &dolfinx::fem::Function<PetscScalar>::name)
      .def_property_readonly("id", &dolfinx::fem::Function<PetscScalar>::id)
      .def("sub", &dolfinx::fem::Function<PetscScalar>::sub,
           "Return sub-function (view into parent Function")
      .def("collapse", &dolfinx::fem::Function<PetscScalar>::collapse,
           "Collapse sub-function view")
      .def(
          "interpolate",
          [](dolfinx::fem::Function<PetscScalar>& self,
             const std::function<py::array_t<PetscScalar>(
                 const py::array_t<double>&)>& f) {
            auto _f = [&f](const xt::xtensor<double, 2>& x)
                -> xt::xarray<PetscScalar> {
              auto strides = x.strides();
              std::transform(strides.begin(), strides.end(), strides.begin(),
                             [](auto s) { return s * sizeof(double); });
              py::array_t _x(x.shape(), strides, x.data(), py::none());
              py::array_t v = f(_x);
              std::vector<std::size_t> shape;
              for (pybind11::ssize_t i = 0; i < v.ndim(); i++)
                shape.push_back(v.shape()[i]);
              return xt::adapt(v.data(), shape);
            };
            self.interpolate(_f);
          },
          py::arg("f"), "Interpolate an expression")
      .def("interpolate",
           py::overload_cast<const dolfinx::fem::Function<PetscScalar>&>(
               &dolfinx::fem::Function<PetscScalar>::interpolate),
           py::arg("u"), "Interpolate a finite element function")
      .def(
          "interpolate_ptr",
          [](dolfinx::fem::Function<PetscScalar>& self, std::uintptr_t addr) {
            const std::function<void(PetscScalar*, int, int, const double*)> f
                = reinterpret_cast<void (*)(PetscScalar*, int, int,
                                            const double*)>(addr);

            auto _f = [&f](xt::xarray<PetscScalar>& values,
                           const xt::xtensor<double, 2>& x) -> void {
              f(values.data(), int(values.shape(1)), int(values.shape(0)),
                x.data());
            };

            assert(self.function_space());
            assert(self.function_space()->element());
            assert(self.function_space()->mesh());
            const int tdim = self.function_space()->mesh()->topology().dim();
            auto cell_map
                = self.function_space()->mesh()->topology().index_map(tdim);
            assert(cell_map);
            const std::int32_t num_cells
                = cell_map->size_local() + cell_map->num_ghosts();
            std::vector<std::int32_t> cells(num_cells, 0);
            std::iota(cells.begin(), cells.end(), 0);
            const auto x = dolfinx::fem::interpolation_coords(
                *self.function_space()->element(),
                *self.function_space()->mesh(), cells);

            dolfinx::fem::interpolate_c<PetscScalar>(self, _f, x, cells);
          },
          "Interpolate using a pointer to an expression with a C "
          "signature")
      .def_property_readonly("vector",
                             &dolfinx::fem::Function<PetscScalar>::vector,
                             "Return the PETSc vector associated with "
                             "the finite element Function")
      .def_property_readonly(
          "x", py::overload_cast<>(&dolfinx::fem::Function<PetscScalar>::x),
          "Return the vector associated with the finite element Function")
      .def(
          "eval",
          [](const dolfinx::fem::Function<PetscScalar>& self,
             const py::array_t<double, py::array::c_style>& x,
             const py::array_t<std::int32_t, py::array::c_style>& cells,
             py::array_t<PetscScalar, py::array::c_style>& u) {
            // TODO: handle 1d case

            std::array<std::size_t, 2> shape_x
                = {static_cast<std::size_t>(x.shape(0)),
                   static_cast<std::size_t>(x.shape(1))};
            auto _x
                = xt::adapt(x.data(), x.size(), xt::no_ownership(), shape_x);

            std::array<std::size_t, 2> shape_u
                = {static_cast<std::size_t>(u.shape(0)),
                   static_cast<std::size_t>(u.shape(1))};
            xt::xtensor<PetscScalar, 2> _u = xt::adapt(
                u.mutable_data(), u.size(), xt::no_ownership(), shape_u);
            self.eval(_x, xtl::span(cells.data(), cells.size()), _u);
            std::copy_n(_u.data(), _u.size(), u.mutable_data());
          },
          py::arg("x"), py::arg("cells"), py::arg("values"),
          "Evaluate Function")
      .def(
          "compute_point_values",
          [](const dolfinx::fem::Function<PetscScalar>& self) {
            return xt_as_pyarray(self.compute_point_values());
          },
          "Compute values at all mesh points")
      .def_property_readonly(
          "function_space",
          &dolfinx::fem::Function<PetscScalar>::function_space);

  // dolfinx::fem::FunctionSpace
  py::class_<dolfinx::fem::FunctionSpace,
             std::shared_ptr<dolfinx::fem::FunctionSpace>>(m, "FunctionSpace")
      .def(py::init<std::shared_ptr<dolfinx::mesh::Mesh>,
                    std::shared_ptr<dolfinx::fem::FiniteElement>,
                    std::shared_ptr<dolfinx::fem::DofMap>>())
      .def_property_readonly("id", &dolfinx::fem::FunctionSpace::id)
      .def("__hash__", &dolfinx::fem::FunctionSpace::id)
      .def("__eq__", &dolfinx::fem::FunctionSpace::operator==)
      .def("collapse", &dolfinx::fem::FunctionSpace::collapse)
      .def("component", &dolfinx::fem::FunctionSpace::component)
      .def("contains", &dolfinx::fem::FunctionSpace::contains)
      .def_property_readonly("element", &dolfinx::fem::FunctionSpace::element)
      .def_property_readonly("mesh", &dolfinx::fem::FunctionSpace::mesh)
      .def_property_readonly("dofmap", &dolfinx::fem::FunctionSpace::dofmap)
      .def("sub", &dolfinx::fem::FunctionSpace::sub)
      .def("tabulate_dof_coordinates",
           [](const dolfinx::fem::FunctionSpace& self) {
             return xt_as_pyarray(self.tabulate_dof_coordinates(false));
           });

  // dolfinx::fem::Constant
  py::class_<dolfinx::fem::Constant<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Constant<PetscScalar>>>(
      m, "Constant", "A value constant with respect to integration domain")
      .def(py::init<std::vector<int>, std::vector<PetscScalar>>(),
           "Create a constant from a scalar value array")
      .def(
          "value",
          [](dolfinx::fem::Constant<PetscScalar>& self) {
            return py::array(self.shape, self.value.data(), py::none());
          },
          py::return_value_policy::reference_internal);

  // dolfinx::fem::Expression
  py::class_<dolfinx::fem::Expression<PetscScalar>,
             std::shared_ptr<dolfinx::fem::Expression<PetscScalar>>>(
      m, "Expression", "An Expression")
      .def(py::init(
               [](const std::vector<std::shared_ptr<
                      const dolfinx::fem::Function<PetscScalar>>>& coefficients,
                  const std::vector<std::shared_ptr<
                      const dolfinx::fem::Constant<PetscScalar>>>& constants,
                  const std::shared_ptr<const dolfinx::mesh::Mesh>& mesh,
                  const py::array_t<double, py::array::c_style>& X,
                  py::object addr, const std::size_t value_size) {
                 auto tabulate_expression_ptr = (void (*)(
                     PetscScalar*, const PetscScalar*, const PetscScalar*,
                     const double*))addr.cast<std::uintptr_t>();
                 dolfinx::array2d<double> _X(X.shape()[0], X.shape()[1]);
                 std::copy_n(X.data(), X.size(), _X.data());
                 return dolfinx::fem::Expression<PetscScalar>(
                     coefficients, constants, mesh, _X, tabulate_expression_ptr,
                     value_size);
               }),
           py::arg("coefficients"), py::arg("constants"), py::arg("mesh"),
           py::arg("x"), py::arg("fn"), py::arg("value_size"))
      .def("eval",
           [](const dolfinx::fem::Expression<PetscScalar>& self,
              const py::array_t<std::int32_t, py::array::c_style>& active_cells,
              py::array_t<PetscScalar> values) {
             dolfinx::array2d<PetscScalar> _values(active_cells.shape()[0],
                                                   self.num_points()
                                                       * self.value_size());
             self.eval(xtl::span(active_cells.data(), active_cells.size()),
                       _values);
             assert(values.ndim() == 2);
             assert(values.shape()[0] == (py::ssize_t)_values.shape[0]);
             assert(values.shape()[1] == (py::ssize_t)_values.shape[1]);
             auto v = values.mutable_unchecked();
             for (py::ssize_t i = 0; i < v.shape(0); i++)
               for (py::ssize_t j = 0; j < v.shape(1); j++)
                 v(i, j) = _values(i, j);
           })
      .def_property_readonly("mesh",
                             &dolfinx::fem::Expression<PetscScalar>::mesh,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("num_points",
                             &dolfinx::fem::Expression<PetscScalar>::num_points,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("value_size",
                             &dolfinx::fem::Expression<PetscScalar>::value_size,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("x", &dolfinx::fem::Expression<PetscScalar>::x,
                             py::return_value_policy::reference_internal);
}
} // namespace dolfinx_wrappers
