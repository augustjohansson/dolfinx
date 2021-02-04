// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "caster_mpi.h"
#include <array>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/generation/BoxMesh.h>
#include <dolfinx/generation/IntervalMesh.h>
#include <dolfinx/generation/RectangleMesh.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace
{
using PythonCellPartitionFunction
    = std::function<const dolfinx::graph::AdjacencyList<std::int32_t>(
        dolfinx_wrappers::MPICommWrapper, int, const dolfinx::mesh::CellType,
        const dolfinx::graph::AdjacencyList<std::int64_t>&,
        dolfinx::mesh::GhostMode)>;

auto create_partitioner_wrapper(const PythonCellPartitionFunction& partitioner)
{
  return [partitioner](MPI_Comm comm, int n,
                       const dolfinx::mesh::CellType cell_type,
                       const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
                       dolfinx::mesh::GhostMode ghost_mode) {
    return partitioner(dolfinx_wrappers::MPICommWrapper(comm), n, cell_type,
                       cells, ghost_mode);
  };
}

} // namespace

namespace dolfinx_wrappers
{

void generation(py::module& m)
{
  m.def(
      "create_interval_mesh",
      [](const MPICommWrapper comm, std::size_t n, std::array<double, 2> p,
         const dolfinx::fem::CoordinateElement& element,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner) {
        return dolfinx::generation::IntervalMesh::create(
            comm.get(), n, p, element, ghost_mode,
            create_partitioner_wrapper(partitioner));
      },
      py::arg("comm"), py::arg("n"), py::arg("p"), py::arg("element"),
      py::arg("ghost_mode"), py::arg("partitioner"));

  m.def(
      "create_rectangle_mesh",
      [](const MPICommWrapper comm, std::array<Eigen::Vector3d, 2> p,
         std::array<std::size_t, 2> n,
         const dolfinx::fem::CoordinateElement& element,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner,
         const std::string& diagonal) {
        return dolfinx::generation::RectangleMesh::create(
            comm.get(), p, n, element, ghost_mode,
            create_partitioner_wrapper(partitioner), diagonal);
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("element"),
      py::arg("ghost_mode"), py::arg("partitioner"), py::arg("diagonal"));

  m.def(
      "create_box_mesh",
      [](const MPICommWrapper comm, std::array<Eigen::Vector3d, 2> p,
         std::array<std::size_t, 3> n,
         const dolfinx::fem::CoordinateElement& element,
         dolfinx::mesh::GhostMode ghost_mode,
         const PythonCellPartitionFunction& partitioner) {
        return dolfinx::generation::BoxMesh::create(
            comm.get(), p, n, element, ghost_mode,
            create_partitioner_wrapper(partitioner));
      },
      py::arg("comm"), py::arg("p"), py::arg("n"), py::arg("element"),
      py::arg("ghost_mode"), py::arg("partitioner"));
}
} // namespace dolfinx_wrappers
