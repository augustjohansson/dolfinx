// Copyright (C) 2005-2019 Anders Logg, Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "BoxMesh.h"
#include <cfloat>
#include <cmath>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::generation;

namespace
{
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
create_geom(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
            std::array<std::size_t, 3> n)
{
  // Extract data
  const std::array<double, 3>& p0 = p[0];
  const std::array<double, 3>& p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);
  const double e = z0;
  const double f = z1;
  const double ef = (f - e) / static_cast<double>(nz);

  if (std::abs(x0 - x1) < 2.0 * DBL_EPSILON
      or std::abs(y0 - y1) < 2.0 * DBL_EPSILON
      or std::abs(z0 - z1) < 2.0 * DBL_EPSILON)
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 || ny < 1 || nz < 1)
  {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  xt::xtensor<double, 2> geom(
      {static_cast<std::size_t>(range_p[1] - range_p[0]), 3});
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  std::array<double, 3> point;
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const double z = e + ef * static_cast<double>(iz);
    const double y = c + cd * static_cast<double>(iy);
    const double x = a + ab * static_cast<double>(ix);
    point = {x, y, z};
    for (std::size_t i = 0; i < 3; i++)
      geom(v - range_p[0], i) = point[i];
  }

  return geom;
}
//-----------------------------------------------------------------------------
mesh::Mesh build_tet(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 3> n,
                     const mesh::GhostMode ghost_mode,
                     const mesh::CellPartitionFunction& partitioner)
{
  common::Timer timer("Build BoxMesh");

  xt::xtensor<double, 2> geom = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::size_t cell_range = range_c[1] - range_c[0];
  xt::xtensor<std::int64_t, 2> cells({6 * cell_range, 4});

  // Create tetrahedra
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const int iz = i / (nx * ny);
    const int j = i % (nx * ny);
    const int iy = j / nx;
    const int ix = j % nx;

    const std::int64_t v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    // Note that v0 < v1 < v2 < v3 < vmid
    xt::xtensor_fixed<std::int64_t, xt::xshape<6, 4>> c
        = {{v0, v1, v3, v7}, {v0, v1, v7, v5}, {v0, v5, v7, v4},
           {v0, v3, v2, v7}, {v0, v6, v4, v7}, {v0, v2, v6, v7}};
    std::size_t offset = 6 * (i - range_c[0]);
    xt::view(cells, xt::range(offset, offset + 6), xt::all()) = c;
  }

  fem::CoordinateElement element(mesh::CellType::tetrahedron, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------
mesh::Mesh build_hex(MPI_Comm comm,
                     const std::array<std::array<double, 3>, 2>& p,
                     std::array<std::size_t, 3> n,
                     const mesh::GhostMode ghost_mode,
                     const mesh::CellPartitionFunction& partitioner)
{
  xt::xtensor<double, 2> geom = create_geom(comm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::size_t cell_range = range_c[1] - range_c[0];
  xt::xtensor<std::int64_t, 2> cells({cell_range, 8});

  // Create cuboids
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;

    const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    xt::xtensor_fixed<std::int64_t, xt::xshape<8>> cell
        = {v0, v1, v2, v3, v4, v5, v6, v7};
    xt::view(cells, i - range_c[0], xt::all()) = cell;
  }

  fem::CoordinateElement element(mesh::CellType::hexahedron, 1);
  auto [data, offset] = graph::create_adjacency_data(cells);
  return mesh::create_mesh(
      comm,
      graph::AdjacencyList<std::int64_t>(std::move(data), std::move(offset)),
      element, geom, ghost_mode, partitioner);
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
mesh::Mesh BoxMesh::create(MPI_Comm comm,
                           const std::array<std::array<double, 3>, 2>& p,
                           std::array<std::size_t, 3> n,
                           mesh::CellType celltype,
                           const mesh::GhostMode ghost_mode,
                           const mesh::CellPartitionFunction& partitioner)
{
  switch (celltype)
  {
  case mesh::CellType::tetrahedron:
    return build_tet(comm, p, n, ghost_mode, partitioner);
  case mesh::CellType::hexahedron:
    return build_hex(comm, p, n, ghost_mode, partitioner);
  default:
    throw std::runtime_error("Generate box mesh. Wrong cell type");
  }
}
//-----------------------------------------------------------------------------
