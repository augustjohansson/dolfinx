// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cell_types.h"
#include "Geometry.h"
#include <algorithm>
#include <basix/cell.h>
#include <cfloat>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xfixed.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
std::string mesh::to_string(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return "point";
  case mesh::CellType::interval:
    return "interval";
  case mesh::CellType::triangle:
    return "triangle";
  case mesh::CellType::tetrahedron:
    return "tetrahedron";
  case mesh::CellType::quadrilateral:
    return "quadrilateral";
  case mesh::CellType::pyramid:
    return "pyramid";
  case mesh::CellType::prism:
    return "prism";
  case mesh::CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
    return std::string();
  }
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::to_type(const std::string& cell)
{
  if (cell == "point")
    return mesh::CellType::point;
  else if (cell == "interval")
    return mesh::CellType::interval;
  else if (cell == "triangle")
    return mesh::CellType::triangle;
  else if (cell == "tetrahedron")
    return mesh::CellType::tetrahedron;
  else if (cell == "quadrilateral")
    return mesh::CellType::quadrilateral;
  else if (cell == "pyramid")
    return mesh::CellType::pyramid;
  else if (cell == "prism")
    return mesh::CellType::prism;
  else if (cell == "hexahedron")
    return mesh::CellType::hexahedron;
  else
    throw std::runtime_error("Unknown cell type (" + cell + ")");
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_entity_type(mesh::CellType type, int d)
{
  const int dim = mesh::cell_dim(type);
  if (d == dim)
    return type;
  else if (d == 1)
    return CellType::interval;
  else if (d == (dim - 1))
    return mesh::cell_facet_type(type);
  else
    return CellType::point;
}
//-----------------------------------------------------------------------------
mesh::CellType mesh::cell_facet_type(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return mesh::CellType::point;
  case mesh::CellType::interval:
    return mesh::CellType::point;
  case mesh::CellType::triangle:
    return mesh::CellType::interval;
  case mesh::CellType::tetrahedron:
    return mesh::CellType::triangle;
  case mesh::CellType::quadrilateral:
    return mesh::CellType::interval;
  case mesh::CellType::pyramid:
    throw std::runtime_error("TODO: pyramid");
  case mesh::CellType::prism:
    throw std::runtime_error("TODO: prism");
  case mesh::CellType::hexahedron:
    return mesh::CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
    return mesh::CellType::point;
  }
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<int> mesh::get_entity_vertices(mesh::CellType type,
                                                    int dim)
{
  const std::vector<std::vector<int>> topology
      = basix::cell::topology(basix::cell::str_to_type(to_string(type)))[dim];

  return graph::AdjacencyList<int>(topology);
}
//-----------------------------------------------------------------------------
xt::xtensor<int, 2> mesh::get_sub_entities(CellType type, int dim0, int dim1)
{
  if (dim0 != 2)
  {
    throw std::runtime_error(
        "mesh::get_sub_entities supports faces (d=2) only at present.");
  }
  if (dim1 != 1)
  {
    throw std::runtime_error(
        "mesh::get_sub_entities supports getting edges (d=1) at present.");
  }

  // TODO: get this data from basix
  static const xt::xtensor_fixed<int, xt::xshape<1, 3>> triangle = {{0, 1, 2}};
  static const xt::xtensor_fixed<int, xt::xshape<1, 4>> quadrilateral
      = {{0, 1, 2, 3}};
  static const xt::xtensor_fixed<int, xt::xshape<4, 3>> tetrahedron
      = {{0, 1, 2}, {0, 3, 4}, {1, 3, 5}, {2, 4, 5}};
  static const xt::xtensor_fixed<int, xt::xshape<6, 4>> hexahedron
      = {{0, 1, 3, 5},  {0, 2, 4, 8},  {1, 2, 6, 9},
         {3, 4, 7, 10}, {5, 6, 7, 11}, {8, 9, 10, 11}};
  switch (type)
  {
  case mesh::CellType::interval:
    return xt::empty<int>({0, 0});
  case mesh::CellType::point:
    return xt::empty<int>({0, 0});
  case mesh::CellType::triangle:
    return triangle;
  case mesh::CellType::tetrahedron:
    return tetrahedron;
  case mesh::CellType::quadrilateral:
    return quadrilateral;
  case mesh::CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unsupported cell type.");
  }
}
//-----------------------------------------------------------------------------
int mesh::cell_dim(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 0;
  case mesh::CellType::interval:
    return 1;
  case mesh::CellType::triangle:
    return 2;
  case mesh::CellType::tetrahedron:
    return 3;
  case mesh::CellType::quadrilateral:
    return 2;
  case mesh::CellType::pyramid:
    return 3;
  case mesh::CellType::prism:
    return 3;
  case mesh::CellType::hexahedron:
    return 3;
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
int mesh::cell_num_entities(mesh::CellType type, int dim)
{
  assert(dim <= 3);
  constexpr std::array<int, 4> point = {1, 0, 0, 0};
  constexpr std::array<int, 4> interval = {2, 1, 0, 0};
  constexpr std::array<int, 4> triangle = {3, 3, 1, 0};
  constexpr std::array<int, 4> quadrilateral = {4, 4, 1, 0};
  constexpr std::array<int, 4> tetrahedron = {4, 6, 4, 1};
  constexpr std::array<int, 4> pyramid = {5, 8, 5, 1};
  constexpr std::array<int, 4> prism = {6, 9, 5, 1};
  constexpr std::array<int, 4> hexahedron = {8, 12, 6, 1};
  switch (type)
  {
  case mesh::CellType::point:
    return point[dim];
  case mesh::CellType::interval:
    return interval[dim];
  case mesh::CellType::triangle:
    return triangle[dim];
  case mesh::CellType::tetrahedron:
    return tetrahedron[dim];
  case mesh::CellType::quadrilateral:
    return quadrilateral[dim];
  case mesh::CellType::pyramid:
    return pyramid[dim];
  case mesh::CellType::prism:
    return prism[dim];
  case mesh::CellType::hexahedron:
    return hexahedron[dim];
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
bool mesh::is_simplex(mesh::CellType type)
{
  return static_cast<int>(type) > 0;
}
//-----------------------------------------------------------------------------
int mesh::num_cell_vertices(mesh::CellType type)
{
  return std::abs(static_cast<int>(type));
}
//-----------------------------------------------------------------------------
std::map<std::array<int, 2>, std::vector<std::set<int>>>
mesh::cell_entity_closure(mesh::CellType cell_type)
{
  const int cell_dim = mesh::cell_dim(cell_type);
  std::array<int, 4> num_entities;
  for (int i = 0; i <= cell_dim; ++i)
    num_entities[i] = mesh::cell_num_entities(cell_type, i);

  const graph::AdjacencyList<int> edge_v
      = mesh::get_entity_vertices(cell_type, 1);
  const auto face_e = mesh::get_sub_entities(cell_type, 2, 1);

  std::map<std::array<int, 2>, std::vector<std::set<int>>> entity_closure;
  for (int dim = 0; dim <= cell_dim; ++dim)
  {
    for (int entity = 0; entity < num_entities[dim]; ++entity)
    {
      // Add self
      entity_closure[{{dim, entity}}].resize(cell_dim + 1);
      entity_closure[{{dim, entity}}][dim].insert(entity);

      if (dim == 3)
      {
        // Add all sub-entities
        for (int f = 0; f < num_entities[2]; ++f)
          entity_closure[{{dim, entity}}][2].insert(f);
        for (int e = 0; e < num_entities[1]; ++e)
          entity_closure[{{dim, entity}}][1].insert(e);
        for (int v = 0; v < num_entities[0]; ++v)
          entity_closure[{{dim, entity}}][0].insert(v);
      }

      if (dim == 2)
      {
        mesh::CellType face_type = mesh::cell_entity_type(cell_type, 2);
        const int num_edges = mesh::cell_num_entities(face_type, 1);
        for (int e = 0; e < num_edges; ++e)
        {
          // Add edge
          const int edge_index = face_e(entity, e);
          entity_closure[{{dim, entity}}][1].insert(edge_index);
          for (int v = 0; v < 2; ++v)
          {
            // Add vertex connected to edge
            entity_closure[{{dim, entity}}][0].insert(
                edge_v.links(edge_index)[v]);
          }
        }
      }

      if (dim == 1)
      {
        entity_closure[{{dim, entity}}][0].insert(edge_v.links(entity)[0]);
        entity_closure[{{dim, entity}}][0].insert(edge_v.links(entity)[1]);
      }
    }
  }

  return entity_closure;
}
//-----------------------------------------------------------------------------
