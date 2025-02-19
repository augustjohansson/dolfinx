// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dofmapbuilder.h"
#include "ElementDofLayout.h"
#include <algorithm>
#include <cstdlib>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/boostordering.h>
#include <dolfinx/graph/scotch.h>
#include <dolfinx/mesh/Topology.h>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
//-----------------------------------------------------------------------------

/// Build a simple dofmap from ElementDofmap based on mesh entity
/// indices (local and global)
///
/// @param [in] mesh The mesh to build the dofmap on
/// @param [in] topology The mesh topology
/// @param [in] element_dof_layout The layout of dofs on a cell
/// @return Returns {dofmap (local to the process), local-to-global map
/// to get the global index of local dof i, dof indices, vector of
/// {dimension, mesh entity index} for each local dof i}
std::tuple<graph::AdjacencyList<std::int32_t>, std::vector<std::int64_t>,
           std::vector<std::pair<std::int8_t, std::int32_t>>>
build_basic_dofmap(const mesh::Topology& topology,
                   const ElementDofLayout& element_dof_layout)
{
  // Start timer for dofmap initialization
  common::Timer t0("Init dofmap from element dofmap");

  // Topological dimension
  const int D = topology.dim();

  // Generate and number required mesh entities
  std::vector<bool> needs_entities(D + 1, false);
  std::vector<std::int32_t> num_mesh_entities_local(D + 1, 0),
      num_mesh_entities_global(D + 1, 0);
  for (int d = 0; d <= D; ++d)
  {
    if (element_dof_layout.num_entity_dofs(d) > 0)
    {
      if (!topology.connectivity(d, 0))
      {
        throw std::runtime_error(
            "Cannot create basic dofmap. Missing entities of dimension "
            + std::to_string(d) + " .");
      }
      needs_entities[d] = true;
      num_mesh_entities_local[d] = topology.connectivity(d, 0)->num_nodes();
      assert(topology.index_map(d));
      num_mesh_entities_global[d] = topology.index_map(d)->size_global();
    }
  }

  // Collect cell -> entity connectivities
  std::vector<std::shared_ptr<const graph::AdjacencyList<std::int32_t>>>
      connectivity;
  for (int d = 0; d <= D; ++d)
    connectivity.push_back(topology.connectivity(D, d));

  // Build global dof arrays
  std::vector<std::vector<std::int64_t>> global_indices(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    if (needs_entities[d])
    {
      auto map = topology.index_map(d);
      assert(map);
      global_indices[d] = map->global_indices();
    }
  }

  // Number of dofs on this process
  std::int32_t local_size(0), d(0);
  for (std::int32_t n : num_mesh_entities_local)
    local_size += n * element_dof_layout.num_entity_dofs(d++);

  // Number of dofs per cell
  const int local_dim = element_dof_layout.num_dofs();

  // Allocate dofmap memory
  const int num_cells = topology.connectivity(D, 0)->num_nodes();
  std::vector<std::int32_t> dofs(num_cells * local_dim);
  std::vector<std::int32_t> cell_ptr(num_cells + 1, local_dim);
  cell_ptr[0] = 0;
  std::partial_sum(std::next(cell_ptr.begin(), 1), cell_ptr.end(),
                   std::next(cell_ptr.begin(), 1));

  // Allocate entity indices array
  std::vector<std::vector<int32_t>> entity_indices_local(D + 1);
  std::vector<std::vector<int64_t>> entity_indices_global(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    const int num_entities = mesh::cell_num_entities(topology.cell_type(), d);
    entity_indices_local[d].resize(num_entities);
    entity_indices_global[d].resize(num_entities);
  }

  // Entity dofs on cell (dof = entity_dofs[dim][entity][index])
  const std::vector<std::vector<std::set<int>>>& entity_dofs
      = element_dof_layout.entity_dofs_all();

  // Storage for local-to-global map
  std::vector<std::int64_t> local_to_global(local_size);

  // Dof (dim, entity index) marker
  std::vector<std::pair<std::int8_t, std::int32_t>> dof_entity(local_size);

  // Loops over cells and build dofmaps from ElementDofmap
  for (int c = 0; c < connectivity[0]->num_nodes(); ++c)
  {
    // Get local (process) and global cell entity indices
    for (int d = 0; d < D; ++d)
    {
      if (needs_entities[d])
      {
        auto entities = connectivity[d]->links(c);
        for (std::size_t i = 0; i < entities.size(); ++i)
        {
          entity_indices_local[d][i] = entities[i];
          entity_indices_global[d][i] = global_indices[d][entities[i]];
        }
      }
    }

    // Handle cell index separately because cell.entities(D) doesn't work.
    if (needs_entities[D])
    {
      entity_indices_global[D][0] = global_indices[D][c];
      entity_indices_local[D][0] = c;
    }

    // Iterate over each topological dimension of cell
    std::int32_t offset_local = 0;
    std::int64_t offset_global = 0;
    for (auto e_dofs_d = entity_dofs.begin(); e_dofs_d != entity_dofs.end();
         ++e_dofs_d)
    {
      const std::int8_t d = std::distance(entity_dofs.begin(), e_dofs_d);

      // Iterate over each entity of current dimension d
      for (auto e_dofs = e_dofs_d->begin(); e_dofs != e_dofs_d->end(); ++e_dofs)
      {
        // Get entity indices (local to cell, local to process, and
        // global)
        const std::int32_t e = std::distance(e_dofs_d->begin(), e_dofs);
        const std::int32_t e_index_local = entity_indices_local[d][e];
        const std::int64_t e_index_global = entity_indices_global[d][e];

        // Loop over dofs belong to entity e of dimension d (d, e)
        // d: topological dimension
        // e: local entity index
        // dof_local: local index of dof at (d, e)
        const std::int32_t num_entity_dofs = e_dofs->size();
        for (auto dof_local = e_dofs->begin(); dof_local != e_dofs->end();
             ++dof_local)
        {
          const std::int32_t count = std::distance(e_dofs->begin(), dof_local);
          const std::int32_t dof
              = offset_local + num_entity_dofs * e_index_local + count;
          dofs[cell_ptr[c] + *dof_local] = dof;
          local_to_global[dof]
              = offset_global + num_entity_dofs * e_index_global + count;
          dof_entity[dof] = {d, e_index_local};
        }
      }
      offset_local += entity_dofs[d][0].size() * num_mesh_entities_local[d];
      offset_global += entity_dofs[d][0].size() * num_mesh_entities_global[d];
    }
  }

  return {
      graph::AdjacencyList<std::int32_t>(std::move(dofs), std::move(cell_ptr)),
      std::move(local_to_global), std::move(dof_entity)};
}
//-----------------------------------------------------------------------------

/// Compute re-ordering map from old local index to new local index. The
/// M dofs owned by this process are reordered for locality and fill the
/// positions [0, ..., M). Dof owned by another process are placed at
/// the end, i.e. in the positions [M, ..., N), where N is the total
/// number of dofs on this process.
///
/// @param [in] dofmap The basic dofmap data
/// @param [in] topology The mesh topology
/// @return The pair (old-to-new local index map, M), where M is the
/// number of dofs owned by this process
std::pair<std::vector<std::int32_t>, std::int32_t> compute_reordering_map(
    const graph::AdjacencyList<std::int32_t>& dofmap,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity,
    const mesh::Topology& topology)
{
  common::Timer t0("Compute dof reordering map");

  // Get ownership offset for each dimension
  const int D = topology.dim();
  std::vector<std::int32_t> offset(D + 1, -1);
  for (std::size_t d = 0; d < offset.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
      offset[d] = map->size_local();
  }

  // Create map from old index to new contiguous numbering for locally
  // owned dofs. Set to -1 for unowned dofs
  std::vector<int> original_to_contiguous(dof_entity.size(), -1);
  std::int32_t owned_size = 0;
  for (std::size_t i = 0; i < dof_entity.size(); ++i)
  {
    const std::pair<std::int8_t, std::int32_t>& e = dof_entity[i];
    if (e.second < offset[e.first])
      original_to_contiguous[i] = owned_size++;
  }

  // Build local graph, based on dof map with contiguous numbering
  // (unowned dofs excluded)

  std::vector<std::int32_t> graph_data, graph_offsets;
  {
    // Compute maximum number of graph out edges edges per dof
    std::vector<int> num_edges(owned_size);
    for (std::int32_t cell = 0; cell < dofmap.num_nodes(); ++cell)
    {
      auto nodes = dofmap.links(cell);
      for (std::size_t i = 0; i < nodes.size(); ++i)
      {
        const std::int32_t node_i = original_to_contiguous[nodes[i]];

        // Skip unowned node
        if (node_i == -1)
          continue;

        for (std::size_t j = 0; j < nodes.size(); ++j)
        {
          if (i != j and original_to_contiguous[nodes[j]] != -1)
            ++num_edges[node_i];
        }
      }
    }

    // Compute adjacency list with duplicate edges
    std::vector<std::int32_t> offsets(num_edges.size() + 1, 0);
    std::partial_sum(num_edges.begin(), num_edges.end(),
                     std::next(offsets.begin(), 1));
    std::vector<std::int32_t> edges(offsets.back());
    for (std::int32_t cell = 0; cell < dofmap.num_nodes(); ++cell)
    {
      auto nodes = dofmap.links(cell);
      for (std::size_t i = 0; i < nodes.size(); ++i)
      {
        const std::int32_t node_i = original_to_contiguous[nodes[i]];
        if (node_i == -1)
          continue;
        for (std::size_t j = 0; j < nodes.size(); ++j)
        {
          if (const std::int32_t node_j = original_to_contiguous[nodes[j]];
              i != j and node_j != -1)
          {
            edges[offsets[node_i]++] = node_j;
          }
        }
      }
    }
    // Release memory
    std::vector<std::int32_t>().swap(offsets);

    // Eliminate duplicate edges and create AdjacencyList
    graph_offsets.resize(num_edges.size() + 1, 0);
    std::int32_t current_offset = 0;
    for (std::size_t i = 0; i < num_edges.size(); ++i)
    {
      std::sort(std::next(edges.begin(), current_offset),
                std::next(edges.begin(), current_offset + num_edges[i]));
      auto it = std::unique(
          std::next(edges.begin(), current_offset),
          std::next(edges.begin(), current_offset + num_edges[i]));
      graph_data.insert(graph_data.end(),
                        std::next(edges.begin(), current_offset), it);
      graph_offsets[i + 1]
          = graph_offsets[i]
            + std::distance(std::next(edges.begin(), current_offset), it);
      current_offset += num_edges[i];
    }
  }

  std::vector<int> node_remap;
  {
    const graph::AdjacencyList<std::int32_t> graph(std::move(graph_data),
                                                   std::move(graph_offsets));

    // Reorder owned nodes
    const std::string ordering_library = "SCOTCH";
    if (ordering_library == "Boost")
      node_remap = graph::compute_cuthill_mckee(graph, true);
    else if (ordering_library == "SCOTCH")
      std::tie(node_remap, std::ignore) = graph::scotch::compute_gps(graph);
    else if (ordering_library == "random")
    {
      // NOTE: Randomised dof ordering should only be used for
      // testing/benchmarking
      node_remap.resize(graph.num_nodes());
      std::iota(node_remap.begin(), node_remap.end(), 0);
      std::random_device rd;
      std::default_random_engine g(rd());
      std::shuffle(node_remap.begin(), node_remap.end(), g);
    }
    else
    {
      throw std::runtime_error("Requested library '" + ordering_library
                               + "' is unknown");
    }
  }

  // Reconstruct remapped nodes, and place un-owned nodes at the end
  std::vector<int> old_to_new(dof_entity.size(), -1);
  std::int32_t unowned_pos = owned_size;
  assert(old_to_new.size() == original_to_contiguous.size());
  for (std::size_t i = 0; i < original_to_contiguous.size(); ++i)
  {
    // Put nodes that are not owned at the end, otherwise re-number
    const std::int32_t index = original_to_contiguous[i];
    if (index >= 0)
      old_to_new[i] = node_remap[index];
    else
      old_to_new[i] = unowned_pos++;
  }

  return {std::move(old_to_new), owned_size};
}
//-----------------------------------------------------------------------------

/// Get global indices for unowned dofs
/// @param [in] topology The mesh topology
/// @param [in] num_owned The number of nodes owned by this process
/// @param [in] process_offset The node offset for this process, i.e.
///   the global index of owned node i is i + process_offset
/// @param [in] global_indices_old The old global index of the old local
///   node i
/// @param [in] old_to_new The old local index to new local index map
/// @param [in] dof_entity The ith entry gives (topological dim, local
///   index) of the mesh entity to which node i (old local index) is
///   associated
/// @returns The (0) global indices for unowned dofs, (1) owner rank of each
///   unowned dof
std::pair<std::vector<std::int64_t>, std::vector<int>> get_global_indices(
    const mesh::Topology& topology, const std::int32_t num_owned,
    const std::int64_t process_offset,
    const std::vector<std::int64_t>& global_indices_old,
    const std::vector<std::int32_t>& old_to_new,
    const std::vector<std::pair<std::int8_t, std::int32_t>>& dof_entity)
{
  assert(dof_entity.size() == global_indices_old.size());

  const int D = topology.dim();

  // Build list flag for owned mesh entities that are shared, i.e. are a
  // ghost on a neighbor
  std::vector<std::vector<bool>> shared_entity(D + 1);
  for (std::size_t d = 0; d < shared_entity.size(); ++d)
  {
    auto map = topology.index_map(d);
    if (map)
    {
      shared_entity[d] = std::vector<bool>(map->size_local(), false);
      const std::vector<std::int32_t>& forward_indices
          = map->shared_indices().array();
      for (auto entity : forward_indices)
        shared_entity[d][entity] = true;
    }
  }

  // Build list of (global old, global new) index pairs for dofs that
  // are ghosted on other processes
  std::vector<std::vector<std::int64_t>> global(D + 1);

  // Loop over all dofs
  for (std::size_t i = 0; i < dof_entity.size(); ++i)
  {
    // Topological dimension of mesh entity that dof is associated with
    const int d = dof_entity[i].first;

    // Index of mesh entity that dof is associated with
    const int entity = dof_entity[i].second;
    if (entity < (int)shared_entity[d].size() and shared_entity[d][entity])
    {
      global[d].push_back(global_indices_old[i]);
      global[d].push_back(old_to_new[i] + process_offset);
    }
  }

  std::vector<int> requests_dim;
  std::vector<MPI_Request> requests(D + 1);
  std::vector<MPI_Comm> comm(D + 1, MPI_COMM_NULL);
  std::vector<std::vector<std::int64_t>> all_dofs_received(D + 1);
  std::vector<std::vector<int>> recv_offsets(D + 1);
  for (int d = 0; d <= D; ++d)
  {
    // FIXME: This should check which dimension are needed by the dofmap
    auto map = topology.index_map(d);
    if (map)
    {
      comm[d] = map->comm(common::IndexMap::Direction::symmetric);

      // Get number of neighbors
      int indegree(-1), outdegree(-2), weighted(-1);
      MPI_Dist_graph_neighbors_count(comm[d], &indegree, &outdegree, &weighted);

      // Number and values to send and receive
      const int num_indices = global[d].size();
      std::vector<int> num_indices_recv(indegree);
      MPI_Neighbor_allgather(&num_indices, 1, MPI_INT, num_indices_recv.data(),
                             1, MPI_INT, comm[d]);

      // Compute displacements for data to receive. Last entry has total
      // number of received items.
      std::vector<int>& disp = recv_offsets[d];
      disp.resize(indegree + 1);
      std::partial_sum(num_indices_recv.begin(), num_indices_recv.end(),
                       disp.begin() + 1);

      // TODO: use MPI_Ineighbor_alltoallv
      // Send global index of dofs with bcs to all neighbors
      std::vector<std::int64_t>& dofs_received = all_dofs_received[d];
      dofs_received.resize(disp.back());
      MPI_Ineighbor_allgatherv(global[d].data(), global[d].size(), MPI_INT64_T,
                               dofs_received.data(), num_indices_recv.data(),
                               disp.data(), MPI_INT64_T, comm[d],
                               &requests[requests_dim.size()]);
      requests_dim.push_back(d);
    }
  }

  // Build  [local_new - num_owned] -> global old array  broken down by
  // dimension
  std::vector<std::vector<std::int64_t>> local_new_to_global_old(D + 1);
  for (std::size_t i = 0; i < global_indices_old.size(); ++i)
  {
    const int d = dof_entity[i].first;
    std::int32_t local_new = old_to_new[i] - num_owned;
    if (local_new >= 0)
    {
      local_new_to_global_old[d].push_back(global_indices_old[i]);
      local_new_to_global_old[d].push_back(local_new);
    }
  }

  std::vector<std::int64_t> local_to_global_new(old_to_new.size() - num_owned);
  std::vector<int> local_to_global_new_owner(old_to_new.size() - num_owned);
  for (std::size_t i = 0; i < requests_dim.size(); ++i)
  {
    int idx, d;
    MPI_Waitany(requests_dim.size(), requests.data(), &idx, MPI_STATUS_IGNORE);
    d = requests_dim[idx];

    MPI_Comm neighbor_comm
        = topology.index_map(d)->comm(common::IndexMap::Direction::symmetric);
    auto [neighbors, neighbors1] = dolfinx::MPI::neighbors(neighbor_comm);
    assert(neighbors == neighbors1);

    // Build (global old, global new) map for dofs of dimension d
    std::unordered_map<std::int64_t, std::pair<int64_t, int>> global_old_new;
    std::vector<std::int64_t>& dofs_received = all_dofs_received[d];
    std::vector<int>& offsets = recv_offsets[d];
    for (std::size_t j = 0; j < dofs_received.size(); j += 2)
    {
      const auto pos = std::upper_bound(offsets.begin(), offsets.end(), j);
      const int owner = std::distance(offsets.begin(), pos) - 1;
      global_old_new.insert(
          {dofs_received[j], {dofs_received[j + 1], neighbors[owner]}});
    }

    // Build the dimension d part of local_to_global_new vector
    std::vector<std::int64_t>& local_new_to_global_old_d
        = local_new_to_global_old[d];
    for (std::size_t i = 0; i < local_new_to_global_old_d.size(); i += 2)
    {
      auto it = global_old_new.find(local_new_to_global_old_d[i]);
      assert(it != global_old_new.end());
      local_to_global_new[local_new_to_global_old_d[i + 1]] = it->second.first;
      local_to_global_new_owner[local_new_to_global_old_d[i + 1]]
          = it->second.second;
    }
  }

  return {std::move(local_to_global_new), std::move(local_to_global_new_owner)};
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::shared_ptr<common::IndexMap>, int,
           graph::AdjacencyList<std::int32_t>>
fem::build_dofmap_data(MPI_Comm comm, const mesh::Topology& topology,
                       const ElementDofLayout& element_dof_layout)
{
  common::Timer t0("Build dofmap data");

  const int D = topology.dim();

  // Build a simple dofmap based on mesh entity numbering, returning (i)
  // a local dofmap, (ii) local-to-global map for dof indices, and (iii)
  // pair {dimension, mesh entity index} giving the mesh entity that dof
  // i is associated with.
  const auto [node_graph0, local_to_global0, dof_entity0]
      = build_basic_dofmap(topology, element_dof_layout);

  // Compute global dofmap dimension
  std::int64_t global_dimension = 0;
  for (int d = 0; d <= D; ++d)
  {
    if (element_dof_layout.num_entity_dofs(d) > 0)
    {
      assert(topology.index_map(d));
      const std::int64_t n = topology.index_map(d)->size_global();
      global_dimension += n * element_dof_layout.num_entity_dofs(d);
    }
  }

  // Build re-ordering map for data locality and get number of owned
  // nodes
  const auto [old_to_new, num_owned]
      = compute_reordering_map(node_graph0, dof_entity0, topology);

  // Compute process offset for owned nodes
  const std::int64_t process_offset
      = dolfinx::MPI::global_offset(comm, num_owned, true);

  // Get global indices for unowned dofs
  const auto [local_to_global_unowned, local_to_global_owner]
      = get_global_indices(topology, num_owned, process_offset,
                           local_to_global0, old_to_new, dof_entity0);
  assert(local_to_global_unowned.size() == local_to_global_owner.size());

  // Create IndexMap for dofs range on this process
  auto index_map = std::make_unique<common::IndexMap>(
      comm, num_owned,
      dolfinx::MPI::compute_graph_edges(
          comm, std::set<int>(local_to_global_owner.begin(),
                              local_to_global_owner.end())),
      local_to_global_unowned, local_to_global_owner);
  assert(index_map);

  // Build re-ordered dofmap
  std::vector<std::int32_t> dofmap(node_graph0.array().size());
  for (std::int32_t cell = 0; cell < node_graph0.num_nodes(); ++cell)
  {
    // Get dof order on this cell
    auto old_nodes = node_graph0.links(cell);
    const std::int32_t local_dim0 = old_nodes.size();

    for (std::int32_t j = 0; j < local_dim0; ++j)
    {
      const std::int32_t old_node = old_nodes[j];
      const std::int32_t new_node = old_to_new[old_node];
      dofmap[local_dim0 * cell + j] = new_node;
    }
  }

  assert(dofmap.size() % node_graph0.num_nodes() == 0);
  return {std::move(index_map), element_dof_layout.block_size(),
          graph::build_adjacency_list<std::int32_t>(
              std::move(dofmap), dofmap.size() / node_graph0.num_nodes())};
}
//-----------------------------------------------------------------------------
