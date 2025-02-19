// Copyright (C) 2015-2019 Chris Richardson, Garth N. Wells and Igor Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "IndexMap.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>

using namespace dolfinx;
using namespace dolfinx::common;

namespace
{
//-----------------------------------------------------------------------------

/// Compute the owning rank of ghost indices
std::vector<int> get_ghost_ranks(MPI_Comm comm, std::int32_t local_size,
                                 const std::vector<std::int64_t>& ghosts)
{
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  std::vector<std::int32_t> local_sizes(mpi_size);
  MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1, MPI_INT32_T,
                comm);

  // NOTE: We do not use std::partial_sum here as it narrows std::int64_t to
  // std::int32_t.
  // NOTE: Using std::inclusive_scan is possible, but GCC prior to 9.3.0 only
  // includes the parallel version of this algorithm, requiring e.g. Intel TBB.
  std::vector<std::int64_t> all_ranges(mpi_size + 1, 0);
  for (int i = 0; i < mpi_size; ++i)
    all_ranges[i + 1] = all_ranges[i] + local_sizes[i];

  // Compute rank of ghost owners
  std::vector<int> ghost_ranks(ghosts.size(), -1);
  for (std::size_t i = 0; i < ghosts.size(); ++i)
  {
    auto it = std::upper_bound(all_ranges.begin(), all_ranges.end(), ghosts[i]);
    const int p = std::distance(all_ranges.begin(), it) - 1;
    ghost_ranks[i] = p;
  }

  return ghost_ranks;
}
//-----------------------------------------------------------------------------

// FIXME: This functions returns with a special ordering that is not
// documented. Document properly.

/// Compute (owned) global indices shared with neighbor processes
///
/// @param[in] comm MPI communicator where the neighborhood sources are
///   the owning ranks of the callers ghosts (comm_ghost_to_owner)
/// @param[in] ghosts Global index of ghosts indices on the caller
/// @param[in] ghost_src_ranks The src rank on @p comm for each ghost on
///   the caller
/// @return  (i) For each neighborhood rank (destination ranks on comm)
///   a list of my global indices that are ghost on the rank and (ii)
///   displacement vector for each rank
std::tuple<std::vector<std::int64_t>, std::vector<std::int32_t>>
compute_owned_shared(MPI_Comm comm, const std::vector<std::int64_t>& ghosts,
                     const std::vector<std::int32_t>& ghost_src_ranks)
{
  assert(ghosts.size() == ghost_src_ranks.size());

  // Send global index of my ghost indices to the owning rank

  // src ranks have ghosts, dest ranks hold the index owner
  const auto [src_ranks, dest_ranks] = dolfinx::MPI::neighbors(comm);

  // Compute number of ghost indices to send to each owning rank
  std::vector<int> out_edges_num(dest_ranks.size(), 0);
  for (std::size_t i = 0; i < ghost_src_ranks.size(); ++i)
    out_edges_num[ghost_src_ranks[i]]++;

  // Send number of my ghost indices to each owner, and receive number
  // of my owned indices that are ghosted on other ranks
  std::vector<int> in_edges_num(src_ranks.size());
  MPI_Neighbor_alltoall(out_edges_num.data(), 1, MPI_INT, in_edges_num.data(),
                        1, MPI_INT, comm);

  // Prepare communication displacements
  std::vector<int> send_disp(dest_ranks.size() + 1, 0),
      recv_disp(src_ranks.size() + 1, 0);
  std::partial_sum(out_edges_num.begin(), out_edges_num.end(),
                   send_disp.begin() + 1);
  std::partial_sum(in_edges_num.begin(), in_edges_num.end(),
                   recv_disp.begin() + 1);

  // Pack the ghost indices to send the owning rank
  std::vector<std::int64_t> send_indices(send_disp.back());
  {
    std::vector<int> insert_disp(send_disp);
    for (std::size_t i = 0; i < ghosts.size(); ++i)
    {
      const int owner_rank = ghost_src_ranks[i];
      send_indices[insert_disp[owner_rank]] = ghosts[i];
      insert_disp[owner_rank]++;
    }
  }

  // A rank in the neighborhood communicator can have no incoming or
  // outcoming edges. This may cause OpenMPI to crash. Workaround:
  if (in_edges_num.empty())
    in_edges_num.reserve(1);
  if (out_edges_num.empty())
    out_edges_num.reserve(1);

  // May have repeated shared indices with different processes
  std::vector<std::int64_t> recv_indices(recv_disp.back());
  MPI_Neighbor_alltoallv(send_indices.data(), out_edges_num.data(),
                         send_disp.data(), MPI_INT64_T, recv_indices.data(),
                         in_edges_num.data(), recv_disp.data(), MPI_INT64_T,
                         comm);

  // Return global indices received from each rank that ghost my owned
  // indices, and return how many global indices are received from each
  // neighborhood rank
  return {recv_indices, recv_disp};
}
//-----------------------------------------------------------------------------
/// Create neighborhood communicators
/// @param[in] comm Communicator create communicators with neighborhood
///   topology from
/// @param[in] halo_src_ranks Ranks that own indices in the halo (ghost
///   region) of the calling rank
/// @param[in] halo_dest_ranks Ranks that have indices owned by the
///   calling process own in their halo (ghost region)
std::array<MPI_Comm, 3>
compute_asymmetric_communicators(MPI_Comm comm,
                                 const std::vector<int>& halo_src_ranks,
                                 const std::vector<int>& halo_dest_ranks)
{
  std::array comms{MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};

  // Create communicator with edges owner (sources) -> ghost
  // (destinations)
  {
    std::vector<int> sourceweights(halo_src_ranks.size(), 1);
    std::vector<int> destweights(halo_dest_ranks.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, halo_src_ranks.size(), halo_src_ranks.data(),
        sourceweights.data(), halo_dest_ranks.size(), halo_dest_ranks.data(),
        destweights.data(), MPI_INFO_NULL, false, &comms[0]);
  }

  // Create communicator with edges ghost (sources) -> owner
  // (destinations)
  {
    std::vector<int> sourceweights(halo_dest_ranks.size(), 1);
    std::vector<int> destweights(halo_src_ranks.size(), 1);
    MPI_Dist_graph_create_adjacent(
        comm, halo_dest_ranks.size(), halo_dest_ranks.data(),
        sourceweights.data(), halo_src_ranks.size(), halo_src_ranks.data(),
        destweights.data(), MPI_INFO_NULL, false, &comms[1]);
  }

  // Create communicator two-way edges
  // TODO: Aim to remove? used for compatibility
  {
    std::vector<int> neighbors;
    std::set_union(halo_dest_ranks.begin(), halo_dest_ranks.end(),
                   halo_src_ranks.begin(), halo_src_ranks.end(),
                   std::back_inserter(neighbors));
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                    neighbors.end());

    std::vector<int> sourceweights(neighbors.size(), 1);
    std::vector<int> destweights(neighbors.size(), 1);
    MPI_Dist_graph_create_adjacent(comm, neighbors.size(), neighbors.data(),
                                   sourceweights.data(), neighbors.size(),
                                   neighbors.data(), destweights.data(),
                                   MPI_INFO_NULL, false, &comms[2]);
  }

  return comms;
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
std::tuple<std::int64_t, std::vector<std::int32_t>,
           std::vector<std::vector<std::int64_t>>,
           std::vector<std::vector<int>>>
common::stack_index_maps(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // Get process offset
  std::int64_t process_offset = 0;
  for (auto& map : maps)
    process_offset += map.first.get().local_range()[0] * map.second;

  // Get local map offset
  std::vector<std::int32_t> local_offset(maps.size() + 1, 0);
  for (std::size_t f = 1; f < local_offset.size(); ++f)
  {
    const std::int32_t local_size = maps[f - 1].first.get().size_local();
    const int bs = maps[f - 1].second;
    local_offset[f] = local_offset[f - 1] + bs * local_size;
  }

  // Pack old and new composite indices for owned entries that are ghost
  // on other ranks
  std::vector<std::int64_t> indices;
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].second;
    const std::vector<std::int32_t>& forward_indices
        = maps[f].first.get().shared_indices().array();
    const std::int64_t offset = bs * maps[f].first.get().local_range()[0];
    for (std::int32_t local_index : forward_indices)
    {
      for (std::int32_t i = 0; i < bs; ++i)
      {
        // Insert field index, global index, composite global index
        indices.push_back(f);
        indices.push_back(bs * local_index + i + offset);
        indices.push_back(bs * local_index + i + local_offset[f]
                          + process_offset);
      }
    }
  }

  // Build arrays of incoming and outcoming neighborhood ranks
  std::set<std::int32_t> in_neighbor_set, out_neighbor_set;
  for (auto& map : maps)
  {
    MPI_Comm neighbor_comm = map.first.get().comm(IndexMap::Direction::forward);
    auto [source, dest] = dolfinx::MPI::neighbors(neighbor_comm);
    in_neighbor_set.insert(source.begin(), source.end());
    out_neighbor_set.insert(dest.begin(), dest.end());
  }

  const std::vector<int> in_neighbors(in_neighbor_set.begin(),
                                      in_neighbor_set.end());
  const std::vector<int> out_neighbors(out_neighbor_set.begin(),
                                       out_neighbor_set.end());

  // Create neighborhood communicator
  MPI_Comm comm;
  MPI_Dist_graph_create_adjacent(
      maps.at(0).first.get().comm(), in_neighbors.size(), in_neighbors.data(),
      MPI_UNWEIGHTED, out_neighbors.size(), out_neighbors.data(),
      MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted);

  // Figure out how much data to receive from each neighbor
  const int num_my_rows = indices.size();
  std::vector<int> num_rows_recv(indegree);
  MPI_Neighbor_allgather(&num_my_rows, 1, MPI_INT, num_rows_recv.data(), 1,
                         MPI_INT, comm);

  // Compute displacements for data to receive
  std::vector<int> disp(indegree + 1, 0);
  std::partial_sum(num_rows_recv.begin(), num_rows_recv.end(),
                   disp.begin() + 1);

  // Send data to neighbors, and receive data
  std::vector<std::int64_t> data_recv(disp.back());
  MPI_Neighbor_allgatherv(indices.data(), indices.size(), MPI_INT64_T,
                          data_recv.data(), num_rows_recv.data(), disp.data(),
                          MPI_INT64_T, comm);

  // Destroy communicator
  MPI_Comm_free(&comm);

  // Create map (old global index -> new global index) for each field
  std::vector<std::map<int64_t, std::int64_t>> ghost_maps(maps.size());
  for (std::size_t i = 0; i < data_recv.size(); i += 3)
    ghost_maps[data_recv[i]].insert({data_recv[i + 1], data_recv[i + 2]});

  /// Build arrays from old ghost index to composite ghost index for
  /// each field
  std::vector<std::vector<std::int64_t>> ghosts_new(maps.size());
  std::vector<std::vector<int>> ghost_owners_new(maps.size());
  for (std::size_t f = 0; f < maps.size(); ++f)
  {
    const int bs = maps[f].second;
    const std::vector<std::int64_t>& ghosts = maps[f].first.get().ghosts();
    const std::vector<int>& ghost_owners
        = maps[f].first.get().ghost_owner_rank();
    for (std::size_t i = 0; i < ghosts.size(); ++i)
    {
      for (int j = 0; j < bs; ++j)
      {
        auto it = ghost_maps[f].find(bs * ghosts[i] + j);
        assert(it != ghost_maps[f].end());
        ghosts_new[f].push_back(it->second);
        ghost_owners_new[f].push_back(ghost_owners[i]);
      }
    }
  }

  return {process_offset, std::move(local_offset), std::move(ghosts_new),
          std::move(ghost_owners_new)};
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm comm, std::int32_t local_size)
    : _comm_owner_to_ghost(MPI_COMM_NULL), _comm_ghost_to_owner(MPI_COMM_NULL),
      _comm_symmetric(MPI_COMM_NULL)
{
  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM, comm,
                 &request);

  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // FIXME: Remove need to do this
  // Create communicators with empty neighborhoods
  MPI_Comm comm0, comm1, comm2;
  std::vector<int> ranks(0);
  std::vector<int> weights(ranks.size(), 1);
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(),
                                 weights.data(), ranks.size(), ranks.data(),
                                 weights.data(), MPI_INFO_NULL, false, &comm0);
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(),
                                 weights.data(), ranks.size(), ranks.data(),
                                 weights.data(), MPI_INFO_NULL, false, &comm1);
  MPI_Dist_graph_create_adjacent(comm, ranks.size(), ranks.data(),
                                 weights.data(), ranks.size(), ranks.data(),
                                 weights.data(), MPI_INFO_NULL, false, &comm2);
  _comm_owner_to_ghost = dolfinx::MPI::Comm(comm0, false);
  _comm_ghost_to_owner = dolfinx::MPI::Comm(comm1, false);
  _comm_symmetric = dolfinx::MPI::Comm(comm2, false);
  _shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(0);
}
//-----------------------------------------------------------------------------
IndexMap::IndexMap(MPI_Comm mpi_comm, std::int32_t local_size,
                   const std::vector<int>& dest_ranks,
                   const std::vector<std::int64_t>& ghosts,
                   const std::vector<int>& src_ranks)
    : _comm_owner_to_ghost(MPI_COMM_NULL), _comm_ghost_to_owner(MPI_COMM_NULL),
      _comm_symmetric(MPI_COMM_NULL), _ghosts(ghosts)
{
  assert(size_t(ghosts.size()) == src_ranks.size());
  assert(src_ranks == get_ghost_ranks(mpi_comm, local_size, _ghosts));

  // Get global offset (index), using partial exclusive reduction
  std::int64_t offset = 0;
  const std::int64_t local_size_tmp = (std::int64_t)local_size;
  MPI_Request request_scan;
  MPI_Iexscan(&local_size_tmp, &offset, 1, MPI_INT64_T, MPI_SUM, mpi_comm,
              &request_scan);

  // Send local size to sum reduction to get global size
  MPI_Request request;
  MPI_Iallreduce(&local_size_tmp, &_size_global, 1, MPI_INT64_T, MPI_SUM,
                 mpi_comm, &request);

  // Build vector of src ranks for ghosts, i.e. the ranks that own the
  // callers ghosts
  std::vector<std::int32_t> halo_src_ranks = src_ranks;
  std::sort(halo_src_ranks.begin(), halo_src_ranks.end());
  halo_src_ranks.erase(
      std::unique(halo_src_ranks.begin(), halo_src_ranks.end()),
      halo_src_ranks.end());

  // Map ghost owner rank to the rank on neighborhood communicator
  int myrank = -1;
  MPI_Comm_rank(mpi_comm, &myrank);
  _ghost_owners.resize(ghosts.size());
  for (std::size_t j = 0; j < _ghosts.size(); ++j)
  {
    // Get rank of owner on the neighborhood communicator (rank of out
    // edge on _comm_owner_to_ghost)
    const auto it
        = std::find(halo_src_ranks.begin(), halo_src_ranks.end(), src_ranks[j]);
    assert(it != halo_src_ranks.end());
    const int p_neighbor = std::distance(halo_src_ranks.begin(), it);
    if (src_ranks[j] == myrank)
    {
      throw std::runtime_error("IndexMap Error: Ghost in local range. Rank = "
                               + std::to_string(myrank)
                               + ", ghost = " + std::to_string(ghosts[j]));
    }

    // Store owner neighborhood rank for each ghost
    _ghost_owners[j] = p_neighbor;
  }

  // Create communicators with directional edges:
  // (0) owner -> ghost, (1) ghost -> owner, (2) two-way
  std::array comm_array
      = compute_asymmetric_communicators(mpi_comm, halo_src_ranks, dest_ranks);
  _comm_owner_to_ghost = dolfinx::MPI::Comm(comm_array[0], false);
  _comm_ghost_to_owner = dolfinx::MPI::Comm(comm_array[1], false);
  _comm_symmetric = dolfinx::MPI::Comm(comm_array[2], false);

  // Compute owned indices which are ghosted by other ranks, and how
  // many of my indices each neighbor ghosts
  const auto [shared_ind, shared_disp] = compute_owned_shared(
      _comm_ghost_to_owner.comm(), _ghosts, _ghost_owners);

  // Wait for MPI_Iexscan to complete (get offset)
  MPI_Wait(&request_scan, MPI_STATUS_IGNORE);
  _local_range = {offset, offset + local_size};

  // Convert owned global indices that are ghosts on another rank to
  // local indexing
  std::vector<std::int32_t> local_shared_ind(shared_ind.size());
  std::transform(
      shared_ind.begin(), shared_ind.end(), local_shared_ind.begin(),
      [offset](std::int64_t x) -> std::int32_t { return x - offset; });

  _shared_indices = std::make_unique<graph::AdjacencyList<std::int32_t>>(
      std::move(local_shared_ind), std::move(shared_disp));

  // Wait for the MPI_Iallreduce to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> IndexMap::local_range() const noexcept
{
  return _local_range;
}
//-----------------------------------------------------------------------------
std::int32_t IndexMap::num_ghosts() const noexcept { return _ghosts.size(); }
//-----------------------------------------------------------------------------
std::int32_t IndexMap::size_local() const noexcept
{
  return _local_range[1] - _local_range[0];
}
//-----------------------------------------------------------------------------
std::int64_t IndexMap::size_global() const noexcept { return _size_global; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& IndexMap::ghosts() const noexcept
{
  return _ghosts;
}
//-----------------------------------------------------------------------------
void IndexMap::local_to_global(const xtl::span<const std::int32_t>& local,
                               const xtl::span<std::int64_t>& global) const
{
  assert(local.size() <= global.size());
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  for (std::size_t i = 0; i < local.size(); ++i)
  {
    if (local[i] < local_size)
      global[i] = _local_range[0] + local[i];
    else
    {
      assert((local[i] - local_size) < (int)_ghosts.size());
      global[i] = _ghosts[local[i] - local_size];
    }
  }
}
//-----------------------------------------------------------------------------
void IndexMap::global_to_local(const xtl::span<const std::int64_t>& global,
                               const xtl::span<std::int32_t>& local) const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];

  std::vector<std::pair<std::int64_t, std::int32_t>> global_local_ghosts;
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global_local_ghosts.emplace_back(_ghosts[i], i + local_size);

  std::map<std::int64_t, std::int32_t> global_to_local(
      global_local_ghosts.begin(), global_local_ghosts.end());
  for (std::size_t i = 0; i < global.size(); i++)
  {
    std::int64_t index = global[i];
    if (index >= _local_range[0] and index < _local_range[1])
      local[i] = index - _local_range[0];
    else
    {
      if (auto it = global_to_local.find(index); it != global_to_local.end())
        local[i] = it->second;
      else
        local[i] = -1;
    }
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t> IndexMap::global_indices() const
{
  const std::int32_t local_size = _local_range[1] - _local_range[0];
  const std::int32_t num_ghosts = _ghosts.size();
  const std::int64_t global_offset = _local_range[0];
  std::vector<std::int64_t> global(local_size + num_ghosts);
  std::iota(global.begin(), global.begin() + local_size, global_offset);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    global[local_size + i] = _ghosts[i];

  return global;
}
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>&
IndexMap::shared_indices() const noexcept
{
  return *_shared_indices;
}
//-----------------------------------------------------------------------------
std::vector<int> IndexMap::ghost_owner_rank() const
{
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  std::vector<std::int32_t> owners(_ghost_owners.size());
  for (std::size_t i = 0; i < owners.size(); ++i)
    owners[i] = neighbors_in[_ghost_owners[i]];

  return owners;
}
//----------------------------------------------------------------------------
MPI_Comm IndexMap::comm(Direction dir) const
{
  switch (dir)
  {
  case Direction::reverse:
    return _comm_ghost_to_owner.comm();
  case Direction::forward:
    return _comm_owner_to_ghost.comm();
  case Direction::symmetric:
    return _comm_symmetric.comm();
  default:
    throw std::runtime_error("Unknown edge direction for communicator.");
  }
}
//----------------------------------------------------------------------------
std::map<std::int32_t, std::set<int>> IndexMap::compute_shared_indices() const
{
  // Get number of neighbors and neighbor ranks
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  std::map<std::int32_t, std::set<int>> shared_indices;

  // Build map from owned local index to ranks that ghost the index
  for (std::int32_t p = 0; p < _shared_indices->num_nodes(); ++p)
  {
    const int rank_global = neighbors_out[p];
    for (const std::int32_t& idx : _shared_indices->links(p))
      shared_indices[idx].insert(rank_global);
  }

  // Ghost indices know the owner rank, but they don't know about other
  // ranks that also ghost the index. If an index is a ghost on more
  // than one rank, we need to send each rank that ghosts the index the
  // other ranks which also ghost the index.

  std::vector<std::int64_t> fwd_sharing_data;
  std::vector<int> fwd_sharing_offsets{0};
  for (std::int32_t p = 0; p < _shared_indices->num_nodes(); ++p)
  {
    for (const std::int32_t& idx : _shared_indices->links(p))
    {
      assert(shared_indices.find(idx) != shared_indices.end());
      if (auto it = shared_indices.find(idx); it->second.size() > 1)
      {
        // Add global index
        fwd_sharing_data.push_back(idx + _local_range[0]);

        // Add number of sharing ranks
        fwd_sharing_data.push_back(shared_indices[idx].size());

        // Add sharing ranks
        fwd_sharing_data.insert(fwd_sharing_data.end(), it->second.begin(),
                                it->second.end());
      }
    }

    // Add process offset
    fwd_sharing_offsets.push_back(fwd_sharing_data.size());
  }

  // Send sharing rank data from owner to ghosts

  // Send data size to send, and get to-receive sizes
  std::vector<int> send_sizes(outdegree, 0);
  std::vector<int> recv_sizes(indegree);
  std::adjacent_difference(fwd_sharing_offsets.begin() + 1,
                           fwd_sharing_offsets.end(), send_sizes.begin());
  MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                        MPI_INT, _comm_owner_to_ghost.comm());

  // Work out recv offsets and send/receive
  std::vector<int> recv_offsets(recv_sizes.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   recv_offsets.begin() + 1);
  std::vector<std::int64_t> recv_data(recv_offsets.back());

  // Work-around for OpenMPI
  send_sizes.reserve(1);
  recv_sizes.reserve(1);

  // Start data exchange
  MPI_Request request;
  MPI_Ineighbor_alltoallv(
      fwd_sharing_data.data(), send_sizes.data(), fwd_sharing_offsets.data(),
      MPI_INT64_T, recv_data.data(), recv_sizes.data(), recv_offsets.data(),
      MPI_INT64_T, _comm_owner_to_ghost.comm(), &request);

  // For my ghosts, add owning rank to list of sharing ranks
  const std::int32_t size_local = this->size_local();
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    shared_indices[size_local + i].insert(neighbors_in[_ghost_owners[i]]);

  // Build map from global index to local index for ghosts
  std::unordered_map<std::int64_t, std::int32_t> ghosts;
  ghosts.reserve(_ghosts.size());
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    ghosts.emplace(_ghosts[i], i + size_local);

  // Wait for all-to-all to complete
  MPI_Wait(&request, MPI_STATUS_IGNORE);

  // Add other ranks that also 'ghost' my ghost indices
  int myrank = -1;
  MPI_Comm_rank(_comm_owner_to_ghost.comm(), &myrank);
  for (std::size_t i = 0; i < recv_data.size();)
  {
    auto it = ghosts.find(recv_data[i]);
    assert(it != ghosts.end());
    const std::int32_t idx = it->second;
    const int set_size = recv_data[i + 1];
    for (int j = 0; j < set_size; j++)
    {
      if (recv_data[i + 2 + j] != myrank)
        shared_indices[idx].insert(recv_data[i + 2 + +j]);
    }
    i += set_size + 2;
  }

  return shared_indices;
}
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_fwd(xtl::span<const T> local_data,
                           xtl::span<T> remote_data, int n) const
{

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_owner_to_ghost.comm(), &indegree,
                                 &outdegree, &weighted);

  // Get neighbor processes
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_owner_to_ghost.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  const std::int32_t _size_local = size_local();
  if ((int)local_data.size() != n * _size_local)
    throw std::runtime_error("Invalid local size in scatter_fwd");
  if (remote_data.size() != n * _ghosts.size())
    throw std::runtime_error("Invalid remote size in scatter_fwd");

  // Create displacement vectors
  std::vector<std::int32_t> sizes_recv(indegree, 0);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    sizes_recv[_ghost_owners[i]] += n;

  const std::vector<int32_t>& shared_disp = _shared_indices->offsets();
  std::vector<std::int32_t> displs_send(shared_disp.size());
  std::transform(shared_disp.begin(), shared_disp.end(), displs_send.begin(),
                 [n](auto x) { return x * n; });
  std::vector<std::int32_t> sizes_send(outdegree, 0);
  std::adjacent_difference(displs_send.begin() + 1, displs_send.end(),
                           sizes_send.begin());
  std::vector<std::int32_t> displs_recv(indegree + 1, 0);
  std::partial_sum(sizes_recv.begin(), sizes_recv.end(),
                   displs_recv.begin() + 1);

  // Copy into sending buffer
  std::vector<T> data_to_send(displs_send.back());
  const std::vector<std::int32_t>& indices = _shared_indices->array();
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const std::int32_t index = indices[i];
    for (int j = 0; j < n; ++j)
      data_to_send[i * n + j] = local_data[index * n + j];
  }

  // Send/receive data
  std::vector<T> data_to_recv(displs_recv.back());
  MPI_Neighbor_alltoallv(
      data_to_send.data(), sizes_send.data(), displs_send.data(),
      MPI::mpi_type<T>(), data_to_recv.data(), sizes_recv.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _comm_owner_to_ghost.comm());

  // Copy into ghost area ("remote_data")
  std::vector<std::int32_t> displs(displs_recv);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (int j = 0; j < n; ++j)
      remote_data[i * n + j] = data_to_recv[displs[np] + j];
    displs[np] += n;
  }
}
//-----------------------------------------------------------------------------
// \cond turn off doxygen
template void
IndexMap::scatter_fwd<std::int64_t>(xtl::span<const std::int64_t> local_data,
                                    xtl::span<std::int64_t> remote_data,
                                    int n) const;
template void
IndexMap::scatter_fwd<std::int32_t>(xtl::span<const std::int32_t> local_data,
                                    xtl::span<std::int32_t> remote_data,
                                    int n) const;
template void IndexMap::scatter_fwd<double>(xtl::span<const double> local_data,
                                            xtl::span<double> remote_data,
                                            int n) const;
template void IndexMap::scatter_fwd<std::complex<double>>(
    xtl::span<const std::complex<double>> local_data,
    xtl::span<std::complex<double>> remote_data, int n) const;
// \endcond
//-----------------------------------------------------------------------------
template <typename T>
void IndexMap::scatter_rev(xtl::span<T> local_data,
                           xtl::span<const T> remote_data, int n,
                           IndexMap::Mode op) const
{
  if ((int)remote_data.size() != n * num_ghosts())
    throw std::runtime_error("Invalid remote size in scatter_rev");

  if ((int)local_data.size() != n * size_local())
    throw std::runtime_error("Invalid local size in scatter_rev");

  // Get number of neighbors
  int indegree(-1), outdegree(-2), weighted(-1);
  MPI_Dist_graph_neighbors_count(_comm_ghost_to_owner.comm(), &indegree,
                                 &outdegree, &weighted);

  // Get neighbor processes
  std::vector<int> neighbors_in(indegree), neighbors_out(outdegree);
  MPI_Dist_graph_neighbors(_comm_ghost_to_owner.comm(), indegree,
                           neighbors_in.data(), MPI_UNWEIGHTED, outdegree,
                           neighbors_out.data(), MPI_UNWEIGHTED);

  // Compute number of items to send to each process
  std::vector<std::int32_t> send_sizes(outdegree, 0);
  std::vector<std::int32_t> recv_sizes(indegree, 0);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
    send_sizes[_ghost_owners[i]] += n;

  // Create displacement vectors
  std::vector<std::int32_t> displs_send(outdegree + 1, 0);
  std::vector<std::int32_t> displs_recv(indegree + 1, 0);
  for (int i = 0; i < indegree; ++i)
  {
    recv_sizes[i] = _shared_indices->num_links(i) * n;
    displs_recv[i + 1] = displs_recv[i] + recv_sizes[i];
  }

  for (int i = 0; i < outdegree; ++i)
    displs_send[i + 1] = displs_send[i] + send_sizes[i];

  // Fill sending data
  std::vector<T> send_data(displs_send.back());
  std::vector<std::int32_t> displs(displs_send);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    const int np = _ghost_owners[i];
    for (int j = 0; j < n; ++j)
      send_data[displs[np] + j] = remote_data[i * n + j];
    displs[np] += n;
  }

  // Send and receive data
  std::vector<T> recv_data(displs_recv.back());
  MPI_Neighbor_alltoallv(
      send_data.data(), send_sizes.data(), displs_send.data(),
      MPI::mpi_type<T>(), recv_data.data(), recv_sizes.data(),
      displs_recv.data(), MPI::mpi_type<T>(), _comm_ghost_to_owner.comm());

  const std::vector<std::int32_t>& shared_indices = _shared_indices->array();
  // Copy or accumulate into "local_data"
  if (op == Mode::insert)
  {
    for (std::size_t i = 0; i < shared_indices.size(); ++i)
    {
      const std::int32_t index = shared_indices[i];
      for (int j = 0; j < n; ++j)
        local_data[index * n + j] = recv_data[i * n + j];
    }
  }
  else if (op == Mode::add)
  {
    for (std::size_t i = 0; i < shared_indices.size(); ++i)
    {
      const std::int32_t index = shared_indices[i];
      for (int j = 0; j < n; ++j)
        local_data[index * n + j] += recv_data[i * n + j];
    }
  }
}
//-----------------------------------------------------------------------------
// \cond turn off doxygen
template void
IndexMap::scatter_rev<std::int64_t>(xtl::span<std::int64_t> local_data,
                                    xtl::span<const std::int64_t> remote_data,
                                    int n, IndexMap::Mode op) const;

template void
IndexMap::scatter_rev<std::int32_t>(xtl::span<std::int32_t> local_data,
                                    xtl::span<const std::int32_t> remote_data,
                                    int n, IndexMap::Mode op) const;

template void IndexMap::scatter_rev<double>(xtl::span<double> local_data,
                                            xtl::span<const double> remote_data,
                                            int n, IndexMap::Mode op) const;

template void IndexMap::scatter_rev<std::complex<double>>(
    xtl::span<std::complex<double>> local_data,
    xtl::span<const std::complex<double>> remote_data, int n,
    IndexMap::Mode op) const;
// \endcond