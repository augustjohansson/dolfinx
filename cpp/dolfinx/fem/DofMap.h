// Copyright (C) 2007-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <utility>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
}

namespace dolfinx::fem
{
class ElementDofLayout;

/// Create an adjacency list that maps a global index (process-wise) to
/// the 'unassembled' cell-wise contributions. It is built from the
/// usual (cell, local index) -> global index dof map. An 'unassembled'
/// vector is the stacked cell contributions, ordered by cell index.
///
/// If the usual dof map is:
///
///  Cell:                0          1          2          3
///  Global index:  [ [0, 3, 5], [3, 2, 4], [4, 3, 2], [2, 1, 0]]
///
/// the 'transpose' dof map will be:
///
///  Global index:           0      1        2          3        4      5
///  Unassembled index: [ [0, 11], [10], [4, 8, 9], [1, 3, 7], [5, 6], [2] ]
///
/// @param[in] dofmap The standard dof map that for each cell (node)
/// gives the global (process-wise) index of each local (cell-wise)
/// index.
/// @param[in] num_cells The number of cells (nodes) in @p dofmap to
/// consider. The first @p num_cells are used. This is argument is
/// typically used to exclude ghost cell contributions.
/// @return Map from global (process-wise) index to positions in an
/// unaassembled array. The links for each node are sorted.
graph::AdjacencyList<std::int32_t>
transpose_dofmap(const graph::AdjacencyList<std::int32_t>& dofmap,
                 std::int32_t num_cells);

/// Degree-of-freedom map
///
/// This class handles the mapping of degrees of freedom. It builds a
/// dof map based on an ElementDofLayout on a specific mesh topology. It
/// will reorder the dofs when running in parallel. Sub-dofmaps, both
/// views and copies, are supported.

class DofMap
{
public:
  /// Create a DofMap from the layout of dofs on a reference element, an
  /// IndexMap defining the distribution of dofs across processes and a
  /// vector of indices
  /// @param[in] element The layout of the degrees of freedom on an element
  /// @param[in] index_map The map describing the parallel distribution
  /// of the degrees of freedom
  /// @param[in] index_map_bs The block size associated with the @p
  /// index_map
  /// @param[in] dofmap Adjacency list
  /// (graph::AdjacencyList<std::int32_t>) with the degrees-of-freedom
  /// for each cell
  /// @param[in] bs The block size of the @p dofmap
  template <typename U,
            typename = std::enable_if_t<std::is_same<
                graph::AdjacencyList<std::int32_t>, std::decay_t<U>>::value>>
  DofMap(std::shared_ptr<const ElementDofLayout> element,
         std::shared_ptr<const common::IndexMap> index_map, int index_map_bs,
         U&& dofmap, int bs)
      : element_dof_layout(element), index_map(index_map),
        _index_map_bs(index_map_bs), _dofmap(std::forward<U>(dofmap)), _bs(bs)
  {
    // Do nothing
  }

  // Copy constructor
  DofMap(const DofMap& dofmap) = delete;

  /// Move constructor
  DofMap(DofMap&& dofmap) = default;

  /// Destructor
  virtual ~DofMap() = default;

  // Copy assignment
  DofMap& operator=(const DofMap& dofmap) = delete;

  /// Move assignment
  DofMap& operator=(DofMap&& dofmap) = default;

  /// Local-to-global mapping of dofs on a cell
  /// @param[in] cell The cell index
  /// @return Local-global dof map for the cell (using process-local
  /// indices)
  xtl::span<const std::int32_t> cell_dofs(int cell) const
  {
    return _dofmap.links(cell);
  }

  /// Return the block size for the dofmap
  int bs() const noexcept;

  /// Extract subdofmap component
  /// @param[in] component The component indices
  /// @return The dofmap for the component
  DofMap extract_sub_dofmap(const std::vector<int>& component) const;

  /// Create a "collapsed" dofmap (collapses a sub-dofmap)
  /// @param[in] comm MPI Communicator
  /// @param[in] topology The mesh topology that the dofmap is defined
  /// on
  /// @return The collapsed dofmap
  std::pair<std::unique_ptr<DofMap>, std::vector<std::int32_t>>
  collapse(MPI_Comm comm, const mesh::Topology& topology) const;

  /// Get dofmap data
  /// @return The adjacency list with dof indices for each cell
  const graph::AdjacencyList<std::int32_t>& list() const;

  /// Layout of dofs on an element
  std::shared_ptr<const ElementDofLayout> element_dof_layout;

  /// Index map that describes the parallel distribution of the dofmap
  std::shared_ptr<const common::IndexMap> index_map;

  /// Block size associated with the index_map
  int index_map_bs() const;

private:
  // Block size for the IndexMap
  int _index_map_bs = -1;

  // Cell-local-to-dof map (dofs for cell dofmap[cell])
  graph::AdjacencyList<std::int32_t> _dofmap;

  // Block size for the dofmap
  int _bs = -1;
};
} // namespace dolfinx::fem
