// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_utils.h"
#include "pugixml.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <map>
#include <xtensor/xadapt.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------
// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
std::int64_t get_padded_width(const fem::FiniteElement& e)
{
  const int width = e.value_size();
  const int rank = e.value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  else
    return width;
}
//-----------------------------------------------------------------------------
template <typename Scalar>
std::vector<Scalar> _get_point_data_values(const fem::Function<Scalar>& u)
{
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  const xt::xtensor<Scalar, 2> data_values = u.compute_point_values();

  const int width = get_padded_width(*u.function_space()->element());
  assert(mesh->geometry().index_map());
  const std::size_t num_local_points
      = mesh->geometry().index_map()->size_local();
  assert(data_values.shape(0) >= num_local_points);

  // FIXME: Unpick the below code for the new layout of data from
  //        GenericFunction::compute_vertex_values
  std::vector<Scalar> _data_values(width * num_local_points, 0.0);
  const int value_rank = u.function_space()->element()->value_rank();
  if (value_rank > 0)
  {
    // Transpose vector/tensor data arrays
    const int value_size = u.function_space()->element()->value_size();
    for (std::size_t i = 0; i < num_local_points; i++)
    {
      for (int j = 0; j < value_size; j++)
      {
        int tensor_2d_offset
            = (j > 1 && value_rank == 2 && value_size == 4) ? 1 : 0;
        _data_values[i * width + j + tensor_2d_offset] = data_values(i, j);
      }
    }
  }
  else
  {
    _data_values = std::vector<Scalar>(
        data_values.data(),
        data_values.data() + num_local_points * data_values.shape(1));
  }

  return _data_values;
}
//-----------------------------------------------------------------------------
template <typename Scalar>
std::vector<Scalar> _get_cell_data_values(const fem::Function<Scalar>& u)
{
  assert(u.function_space()->dofmap());
  const auto mesh = u.function_space()->mesh();
  const int value_size = u.function_space()->element()->value_size();
  const int value_rank = u.function_space()->element()->value_rank();

  // Allocate memory for function values at cell centres
  const int tdim = mesh->topology().dim();
  const std::int32_t num_local_cells
      = mesh->topology().index_map(tdim)->size_local();
  const std::int32_t local_size = num_local_cells * value_size;

  // Build lists of dofs and create map
  std::vector<std::int32_t> dof_set;
  dof_set.reserve(local_size);
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap->element_dof_layout);
  const int ndofs = dofmap->element_dof_layout->num_dofs();
  const int bs = dofmap->bs();
  assert(ndofs * bs == value_size);

  for (int cell = 0; cell < num_local_cells; ++cell)
  {
    // Tabulate dofs
    auto dofs = dofmap->cell_dofs(cell);
    for (int i = 0; i < ndofs; ++i)
    {
      for (int j = 0; j < bs; ++j)
        dof_set.push_back(bs * dofs[i] + j);
    }
  }

  // Get values
  std::vector<Scalar> values(dof_set.size());
  const std::vector<Scalar>& _u = u.x()->array();
  for (std::size_t i = 0; i < dof_set.size(); ++i)
    values[i] = _u[dof_set[i]];

  // Pad out data for 2D vectors/tensors
  if (value_rank == 1 and value_size == 2)
  {
    values.resize(3 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      std::array<Scalar, 3> nd = {values[j * 2], values[j * 2 + 1], 0.0};
      std::copy(nd.begin(), nd.end(), std::next(values.begin(), 3 * j));
    }
  }
  else if (value_rank == 2 and value_size == 4)
  {
    values.resize(9 * num_local_cells);
    for (int j = (num_local_cells - 1); j >= 0; --j)
    {
      std::array<Scalar, 9> nd = {values[j * 4],
                                  values[j * 4 + 1],
                                  0.0,
                                  values[j * 4 + 2],
                                  values[j * 4 + 3],
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0};
      std::copy(nd.begin(), nd.end(), std::next(values.begin(), 9 * j));
    }
  }
  return values;
}
//-----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
std::pair<std::string, int>
xdmf_utils::get_cell_type(const pugi::xml_node& topology_node)
{
  assert(topology_node);
  pugi::xml_attribute type_attr = topology_node.attribute("TopologyType");
  assert(type_attr);

  const static std::map<std::string, std::pair<std::string, int>> xdmf_to_dolfin
      = {{"polyvertex", {"point", 1}},
         {"polyline", {"interval", 1}},
         {"edge_3", {"interval", 2}},
         {"triangle", {"triangle", 1}},
         {"triangle_6", {"triangle", 2}},
         {"tetrahedron", {"tetrahedron", 1}},
         {"tetrahedron_10", {"tetrahedron", 2}},
         {"quadrilateral", {"quadrilateral", 1}},
         {"quadrilateral_9", {"quadrilateral", 2}},
         {"quadrilateral_16", {"quadrilateral", 3}},
         {"hexahedron", {"hexahedron", 1}}};

  // Convert XDMF cell type string to DOLFINX cell type string
  std::string cell_type = type_attr.as_string();
  std::transform(cell_type.begin(), cell_type.end(), cell_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto it = xdmf_to_dolfin.find(cell_type);
  if (it == xdmf_to_dolfin.end())
  {
    throw std::runtime_error("Cannot recognise cell type. Unknown value: "
                             + cell_type);
  }
  return it->second;
}
//----------------------------------------------------------------------------
std::array<std::string, 2>
xdmf_utils::get_hdf5_paths(const pugi::xml_node& dataitem_node)
{
  // Check that node is a DataItem node
  assert(dataitem_node);
  const std::string dataitem_str = "DataItem";
  if (dataitem_node.name() != dataitem_str)
  {
    throw std::runtime_error("Node name is \""
                             + std::string(dataitem_node.name())
                             + R"(", expecting "DataItem")");
  }

  // Check that format is HDF
  pugi::xml_attribute format_attr = dataitem_node.attribute("Format");
  assert(format_attr);
  const std::string format = format_attr.as_string();
  if (format.compare("HDF") != 0)
  {
    throw std::runtime_error("DataItem format \"" + format
                             + R"(" is not "HDF")");
  }

  // Get path data
  pugi::xml_node path_node = dataitem_node.first_child();
  assert(path_node);

  // Create string from path and trim leading and trailing whitespace
  std::string path = path_node.text().get();
  boost::algorithm::trim(path);

  // Split string into file path and HD5 internal path
  std::vector<std::string> paths;
  boost::split(paths, path, boost::is_any_of(":"));
  assert(paths.size() == 2);

  return {{paths[0], paths[1]}};
}
//-----------------------------------------------------------------------------
std::string xdmf_utils::get_hdf5_filename(std::string xdmf_filename)
{
  boost::filesystem::path p(xdmf_filename);
  p.replace_extension(".h5");
  if (p.string() == xdmf_filename)
  {
    throw std::runtime_error("Cannot deduce name of HDF5 file from XDMF "
                             "filename. Filename clash. Check XDMF filename");
  }

  return p.string();
}
//-----------------------------------------------------------------------------
std::vector<std::int64_t>
xdmf_utils::get_dataset_shape(const pugi::xml_node& dataset_node)
{
  // Get Dimensions attribute string
  assert(dataset_node);
  pugi::xml_attribute dimensions_attr = dataset_node.attribute("Dimensions");

  // Gets dimensions, if attribute is present
  std::vector<std::int64_t> dims;
  if (dimensions_attr)
  {
    // Split dimensions string
    const std::string dims_str = dimensions_attr.as_string();
    std::vector<std::string> dims_list;
    boost::split(dims_list, dims_str, boost::is_any_of(" "));

    // Cast dims to integers
    for (const auto& d : dims_list)
      dims.push_back(boost::lexical_cast<std::int64_t>(d));
  }

  return dims;
}
//----------------------------------------------------------------------------
std::int64_t xdmf_utils::get_num_cells(const pugi::xml_node& topology_node)
{
  assert(topology_node);

  // Get number of cells from topology
  std::int64_t num_cells_topology = -1;
  pugi::xml_attribute num_cells_attr
      = topology_node.attribute("NumberOfElements");
  if (num_cells_attr)
    num_cells_topology = num_cells_attr.as_llong();

  // Get number of cells from topology dataset
  pugi::xml_node topology_dataset_node = topology_node.child("DataItem");
  assert(topology_dataset_node);
  const std::vector tdims = get_dataset_shape(topology_dataset_node);

  // Check that number of cells can be determined
  if (tdims.size() != 2 and num_cells_topology == -1)
    throw std::runtime_error("Cannot determine number of cells in XDMF mesh");

  // Check for consistency if number of cells appears in both the topology
  // and DataItem nodes
  if (num_cells_topology != -1 and tdims.size() == 2)
  {
    if (num_cells_topology != tdims[0])
      throw std::runtime_error("Cannot determine number of cells in XDMF mesh");
  }

  return std::max(num_cells_topology, tdims[0]);
}
//----------------------------------------------------------------------------
std::vector<double>
xdmf_utils::get_point_data_values(const fem::Function<double>& u)
{
  return _get_point_data_values(u);
}
//-----------------------------------------------------------------------------
std::vector<std::complex<double>>
xdmf_utils::get_point_data_values(const fem::Function<std::complex<double>>& u)
{
  return _get_point_data_values(u);
}
//-----------------------------------------------------------------------------
std::vector<double>
xdmf_utils::get_cell_data_values(const fem::Function<double>& u)
{
  return _get_cell_data_values(u);
}
//-----------------------------------------------------------------------------
std::vector<std::complex<double>>
xdmf_utils::get_cell_data_values(const fem::Function<std::complex<double>>& u)
{
  return _get_cell_data_values(u);
}
//-----------------------------------------------------------------------------
std::string xdmf_utils::vtk_cell_type_str(mesh::CellType cell_type,
                                          int num_nodes)
{
  static const std::map<mesh::CellType, std::map<int, std::string>> vtk_map = {
      {mesh::CellType::point, {{1, "PolyVertex"}}},
      {mesh::CellType::interval, {{2, "PolyLine"}, {3, "Edge_3"}}},
      {mesh::CellType::triangle,
       {{3, "Triangle"}, {6, "Triangle_6"}, {10, "Triangle_10"}}},
      {mesh::CellType::quadrilateral,
       {{4, "Quadrilateral"},
        {9, "Quadrilateral_9"},
        {16, "Quadrilateral_16"}}},
      {mesh::CellType::tetrahedron,
       {{4, "Tetrahedron"}, {10, "Tetrahedron_10"}, {20, "Tetrahedron_20"}}},
      {mesh::CellType::hexahedron, {{8, "Hexahedron"}, {27, "Hexahedron_27"}}}};

  // Get cell family
  auto cell = vtk_map.find(cell_type);
  if (cell == vtk_map.end())
    throw std::runtime_error("Could not find cell type.");

  // Get cell string
  auto cell_str = cell->second.find(num_nodes);
  if (cell_str == cell->second.end())
    throw std::runtime_error("Could not find VTK string for cell order.");

  return cell_str->second;
}
//-----------------------------------------------------------------------------
std::pair<xt::xtensor<std::int32_t, 2>, std::vector<std::int32_t>>
xdmf_utils::extract_local_entities(const mesh::Mesh& mesh, const int entity_dim,
                                   const xt::xtensor<std::int64_t, 2>& entities,
                                   const xtl::span<const std::int32_t>& values)
{
  if (entities.shape(0) != values.size())
    throw std::runtime_error("Number of entities and values must match");

  // Get layout of dofs on 0th entity
  const std::vector<int> entity_layout
      = mesh.geometry().cmap().dof_layout().entity_closure_dofs(entity_dim, 0);
  assert(entity_layout.size() == entities.shape(1));

  auto c_to_v = mesh.topology().connectivity(mesh.topology().dim(), 0);
  if (!c_to_v)
    throw std::runtime_error("Missing cell-vertex connectivity.");

  // Use ElementDofLayout of the cell to get vertex dof indices (local to a
  // cell) i.e. find a map from local vertex index to associated local dof index
  const int num_vertices_per_cell = c_to_v->num_links(0);
  std::vector<int> cell_vertex_dofs(num_vertices_per_cell);
  for (int i = 0; i < num_vertices_per_cell; ++i)
  {
    const std::vector<int> local_index
        = mesh.geometry().cmap().dof_layout().entity_dofs(0, i);
    assert(local_index.size() == 1);
    cell_vertex_dofs[i] = local_index[0];
  }

  // Find map from entity vertex to local (wrt. dof numbering on the entity) dof
  // number E.g. if there are dofs on entity [0 3 6 7 9] and dofs 3 and 7 belong
  // to vertices, then this produces map [1, 3]
  std::vector<int> entity_vertex_dofs;
  for (std::size_t i = 0; i < cell_vertex_dofs.size(); ++i)
  {
    auto it = std::find(entity_layout.begin(), entity_layout.end(),
                        cell_vertex_dofs[i]);
    if (it != entity_layout.end())
      entity_vertex_dofs.push_back(std::distance(entity_layout.begin(), it));
  }

  const mesh::CellType entity_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), entity_dim);
  const std::size_t num_vertices_per_entity
      = mesh::cell_num_entities(entity_type, 0);
  assert(entity_vertex_dofs.size() == num_vertices_per_entity);

  // Throw away input global indices which do not belong to entity vertices
  // This decreases the amount of data needed in parallel communication
  xt::xtensor<std::int64_t, 2> entities_vertices(
      {entities.shape(0), num_vertices_per_entity});
  for (std::size_t e = 0; e < entities_vertices.shape(0); ++e)
  {
    for (std::size_t i = 0; i < entities_vertices.shape(1); ++i)
      entities_vertices(e, i) = entities(e, entity_vertex_dofs[i]);
  }

  // -------------------
  // 1. Send this rank's global "input" nodes indices to the
  //    'postmaster' rank, and receive global "input" nodes for which
  //    this rank is the postmaster

  // Get "input" global node indices (as in the input file before any
  // internal re-ordering)
  const std::vector<std::int64_t>& nodes_g
      = mesh.geometry().input_global_indices();

  // Send input global indices to 'post master' rank, based on input
  // global index value
  const std::int64_t num_nodes_g = mesh.geometry().index_map()->size_global();
  const MPI_Comm comm = mesh.mpi_comm();
  const int comm_size = MPI::size(comm);
  // NOTE: could make this int32_t be sending: index <- index - dest_rank_offset
  std::vector<std::vector<std::int64_t>> nodes_g_send(comm_size);
  for (std::int64_t node : nodes_g)
  {
    // TODO: Optimise this call by adding 'vectorised verion of
    //       MPI::index_owner
    // Figure out which process is the postmaster for the input global index
    const std::int32_t p
        = dolfinx::MPI::index_owner(comm_size, node, num_nodes_g);
    nodes_g_send[p].push_back(node);
  }

  // Send/receive
  const graph::AdjacencyList<std::int64_t> nodes_g_recv
      = MPI::all_to_all(comm, graph::AdjacencyList<std::int64_t>(nodes_g_send));

  // -------------------
  // 2. Send the entity key (nodes list) and tag to the postmaster based
  //    on the lowest index node in the entity 'key'
  //
  //    NOTE: Stage 2 doesn't depend on the data received in Step 1, so
  //    data (i) the communication could be combined, or (ii) the
  //    communication in Step 1 could be make non-blocking.

  std::vector<std::vector<std::int64_t>> entities_send(comm_size);
  std::vector<std::vector<std::int32_t>> values_send(comm_size);
  std::vector<std::int64_t> entity(num_vertices_per_entity);
  for (std::size_t e = 0; e < entities_vertices.shape(0); ++e)
  {
    // Copy vertices for entity and sort
    auto ev = xt::row(entities_vertices, e);
    std::copy(ev.cbegin(), ev.cend(), entity.begin());
    std::sort(entity.begin(), entity.end());

    // Determine postmaster based on lowest entity node
    const std::int32_t p
        = dolfinx::MPI::index_owner(comm_size, entity.front(), num_nodes_g);
    entities_send[p].insert(entities_send[p].end(), entity.begin(),
                            entity.end());
    values_send[p].push_back(values[e]);
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> entities_recv = MPI::all_to_all(
      comm, graph::AdjacencyList<std::int64_t>(entities_send));
  const graph::AdjacencyList<std::int32_t> values_recv
      = MPI::all_to_all(comm, graph::AdjacencyList<std::int32_t>(values_send));

  // -------------------
  // 3. As 'postmaster', send back the entity key (vertex list) and tag
  //    value to ranks that possibly need the data. Do this based on the
  //    first node index in the entity key.

  // NOTE: Could: (i) use a std::unordered_multimap, or (ii) only send
  // owned nodes to the postmaster and use map, unordered_map or
  // std::vector<pair>>, followed by a neighborhood all_to_all at the
  // end.
  //
  // Build map from global node index to ranks that have the node
  std::multimap<std::int64_t, int> node_to_rank;
  for (int p = 0; p < nodes_g_recv.num_nodes(); ++p)
  {
    auto nodes = nodes_g_recv.links(p);
    for (std::int32_t node : nodes)
      node_to_rank.insert({node, p});
  }

  // Figure out which processes are owners of received nodes
  std::vector<std::vector<std::int64_t>> send_nodes_owned(comm_size);
  std::vector<std::vector<std::int32_t>> send_vals_owned(comm_size);
  std::array<std::size_t, 2> shape
      = {entities_recv.array().size() / num_vertices_per_entity,
         num_vertices_per_entity};
  auto _entities_recv
      = xt::adapt(entities_recv.array().data(), entities_recv.array().size(),
                  xt::no_ownership(), shape);

  const std::vector<std::int32_t>& _values_recv = values_recv.array();
  assert(_values_recv.size() == _entities_recv.shape(0));
  for (std::size_t e = 0; e < _entities_recv.shape(0); ++e)
  {
    auto e_recv = xt::row(_entities_recv, e);

    // Find ranks that have node0
    auto [it0, it1] = node_to_rank.equal_range(e_recv[0]);
    for (auto it = it0; it != it1; ++it)
    {
      const int p1 = it->second;
      send_nodes_owned[p1].insert(send_nodes_owned[p1].end(), e_recv.begin(),
                                  e_recv.end());
      send_vals_owned[p1].push_back(_values_recv[e]);
    }
  }

  // TODO: Pack into one MPI call
  const graph::AdjacencyList<std::int64_t> recv_ents = MPI::all_to_all(
      comm, graph::AdjacencyList<std::int64_t>(send_nodes_owned));
  const graph::AdjacencyList<std::int32_t> recv_vals = MPI::all_to_all(
      comm, graph::AdjacencyList<std::int32_t>(send_vals_owned));

  // -------------------
  // 4. From the received (key, value) data, determine which keys
  //    (entities) are on this process.

  // TODO: Rather than using std::map<std::vector<std::int64_t>,
  //       std::int32_t>, use a rectangular array to avoid the
  //       cost of std::vector<std::int64_t> allocations, and sort the
  //       Array by row.
  //
  // TODO: We have already received possibly tagged entities from other
  //       ranks, so we could use the received data to avoid creating
  //       the std::map for *all* entities and just for candidate
  //       entities.

  // Build map from input global indices to local vertex numbers
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  std::map<std::int64_t, std::int32_t> igi_to_vertex;

  for (int c = 0; c < c_to_v->num_nodes(); ++c)
  {
    auto vertices = c_to_v->links(c);
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t v = 0; v < vertices.size(); ++v)
      igi_to_vertex[nodes_g[x_dofs[cell_vertex_dofs[v]]]] = vertices[v];
  }

  std::vector<std::int32_t> entities_new;
  entities_new.reserve(recv_ents.array().size());
  std::vector<std::int32_t> values_new;
  values_new.reserve(recv_vals.array().size());
  for (std::size_t e = 0;
       e < recv_ents.array().size() / num_vertices_per_entity; ++e)
  {
    bool entity_found = true;
    std::vector<std::int32_t> entity(num_vertices_per_entity);
    for (std::size_t i = 0; i < num_vertices_per_entity; ++i)
    {
      const auto it = igi_to_vertex.find(
          recv_ents.array()[e * num_vertices_per_entity + i]);
      if (it == igi_to_vertex.end())
      {
        // As soon as this received index is not in locally owned input
        // global indices skip the entire entity
        entity_found = false;
        break;
      }
      entity[i] = it->second;
    }

    if (entity_found == true)
    {
      entities_new.insert(entities_new.end(), entity.begin(), entity.end());
      values_new.push_back(recv_vals.array()[e]);
    }
  }

  std::array<std::size_t, 2> shape_r = {
      entities_new.size() / num_vertices_per_entity, num_vertices_per_entity};
  auto e_new = xt::adapt(entities_new.data(), entities_new.size(),
                         xt::no_ownership(), shape_r);
  return {e_new, values_new};
}
//-----------------------------------------------------------------------------
