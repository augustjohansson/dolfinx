// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "FiniteElement.h"
#include <basix.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/utils.h>
#include <functional>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
FiniteElement::FiniteElement(const ufc_finite_element& ufc_element)
    : _signature(ufc_element.signature), _family(ufc_element.family),
      _tdim(ufc_element.topological_dimension),
      _space_dim(ufc_element.space_dimension),
      _value_size(ufc_element.value_size),
      _reference_value_size(ufc_element.reference_value_size),
      _hash(std::hash<std::string>{}(_signature)),
      _transform_reference_basis_derivatives(
          ufc_element.transform_reference_basis_derivatives),
      _transform_values(ufc_element.transform_values),
      _apply_dof_transformation(ufc_element.apply_dof_transformation),
      _apply_reverse_dof_transformation(
          ufc_element.apply_reverse_dof_transformation),
      _bs(ufc_element.block_size),
      _interpolation_is_ident(ufc_element.interpolation_is_identity),
      _interpolate_into_cell(ufc_element.interpolate_into_cell),
      _interpolationX(ufc_element.num_interpolation_points,
                      ufc_element.topological_dimension),
      _needs_permutation_data(ufc_element.needs_permutation_data)
{
  // Store interpolation points on the reference element
  std::copy_n(ufc_element.interpolation_points, _interpolationX.size(),
              _interpolationX.data());

  const ufc_shape _shape = ufc_element.cell_shape;
  switch (_shape)
  {
  case interval:
    _cell_shape = mesh::CellType::interval;
    break;
  case triangle:
    _cell_shape = mesh::CellType::triangle;
    break;
  case quadrilateral:
    _cell_shape = mesh::CellType::quadrilateral;
    break;
  case tetrahedron:
    _cell_shape = mesh::CellType::tetrahedron;
    break;
  case hexahedron:
    _cell_shape = mesh::CellType::hexahedron;
    break;
  default:
    throw std::runtime_error(
        "Unknown UFC cell type when building FiniteElement.");
  }
  assert(mesh::cell_dim(_cell_shape) == _tdim);

  static const std::map<ufc_shape, std::string> ufc_to_cell
      = {{vertex, "point"},
         {interval, "interval"},
         {triangle, "triangle"},
         {tetrahedron, "tetrahedron"},
         {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufc_to_cell.at(ufc_element.cell_shape);

  const std::string family = ufc_element.family;

  if (family == "mixed element")
  {
    // This will cause an error, if actually used
    _basix_element_handle = -1;
  }
  else
  {
    _basix_element_handle = basix::register_element(
        family.c_str(), cell_shape.c_str(), ufc_element.degree);

    // Copy over "dof coordinates" from basix
    _refX = basix::points(_basix_element_handle);
  }

  // Fill value dimension
  for (int i = 0; i < ufc_element.value_rank; ++i)
    _value_dimension.push_back(ufc_element.value_dimension(i));

  // Create all sub-elements
  for (int i = 0; i < ufc_element.num_sub_elements; ++i)
  {
    ufc_finite_element* ufc_sub_element = ufc_element.create_sub_element(i);
    _sub_elements.push_back(std::make_shared<FiniteElement>(*ufc_sub_element));
    std::free(ufc_sub_element);
  }
}
//-----------------------------------------------------------------------------
std::string FiniteElement::signature() const noexcept { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType FiniteElement::cell_shape() const noexcept
{
  return _cell_shape;
}
//-----------------------------------------------------------------------------
int FiniteElement::space_dimension() const noexcept { return _space_dim; }
//-----------------------------------------------------------------------------
int FiniteElement::value_size() const noexcept { return _value_size; }
//-----------------------------------------------------------------------------
int FiniteElement::reference_value_size() const noexcept
{
  return _reference_value_size;
}
//-----------------------------------------------------------------------------
int FiniteElement::value_rank() const noexcept
{
  return _value_dimension.size();
}
//-----------------------------------------------------------------------------
int FiniteElement::block_size() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
int FiniteElement::value_dimension(int i) const
{
  if (i >= (int)_value_dimension.size())
    return 1;
  else
    return _value_dimension.at(i);
}
//-----------------------------------------------------------------------------
std::string FiniteElement::family() const noexcept { return _family; }
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  Eigen::ArrayXXd basix_data = basix::tabulate(_basix_element_handle, 0, X)[0];

  const int scalar_reference_value_size = _reference_value_size / _bs;

  assert(basix_data.cols() % scalar_reference_value_size == 0);
  const int scalar_dofs = basix_data.cols() / scalar_reference_value_size;

  assert(reference_values.dimension(0) == X.rows());
  assert(reference_values.dimension(1) == scalar_dofs);
  assert(reference_values.dimension(2) == scalar_reference_value_size);

  assert(basix_data.rows() == X.rows());

  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < scalar_dofs; ++d)
      for (int v = 0; v < scalar_reference_value_size; ++v)
        reference_values(p, d, v) = basix_data(p, d + scalar_dofs * v);
}
//-----------------------------------------------------------------------------
void FiniteElement::evaluate_reference_basis_derivatives(
    Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values, int order,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X) const
{
  assert(order == 1); // TODO: fix this for order > 1

  std::vector<Eigen::ArrayXXd> basix_data
      = basix::tabulate(_basix_element_handle, 1, X);
  for (int p = 0; p < X.rows(); ++p)
    for (int d = 0; d < basix_data[0].cols() / _reference_value_size; ++d)
      for (int v = 0; v < _reference_value_size; ++v)
        for (std::size_t deriv = 0; deriv < basix_data.size() - 1; ++deriv)
          reference_values(p, d, v, deriv)
              = basix_data[deriv](p, d * _reference_value_size + v);
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis(
    Eigen::Tensor<double, 3, Eigen::RowMajor>& values,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();

  int ret = _transform_reference_basis_derivatives(
      values.data(), 0, num_points, reference_values.data(), X.data(), J.data(),
      detJ.data(), K.data());
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
  }
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_reference_basis_derivatives(
    Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
    const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
    const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
{
  assert(_transform_reference_basis_derivatives);
  const int num_points = X.rows();
  int ret = _transform_reference_basis_derivatives(
      values.data(), order, num_points, reference_values.data(), X.data(),
      J.data(), detJ.data(), K.data());
  if (ret == -1)
  {
    throw std::runtime_error("Generated code returned error "
                             "in transform_reference_basis_derivatives");
  }
}
//-----------------------------------------------------------------------------
const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
FiniteElement::dof_coordinates(int cell_perm) const
{
  if (_refX.size() == 0)
  {
    throw std::runtime_error(
        "Dof reference coordinates do not exist for this element.");
  }
  if (!_needs_permutation_data or cell_perm == 0)
    return _refX;

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> outX(
      _refX.rows(), _refX.cols());
  outX = _refX;

  int ret = _apply_dof_transformation(outX.data(), cell_perm, _refX.cols());

  if (ret == -1)
    throw std::runtime_error(
        "Dof reference coordinates cannot be permuted for this element.");

  return outX;
}
//-----------------------------------------------------------------------------
void FiniteElement::transform_values(
    ufc_scalar_t* reference_values,
    const Eigen::Ref<const Eigen::Array<ufc_scalar_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        physical_values,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& coordinate_dofs)
    const
{
  assert(_transform_values);
  _transform_values(reference_values, physical_values.data(),
                    coordinate_dofs.data(), nullptr);
}

//-----------------------------------------------------------------------------
int FiniteElement::num_sub_elements() const noexcept
{
  return _sub_elements.size();
}
//-----------------------------------------------------------------------------
std::size_t FiniteElement::hash() const noexcept { return _hash; }
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const std::vector<int>& component) const
{
  // Recursively extract sub element
  std::shared_ptr<const FiniteElement> sub_finite_element
      = extract_sub_element(*this, component);
  DLOG(INFO) << "Extracted finite element for sub-system: "
             << sub_finite_element->signature().c_str();
  return sub_finite_element;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const FiniteElement>
FiniteElement::extract_sub_element(const FiniteElement& finite_element,
                                   const std::vector<int>& component)
{
  // Check that a sub system has been specified
  if (component.empty())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. No system was specified");
  }

  // Check if there are any sub systems
  if (finite_element.num_sub_elements() == 0)
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. There are no subsystems.");
  }

  // Check the number of available sub systems
  if (component[0] >= finite_element.num_sub_elements())
  {
    throw std::runtime_error(
        "Cannot extract subsystem of finite element. Requested "
        "subsystem out of range.");
  }

  // Get sub system
  std::shared_ptr<const FiniteElement> sub_element
      = finite_element._sub_elements[component[0]];
  assert(sub_element);

  // Return sub system if sub sub system should not be extracted
  if (component.size() == 1)
    return sub_element;

  // Otherwise, recursively extract the sub sub system
  const std::vector<int> sub_component(component.begin() + 1, component.end());

  return extract_sub_element(*sub_element, sub_component);
}
//-----------------------------------------------------------------------------
bool FiniteElement::interpolation_ident() const noexcept
{
  return _interpolation_is_ident;
}
//-----------------------------------------------------------------------------
const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
FiniteElement::interpolation_points() const noexcept
{
  return _interpolationX;
}
//-----------------------------------------------------------------------------
void FiniteElement::interpolate(
    const Eigen::Array<ufc_scalar_t, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& values,
    const std::uint32_t cell_permutation,
    Eigen::Array<ufc_scalar_t, Eigen::Dynamic, 1>& dofs) const
{
  assert(dofs.size() == _space_dim / _bs);
  int ret
      = _interpolate_into_cell(dofs.data(), values.data(), cell_permutation);
  if (ret == -1)
    throw std::runtime_error("Interpolation into this space not supported.");
}
//-----------------------------------------------------------------------------
bool FiniteElement::needs_permutation_data() const noexcept
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_dof_transformation(
    double* data, const std::uint32_t cell_permutation,
    const int block_size) const
{
  _apply_dof_transformation(data, cell_permutation, block_size);
}
//-----------------------------------------------------------------------------
void FiniteElement::apply_reverse_dof_transformation(
    double* data, const std::uint32_t cell_permutation,
    const int block_size) const
{
  _apply_reverse_dof_transformation(data, cell_permutation, block_size);
}
//-----------------------------------------------------------------------------

