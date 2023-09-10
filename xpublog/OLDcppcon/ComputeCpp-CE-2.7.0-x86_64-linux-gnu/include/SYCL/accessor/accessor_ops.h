/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file accessor_ops.h
  @brief Internal file used by the @ref cl::sycl::accessor class
*/
#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_OPS_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_OPS_H_

#include "SYCL/common.h"

namespace cl {
namespace sycl {
namespace detail {

// Forward declaration
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor_buffer_interface;

/** @cond COMPUTECPP_DEV */

/**
  This file contains:
  - @ref subscript_op

  @brief The subscript_op class is used in order to allow multiple subscript
  operator syntax when accessing an accessor object, they are defined for both
  host and device.

  In general they work by having any subscript operator that takes a size_t
  parameter return an subscript_op of 1 dimension less that the original
  accessor or subscript_op object. This continues until you reach an
  subscript_op of dimension 1, at which point it then returns a reference to an
  element of the accessor in a similar way to when the subscript operator is
  called with an id. The exception to this is when you have an accessor that is
  a specialization of access::mode::write and access::target::image or
  access::target:: host_image, instead of returning a reference to an element it
  returns an subscript_op of dimension zero, this is because for writing to
  images an additional intermediate class is required in order to overload the
  assignment operator.
*/

/*******************************************************************************
    subscript_op class
*******************************************************************************/

/**
@brief Intermediate class used to allow multiple subscript operator syntax for
reading and writing to accessors. The class contains multiple different
operators that are enabled using SFINAE, via the COMPUTECPP_ENABLE_IF macro,
depending on
the specialization of the access mode, access target, dimensions and element
type. The subscript_op contains a reference to the original accessor and an
index_array, each subscript operator will add an additional dimension of the
index_array, before returning a new subscript_op. The operators for 1 dimension
are slightly different, they return a reference to the element indexed by the
index_array or read or write the image with the index_array instead.
@ref accessor
@ref COMPUTECPP_ENABLE_IF
@tparam kRefDims Specifies the dimensions of the subscript_op.
@tparam elemT Specifies the element type of the pointer and typedef.
@tparam kDims Specifies the dimensions of the original accessor.
@tparam kMode Specifies the access mode.
@tparam kTarget Specifies the access target.
*/
template <int kRefDims, typename elemT, int kAccDims, access::mode kMode,
          access::target kTarget, access::placeholder isPlaceholder>
class subscript_op {
 public:
  /**
  @brief Alias for the accessor type associated with this subscript class
  */
  using accessor_t = detail::accessor_buffer_interface<elemT, kAccDims, kMode,
                                                       kTarget, isPlaceholder>;

  /**
  Predefined static const declarations of the const boolean expression used
  as enable_if conditions.
  */
  static const bool is_buffer_syntax =
      ((kTarget == access::target::global_buffer ||
        kTarget == access::target::constant_buffer ||
        kTarget == access::target::host_buffer ||
        kTarget == access::target::local));
  static const bool is_atomic = (kMode == access::mode::atomic);
  static constexpr bool is_read_only = (kMode == access::mode::read);
  static const bool is_image_read_syntax =
      ((kTarget == access::target::image ||
        kTarget == access::target::host_image) &&
       (kMode == access::mode::read));
  static const bool is_image_write_syntax =
      ((kTarget == access::target::image ||
        kTarget == access::target::host_image) &&
       (kMode == access::mode::write || kMode == access::mode::discard_write));
  static const bool is_float4 = (std::is_same<elemT, cl::sycl::float4>::value);
  static const bool is_int4 = (std::is_same<elemT, cl::sycl::int4>::value);
  static const bool is_uint4 = (std::is_same<elemT, cl::sycl::uint4>::value);

  using return_t = typename std::conditional<
      is_read_only, elemT,
      typename device_arg<elemT, kRefDims, kMode, kTarget,
                          isPlaceholder>::raw_ref_type>::type;

  /**
  @brief Constructs an subscript_op from a const accessor reference and an
  index_array.
  @param accRef The const reference to the original accessor.
  @param index The index that is being constructed.
  */
  subscript_op(const accessor_t& accRef, detail::index_array index)
      : m_accRef(accRef), m_index(index) {}

  /**
  @brief Constructs an subscript_op from an accessor reference and an
  index_array.
  @param accRef The reference to the original accessor.
  @param index The index that is being constructed.
  */
  subscript_op(accessor_t& accRef, detail::index_array index)
      : m_accRef(accRef), m_index(index) {}

  /**
  @brief Adds the index to the next dimension of the index_array and returns a
  new subscript_op with the original accessor reference and the index_array.
  @tparam COMPUTECPP_ENABLE_IF condition.
  @param index The size_t index for the current subscript operator to be added
  to the index_array.
  */
  template <COMPUTECPP_ENABLE_IF(elemT, (is_buffer_syntax && kRefDims == 2))>
  subscript_op<kRefDims - 1, elemT, kAccDims, kMode, kTarget, isPlaceholder>
  operator[](size_t index) {
    m_index[kAccDims - kRefDims] = index;
    return subscript_op<(kRefDims - 1), elemT, kAccDims, kMode, kTarget,
                        isPlaceholder>(m_accRef, m_index);
  }

  /**
  @brief Adds the index to the next dimension of the index_array and returns a
  reference to the element indexed with the index_array.
  @tparam COMPUTECPP_ENABLE_IF condition.
  @param index The size_t index for the current subscript operator to be added
  to the index_array.
  */
  template <COMPUTECPP_ENABLE_IF(elemT, (is_buffer_syntax && kRefDims == 1))>
  return_t operator[](size_t index) {
    m_index[kAccDims - 1] = index;
    return m_accRef[m_index];
  }

 protected:
  /**
  @brief The reference to the original accessor.
  */
  const accessor_t& m_accRef;
  /**
  @brief The index_array that is being constructed.
  */
  detail::index_array m_index;
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

/******************************************************************************/
/** COMPUTECPP_DEV @endcond */

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_OPS_H_

/******************************************************************************/
