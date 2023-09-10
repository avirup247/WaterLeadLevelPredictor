/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

******************************************************************************/

/** @file range.h
 *
 * @brief This file implements the \ref cl::sycl::range class as defined by the
 * SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_RANGE_H_
#define RUNTIME_INCLUDE_SYCL_RANGE_H_

#include "SYCL/common.h"
#include "SYCL/id.h"
#include "SYCL/index_array.h"
#include "SYCL/info.h"
#include "SYCL/nd_range_base.h"

namespace cl {
namespace sycl {

/** dims-dimensional range
 *
 * @note SYCL only supports dims to be 1, 2, or 3.
 * Using a higher dimension is a Codeplay extension.
 *
 * @tparam dims Number of range dimensions
 */
template <int dims>
class range : public detail::id_range_base<size_t, dims, range<dims>> {
 private:
  using base_t = detail::id_range_base<size_t, dims, range<dims>>;

 public:
  /** Initializes all values to one
   */
  COMPUTECPP_CONSTEXPR_CPP14 range() noexcept
      : base_t{typename base_t::fill_tag{}, 1} {}

  /** Copy constructor from the base class
   */
  constexpr range(const base_t& other) noexcept : base_t{other} {}

  /** Initializes values per dimension. All values must be specified.
   * @tparam indexesTs Indexes are passed as a parameter pack
   *         to allow for any dimension
   * @param firstIndex Value for first dimension (minimum dimension is 1)
   * @param indexes Values for second and further dimensions
   */
  template <class... indexesTs>
  constexpr range(size_t firstIndex, indexesTs... indexes) noexcept
      : base_t{typename base_t::values_tag{}, firstIndex, indexes...} {
    static_assert((sizeof...(indexes) + 1) == dims,
                  "Number of elements passed to range constructor doesn't "
                  "match dimensions");
  }

  /** Implicit conversion from an index_array
   * @param other Object where the values will be copied from
   */
  COMPUTECPP_CONSTEXPR_CPP14 range(const detail::index_array& other) noexcept
      : range{} {
    COMPUTECPP_ASSERT(dims <= 3, "index_array has a maximum of 3 dimensions");
    for (int i = 0; i < dims; ++i) {
      this->operator[](i) = other[i];
    }
  }

  /** Implicit conversion to an index_array
   * @return First dims values used from current instance, the rest are ones
   */
  COMPUTECPP_CONSTEXPR_CPP14 operator detail::index_array() const noexcept {
    COMPUTECPP_ASSERT(dims <= 3, "index_array has a maximum of 3 dimensions");
    detail::index_array ret{1, 1, 1};
    for (int i = 0; i < dims; ++i) {
      ret[i] = this->get(i);
    }
    return ret;
  }

  /** @brief Return the size of the range
   * @return the range size.
   */
  COMPUTECPP_CONSTEXPR_CPP14 size_t size() const noexcept {
    size_t ret = 1;
    for (int i = 0; i < dims; ++i) {
      ret *= this->get(i);
    }
    return ret;
  }

  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, +, +=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, -, -=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, *, *=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, /, /=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, %, %=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, &, &=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, |, |=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, ^, ^=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, >>, >>=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(range, size_t, <<, <<=)
};

/** Implicit construction of an id from a range
 * @tparam dims Number of id and range dimensions
 * @param other Object where the values will be copied from
 */
template <int dims>
COMPUTECPP_CONSTEXPR_CPP14 id<dims>::id(const range<dims>& other) noexcept
    : id{} {
  for (int i = 0; i < dims; ++i) {
    this->operator[](i) = other[i];
  }
}

#if SYCL_LANGUAGE_VERSION >= 202001

/** Deduction guide for range class template.
 *  Templated indexes are required because that's what the constructor uses.
 */
template <class... indexesTs>
range(size_t, indexesTs...)->range<(sizeof...(indexesTs) + 1)>;

// Some compilers don't parse the generic deduction guide properly,
// add specialized ones

range(size_t)->range<1>;
template <class index1>
range(size_t, index1)->range<2>;
template <class index1, class index2>
range(size_t, index1, index2)->range<3>;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

/**
@brief Specialization of info_convert for converting a pointer size_t type to a
range<3> type.
@ref cl::sycl::info_convert
*/
template <>
struct info_convert<size_t*, range<3>> {
  static range<3> cl_to_sycl(size_t* clPtr, size_t numElems,
                             cl_uint /*clParam*/) {
    if (numElems != 3) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          CL_SUCCESS, detail::cpp_error_code::TARGET_FORMAT_ERROR, nullptr,
          "Unable to convert size_t[X] to range<3> because X != 3")
    }
    range<3> syclResult(clPtr[0], clPtr[1], clPtr[2]);
    return syclResult;
  }
};

/** @brief Implements the nd_range class of the SYCL specification.
 * An nd_range contains a global and a local range and an offset.
 */
template <int dimensions = 1>
class nd_range : public detail::nd_range_base {
  using base_t = detail::nd_range_base;

 public:
  static_assert(
      (dimensions > 0 && dimensions < 4),
      "The allowed dimensionality is within the input range of [1,3].");
  /** @brief Construct a nd_range object specifying the global and local range
   * and an optional offset. Note that the global range must divisible by the
   * local range in order to be usable by a \ref handler::parallel_for.
   * @param globalRange The global \ref range
   * @param localRange The local \ref range
   * @param globalOffset The global offset (optional, default to 0)
   */
  nd_range(const range<dimensions> globalRange,
           const range<dimensions> localRange,
           const id<dimensions> globalOffset = id<dimensions>())
      : detail::nd_range_base(globalRange, localRange, globalOffset) {}
  /** @brief Copy constructor. Create a copy of another nd_range.
   * @param ndRangeBase The nd_range to copy
   */
  nd_range(
      const detail::nd_range_base& ndRangeBase)  // NOLINT false +, conversion
      : detail::nd_range_base(ndRangeBase) {}

  /** \brief Return the global \ref range
   * @return The global range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_global with "
      "nd_range::get_global_range.")
  range<dimensions> get_global() const { return this->get_global_range(); }

  /** \brief Return the global \ref range
   * @return The global range
   */
  range<dimensions> get_global_range() const {
    return base_t::get_global_range();
  }
  /** \brief Return the local \ref range
   * @return The local range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_local with "
      "nd_range::get_local_range.")
  range<dimensions> get_local() const { return this->get_local_range(); }

  /** \brief Return the local \ref range
   * @return The local range
   */
  range<dimensions> get_local_range() const {
    return base_t::get_local_range();
  }

  /** \brief Compute the group \ref range
   * @return The group range
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces nd_range::get_group with "
      "nd_range::get_group_range.")
  range<dimensions> get_group() const { return base_t::get_group_range(); }

  /** \brief Compute the group \ref range
   * @return The group range
   */
  range<dimensions> get_group_range() const {
    return base_t::get_group_range();
  }

  /** \brief Return the queue offset
   * @return The offset
   */
  id<dimensions> get_offset() const { return base_t::get_offset(); }

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const nd_range& lhs, const nd_range& rhs) {
    return lhs.is_equal<dimensions>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const nd_range& lhs, const nd_range& rhs) {
    return !(lhs == rhs);
  }
};

namespace detail {

template <int dims = 3>
COMPUTECPP_CONSTEXPR_CPP14 size_t construct_linear_row_major_index(
    const id<dims>& offset, const range<dims>& rng) {
  return construct_linear_row_major_index<dims>(offset[0], offset[1], offset[2],
                                                rng[0], rng[1], rng[2]);
}
template <>
COMPUTECPP_CONSTEXPR_CPP14 size_t
construct_linear_row_major_index<2>(const id<2>& offset, const range<2>& rng) {
  return construct_linear_row_major_index<2>(offset[0], offset[1], 0,  //
                                             rng[0], rng[1], 1);
}
template <>
COMPUTECPP_CONSTEXPR_CPP14 size_t
construct_linear_row_major_index<1>(const id<1>& offset, const range<1>& rng) {
  return construct_linear_row_major_index<1>(offset[0], 0, 0,  //
                                             rng[0], 1, 1);
}

}  // namespace detail

}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_RANGE_H_
