/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file id.h
 *
 * @brief This file implement the @ref cl::sycl::id class as defined by the SYCL
 * 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ID_H_
#define RUNTIME_INCLUDE_SYCL_ID_H_

#include "SYCL/index_array.h"
#include "SYCL/item_base.h"

namespace cl {
namespace sycl {

template <int dims>
class range;

template <int dimensions, bool with_offset>
class item;

/** dims-dimensional index
 *
 * @note SYCL only supports dims to be 1, 2, or 3.
 * Using a higher dimension is a Codeplay extension.
 *
 * @tparam dims Number of index dimensions
 */
template <int dims = 1>
class id : public detail::id_range_base<size_t, dims, id<dims>> {
 private:
  using base_t = detail::id_range_base<size_t, dims, id<dims>>;

 public:
  /** Initializes all values to zero
   */
  COMPUTECPP_CONSTEXPR_CPP14 id() noexcept
      : base_t{typename base_t::fill_tag{}, 0} {}

  /** Copy constructor from the base class
   */
  constexpr id(const base_t& other) noexcept : base_t{other} {}

  /** Initializes values per dimension. All values must be specified.
   * @tparam indexesTs Indexes are passed as a parameter pack
   *         to allow for any dimension
   * @param firstIndex Value for first dimension (minimum dimension is 1)
   * @param indexes Values for second and further dimensions
   */
  template <class... indexesTs>
  constexpr id(size_t firstIndex, indexesTs... indexes) noexcept
      : base_t{typename base_t::values_tag{}, firstIndex, indexes...} {
    static_assert(
        (sizeof...(indexes) + 1) == dims,
        "Number of elements passed to id constructor doesn't match dimensions");
  }

  /** Implicit conversion from an index_array
   * @param other Object where the values will be copied from
   */
  COMPUTECPP_CONSTEXPR_CPP14 id(const detail::index_array& other) noexcept
      : id{} {
    static_assert(dims <= 3, "index_array has only 3 dimensions");
    for (int i = 0; i < dims; ++i) {
      this->operator[](i) = other[i];
    }
  }

  /** Implicit conversion from a range
   * @param other Object where the values will be copied from
   */
  COMPUTECPP_CONSTEXPR_CPP14 id(const range<dims>& other) noexcept;

  /** Implicit conversion from an item_base
   * @param index Object where the values will be copied from
   */
  inline id(const detail::item_base& index) : id{} {
    static_assert(dims <= 3, "detail::item_base has only 3 dimensions");
    for (int i = 0; i < dims; ++i) {
      this->operator[](i) = index[i];
    }
  }

  /** Implicit conversion to an index_array
   * @return First dims values used from current instance, the rest are zeros
   */
  COMPUTECPP_CONSTEXPR_CPP14 operator detail::index_array() const noexcept {
    static_assert(dims <= 3, "index_array has only 3 dimensions");
    detail::index_array ret{0, 0, 0};
    for (int i = 0; i < dims; ++i) {
      ret[i] = this->get(i);
    }
    return ret;
  }

  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, +, +=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, -, -=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, *, *=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, /, /=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, %, %=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, &, &=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, |, |=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, ^, ^=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, >>, >>=)
  COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(id, size_t, <<, <<=)
};

#if SYCL_LANGUAGE_VERSION >= 202001

/** Deduction guide for id class template.
 *  Templated indexes are required because that's what the constructor uses.
 */
template <class... indexesTs>
id(size_t, indexesTs...)->id<(sizeof...(indexesTs) + 1)>;

// Some compilers don't parse the generic deduction guide properly,
// add specialized ones

id(size_t)->id<1>;
template <class index1>
id(size_t, index1)->id<2>;
template <class index1, class index2>
id(size_t, index1, index2)->id<3>;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ID_H_
