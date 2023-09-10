/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file index_array.h
 *
 * @brief This file implement the base class for @ref cl::sycl::id and @ref
 * cl::sycl::range classes.
 */

#ifndef RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_
#define RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_

#include "SYCL/assert.h"
#include "SYCL/common.h"
#include "SYCL/type_traits.h"

#include <array>
#include <functional>
#include <utility>

/** @cond COMPUTECPP_DEV */

namespace cl {
namespace sycl {

template <int dims = 1>
class range;

namespace detail {

/** Base class for an array of dims values of type dataT.
 *
 * This class serves as the base for (at least)
 * id, range, index_array, and device_index_array classes.
 * All functionality is at least constexpr in C++14 mode,
 * some functionality even in C++11.
 *
 * Provides all necessary operators required by id and range classes,
 * except for self-assignment operators,
 * which have to be defined in the derived class.
 *
 * @tparam dataT Underlying data type
 * @tparam dims Number of elements. Must be positive. No upper limit.
 * @tparam CRTP Class deriving from id_range_base
 */
template <class dataT, int dims, class CRTP>
class id_range_base;

/** Constructs an instance of id_range_base where each value is the same.
 *  Required because the constructor it calls is not public.
 * @tparam dims Number of elements
 * @tparam CRTP Class deriving from id_range_base
 * @tparam dataT Underlying data type. Should be deduced.
 * @param fill Value used for each element
 * @return Instance of id_range_base
 */
template <int dims, class CRTP, class dataT>
COMPUTECPP_CONSTEXPR_CPP14 id_range_base<dataT, dims, CRTP> make_id_range_base(
    dataT fill);

/** Performs an element-wise binary operation between two classes
 *  deriving from id_range_base
 * @tparam Func Type of operation to perform
 * @tparam CRTP Class deriving from id_range_base
 * @param lhs
 * @param rhs
 * @param func Function object that performs the operation
 * @return New instance as a result of the binary operation
 *         being performed on each element
 */
template <class Func, class CRTP>
COMPUTECPP_CONSTEXPR_CPP14 CRTP id_range_binary_op(CRTP lhs, const CRTP& rhs,
                                                   const Func& func) noexcept;

/** Performs an element-wise binary comparison operation between two classes
 *  deriving from id_range_base
 * @tparam Func Type of operation to perform
 * @tparam CRTP Class deriving from id_range_base
 * @param lhs
 * @param rhs
 * @param func Function object that performs the operation
 * @return Boolean result of the comparison operation
 *         being performed on each element
 */
template <class Func, class CRTP>
COMPUTECPP_CONSTEXPR_CPP14 bool id_range_comparison_op(
    const CRTP& lhs, const CRTP& rhs, const Func& func) noexcept;

/** Returns the same value regardless of the index passed in
 * @tparam dataT Underlying data type
 */
template <class dataT>
struct fill_helper {
  dataT data;
  constexpr dataT operator()(size_t) const noexcept { return data; }
};

template <class dataT, int dims, class CRTP>
class id_range_base {
 public:
  static_assert(dims > 0, "Number of dimensions must be positive");

 protected:
  /** Used to disambiguate between constructors,
   *  ensures the same value is used for all elements
   */
  struct fill_tag {};

  /** Used to disambiguate between constructors,
   *  ensures all values are passed in
   */
  struct values_tag {};

#if defined(__cpp_constexpr) && (__cpp_constexpr >= 201304)
  /** Helper constructor where each value is the same.
   * @tparam IndexesTs
   * @param fill Value used for each element
   */
  template <size_t... IndexesTs>
  constexpr id_range_base(dataT fill, std::index_sequence<IndexesTs...>)
      : m_data{fill_helper<dataT>{fill}(IndexesTs)...} {}

  /** Constructor where each value is the same
   * @param fill Value used for each element
   */
  constexpr id_range_base(fill_tag, dataT fill) noexcept
      : id_range_base{fill, std::make_index_sequence<dims>()} {}
#else
  /** Constructor where each value is the same
   * @param fill Value used for each element
   */
  inline id_range_base(fill_tag, dataT fill) noexcept {
    for (int i = 0; i < dims; ++i) {
      m_data[i] = fill;
    }
  }
#endif  // __cpp_constexpr >= 201304

  /** Specifies value for each element.
   *  values_tag is used to disambiguate from other constructors.
   * @tparam indexesTs Types of values for each element.
   *         Must be convertible to dataT.
   * @param indexes Values for each element.
   *        The number of these must match dims.
   */
  template <class... indexesTs>
  constexpr id_range_base(values_tag, indexesTs... indexes) noexcept
      : m_data{static_cast<dataT>(indexes)...} {}

  /** Retrieves pointer to underlying data array
   * @return Pointer to elements
   */
  COMPUTECPP_CONSTEXPR_CPP14 dataT* data() noexcept { return m_data.data(); }

  /** Retrieves pointer-to-const to underlying data array
   * @return Pointer to elements
   */
  constexpr const dataT* data() const noexcept { return m_data.data(); }

 public:
  /// Underlying data type
  using value_type = dataT;

  /// Number of elements
  static constexpr const int dimensions = dims;

  // Need to make this function a friend
  // so it can use the non-public constructor
  template <int dims_, class CRTP_, class dataT_>
  friend COMPUTECPP_CONSTEXPR_CPP14 id_range_base<dataT_, dims_, CRTP_>
      make_id_range_base(dataT_);

  /** Provides access to the element at position index
   * @param index The index into the element array
   * @return Reference to element
   */
  COMPUTECPP_CONSTEXPR_CPP14
  dataT& operator[](int index) noexcept {
    COMPUTECPP_ASSERT(index >= 0, "Index must be non-negative");
    COMPUTECPP_ASSERT(index < dims, "Index must not exceed dimensions");
    return m_data[index];
  }

  /** Provides const access to the element at position index
   * @param index The index into the element array
   * @return Const reference to element
   */
  COMPUTECPP_CONSTEXPR_CPP14
  const dataT& operator[](int index) const noexcept {
    COMPUTECPP_ASSERT(index >= 0, "Index must be non-negative");
    COMPUTECPP_ASSERT(index < dims, "Index must not exceed dimensions");
    return m_data[index];
  }

  /** Retrieves element at position index
   * @param index The index into the element array
   * @return Copy of the element
   */
  COMPUTECPP_CONSTEXPR_CPP14 dataT get(int index) const noexcept {
    return this->operator[](index);
  }

/** Defines functions that allow operations between CRTP and dataT
 * @param return_t What the function returns
 * @param op Operation to generate functions for, e.g. +
 */
#define COMPUTECPP_INDEX_ARRAY_OP_SCALAR(return_t, op)                         \
  COMPUTECPP_CONSTEXPR_CPP14                                                   \
  friend return_t operator op(const dataT& lhs, const CRTP& rhs) noexcept {    \
    return make_id_range_base<dims, CRTP>(lhs) op rhs;                         \
  }                                                                            \
  COMPUTECPP_CONSTEXPR_CPP14                                                   \
  friend return_t operator op(const CRTP& lhs, const dataT& rhs) noexcept {    \
    return lhs op make_id_range_base<dims, CRTP>(rhs);                         \
  }

  // Comparison operators that return boolean

  COMPUTECPP_CONSTEXPR_CPP14
  friend bool operator==(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_comparison_op(lhs, rhs, std::equal_to<dataT>());
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(bool, ==)

  COMPUTECPP_CONSTEXPR_CPP14
  friend bool operator!=(const CRTP& lhs, const CRTP& rhs) noexcept {
    return !(lhs == rhs);
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(bool, !=)

  // Comparison operators that return new instance
  // as a result of an element-wise comparison

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator>(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::greater<dataT>());
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, >)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator<(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::less<dataT>());
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, <)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator>=(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::greater_equal<dataT>());
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, >=)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator<=(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::less_equal<dataT>());
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, <=)

  // Arithmetic operators

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator+(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::plus<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, +)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator-(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::minus<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, -)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator*(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::multiplies<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, *)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator/(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::divides<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, /)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator%(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::modulus<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, %)

  // Bitwise operators

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator&(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::bit_and<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, &)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator|(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::bit_or<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, |)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator^(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::bit_xor<dataT>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, ^)

  // Logical operators

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator&&(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::logical_and<bool>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, &&)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator||(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs, std::logical_or<bool>{});
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, ||)

  // Shift operators

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator>>(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs,
                              [](const dataT& lhsData, const dataT& rhsData) {
                                return (lhsData >> rhsData);
                              });
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, >>)

  COMPUTECPP_CONSTEXPR_CPP14
  friend CRTP operator<<(const CRTP& lhs, const CRTP& rhs) noexcept {
    return id_range_binary_op(lhs, rhs,
                              [](const dataT& lhsData, const dataT& rhsData) {
                                return (lhsData << rhsData);
                              });
  }
  COMPUTECPP_INDEX_ARRAY_OP_SCALAR(CRTP, <<)

#undef COMPUTECPP_INDEX_ARRAY_OP_SCALAR

 private:
  /// Storage for elements
  std::array<dataT, dims> m_data;
};

template <int dims, class CRTP, class dataT>
COMPUTECPP_CONSTEXPR_CPP14 id_range_base<dataT, dims, CRTP> make_id_range_base(
    dataT fill) {
  return {typename id_range_base<dataT, dims, CRTP>::fill_tag{}, fill};
}

template <class Func, class CRTP>
COMPUTECPP_CONSTEXPR_CPP14 CRTP id_range_binary_op(CRTP lhs, const CRTP& rhs,
                                                   const Func& func) noexcept {
  using dataT = typename CRTP::value_type;
  constexpr const auto dims = CRTP::dimensions;
  for (int i = 0; i < dims; ++i) {
    lhs[i] = static_cast<dataT>(func(lhs[i], rhs[i]));
  }
  return lhs;
}

template <class Func, class CRTP>
COMPUTECPP_CONSTEXPR_CPP14 bool id_range_comparison_op(
    const CRTP& lhs, const CRTP& rhs, const Func& func) noexcept {
  constexpr const auto dims = CRTP::dimensions;
  bool ret = func(lhs[0], rhs[0]);
  for (int i = 1; i < dims; ++i) {
    ret = ret && func(lhs[i], rhs[i]);
  }
  return ret;
}

/** Represents three elements of size_t.
 *
 * This class is closely related to id and range
 * and provides two-way conversions.
 * The main reason it's used is because it's not templated,
 * making it easier to pass it across the library boundary.
 */
class index_array : public detail::id_range_base<size_t, 3, index_array> {
 private:
  using base_t = detail::id_range_base<size_t, 3, index_array>;

 public:
  /** Initializes to all zeros
   */
  constexpr index_array() noexcept
      : base_t{typename base_t::values_tag{}, 0, 0, 0} {}

  /** Sets values for each element
   */
  constexpr index_array(size_t index0, size_t index1, size_t index2) noexcept
      : base_t{typename base_t::values_tag{}, index0, index1, index2} {}

  using base_t::get;

  /** Retrieves pointer to underlying data array
   * @return Pointer to elements
   */
  COMPUTECPP_CONSTEXPR_CPP14 size_t* get() noexcept { return this->data(); }

  /** Retrieves pointer-to-const to underlying data array
   * @return Pointer to elements
   */
  constexpr const size_t* get() const noexcept { return this->data(); }

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  COMPUTECPP_CONSTEXPR_CPP14 bool is_equal(const index_array& rhs) const
      noexcept {
    bool isEqual = (this->get(0) == rhs.get(0));
    COMPUTECPP_IF_CONSTEXPR(dimensions > 1) {
      isEqual = isEqual && (this->get(1) == rhs.get(1));
      COMPUTECPP_IF_CONSTEXPR(dimensions > 2) {
        isEqual = isEqual && (this->get(2) == rhs.get(2));
      }
    }
    return isEqual;
  }

  /** @brief Calculates the number of elements covered by this index array.
   *        Only valid if this object represents a range.
   * @return Number of elements across three dimensions
   */
  COMPUTECPP_CONSTEXPR_CPP14 size_t get_count_impl() const noexcept {
    return this->get(0) * this->get(1) * this->get(2);
  }
};

/** @brief Calculates a row-major linearized index from an offset and a range
 * @param offset The offset from the beginning
 * @param range The original range
 * @return The linearized index
 */
COMPUTECPP_CONSTEXPR_CPP14 size_t construct_linear_row_major_index(
    const detail::index_array& offset, const detail::index_array& range) {
  return construct_linear_row_major_index(offset[0], offset[1], offset[2],
                                          range[0], range[1], range[2]);
}

}  // namespace detail

}  // namespace sycl
}  // namespace cl

/** Defines member functions for self assignment operations
 * @param CRTP Type of the class that derives from id_range_base
 *        where the member functions will be defined
 * @param dataT Underlying storage type for CRTP
 * @param op Operation to generate functions for, e.g. +
 * @param opAssign Self-assign operation to generate functions for, e.g. +=
 */
#define COMPUTECPP_INDEX_ARRAY_SELF_ASSIGN_OP(CRTP, dataT, op, opAssign)       \
  COMPUTECPP_CONSTEXPR_CPP14 CRTP& operator opAssign(                          \
      const CRTP& rhs) noexcept {                                              \
    *this = (*this op rhs);                                                    \
    return *this;                                                              \
  }                                                                            \
  COMPUTECPP_CONSTEXPR_CPP14 CRTP& operator opAssign(                          \
      const dataT& rhs) noexcept {                                             \
    *this = (*this op rhs);                                                    \
    return *this;                                                              \
  }

/** COMPUTECPP_DEV @endcond */
#endif  // RUNTIME_INCLUDE_SYCL_INDEX_ARRAY_H_
