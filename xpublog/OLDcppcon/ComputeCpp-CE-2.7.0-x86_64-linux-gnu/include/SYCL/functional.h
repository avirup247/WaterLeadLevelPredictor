/*****************************************************************************
  Copyright (C) 2002-2021 Codeplay Software Limited

  All Rights Reserved.
  Codeplay's ComputeCpp
*****************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_FUNCTIONAL_H_
#define RUNTIME_INCLUDE_SYCL_FUNCTIONAL_H_

#include "SYCL/predefines.h"

#include <functional>
#include <limits>

namespace cl {
namespace sycl {

COMPUTECPP_INLINE_EXPERIMENTAL
namespace experimental {

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using plus = std::plus<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using multiplies = std::multiplies<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using bit_and = std::bit_and<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using bit_or = std::bit_or<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using bit_xor = std::bit_xor<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using logical_and = std::logical_and<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
using logical_or = std::logical_or<T>;

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
struct minimum {
  inline constexpr T operator()(const T& lhs, const T& rhs) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return cl::sycl::min(lhs, rhs);
#else
    return std::min(lhs, rhs);
#endif
  }
};

/** @brief Operation for sub_group reduce and scan functions.
 * @see experimental::sub_group.
 */
template <class T = void>
struct maximum {
  inline constexpr T operator()(const T& lhs, const T& rhs) const noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    return cl::sycl::max(lhs, rhs);
#else
    return std::max(lhs, rhs);
#endif
  }
};

/** @brief Specialization for void, the template type will be deduced when
 * calling the functor.
 */
template <>
struct minimum<void> {
  template <class T>
  inline constexpr T operator()(const T& lhs, const T& rhs) const noexcept {
    return std::min(lhs, rhs);
  }
};

/** @brief Specialization for void, the template type will be deduced when
 * calling the functor.
 */
template <>
struct maximum<void> {
  template <class T>
  inline constexpr T operator()(const T& lhs, const T& rhs) const noexcept {
    return std::max(lhs, rhs);
  }
};

}  // namespace experimental

namespace detail {

/** See https://en.cppreference.com/w/cpp/utility/functional/identity
 */
struct identity {
  template <class T>
  constexpr T&& operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }
};

/** @brief Helper struct to get the identity element of BinaryOp.
 * The struct is defined only for supported BinaryOp.
 * @tparam T
 * @tparam BinaryOp
 */
template <class T, class BinaryOp>
struct identity_value;

/** @brief Identity element for @ref experimental::plus.
 */
template <class T>
struct identity_value<T, experimental::plus<T>> {
  static constexpr T value = 0;
};

/** @brief Identity element for @ref experimental::multiplies.
 */
template <class T>
struct identity_value<T, experimental::multiplies<T>> {
  static constexpr T value = 1;
};

/** @brief Identity element for @ref experimental::bit_and.
 */
template <class T>
struct identity_value<T, experimental::bit_and<T>> {
  static constexpr T value = ~T(0);
};

/** @brief Identity element for @ref experimental::bit_or.
 */
template <class T>
struct identity_value<T, experimental::bit_or<T>> {
  static constexpr T value = 0;
};

/** @brief Identity element for @ref experimental::bit_xor.
 */
template <class T>
struct identity_value<T, experimental::bit_xor<T>> {
  static constexpr T value = 0;
};

/** @brief Identity element for @ref experimental::logical_and.
 */
template <class T>
struct identity_value<T, experimental::logical_and<T>> {
  static constexpr T value = true;
};

/** @brief Identity element for @ref experimental::logical_or.
 */
template <class T>
struct identity_value<T, experimental::logical_or<T>> {
  static constexpr T value = false;
};

/** @brief Identity element for @ref experimental::minimum.
 */
template <class T>
struct identity_value<T, experimental::minimum<T>> {
  static constexpr T value = std::numeric_limits<T>::max();
};

/** @brief Identity element for @ref experimental::maximum.
 */
template <class T>
struct identity_value<T, experimental::maximum<T>> {
  static constexpr T value = std::numeric_limits<T>::lowest();
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_FUNCTIONAL_H_
