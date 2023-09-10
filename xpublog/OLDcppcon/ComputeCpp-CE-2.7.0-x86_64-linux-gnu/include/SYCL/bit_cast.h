/*****************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/** @file bit_cast.h
 *
 * @brief This file contains the backporting of C++20's std::bit_cast as defined
 * in the sycl 2020 spec
 */

#ifndef RUNTIME_INCLUDE_SYCL_BIT_CAST_H_
#define RUNTIME_INCLUDE_SYCL_BIT_CAST_H_

#include <type_traits>

namespace cl {
namespace sycl {
/** @brief Backporting of C++20's std::bit_cast. Reinterprets the bits from one
 * type to another. Every bit in the value representation of the returned To
 * object is equal to the corresponding bit in the object representation of
 * @ref src
 * @tparam To The type to cast to
 * @tparam From The type being cast from
 * @param src The object of type From to cast
 * @return A new object of type To with the bits equal to src
 */
#if SYCL_LANGUAGE_VERSION >= 202001
template <class To, class From,
          typename = std::enable_if_t<sizeof(To) == sizeof(From) &&
                                      std::is_trivially_copyable_v<From> &&
                                      std::is_trivially_copyable_v<To>>>
constexpr To bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible_v<To>,
                "Destination type must be trivial constructible");

  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}
#endif  // SYCL_LANGUAGE_VERSION >= 202001
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_BIT_CAST_H_
