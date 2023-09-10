
//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_GSL_DETAIL_CAST_H_
#define RUNTIME_INCLUDE_GSL_DETAIL_CAST_H_
#include "type_traits.h"
#include <cstring>
#include <memory>

namespace computecpp {
namespace gsl {
/** @brief Performs a bitwise conversion from From to To (a la "type punning").
 *
 * Does not violate strict aliasing rules, which reinterpret_cast might do.
 * @tparam To A pointer to the type that this function should pun to.
 * @tparam From A pointer to the type that this function should pun from.
 * @param from A pointer to the object to pun.
 * @return A pointer to from, but with the type To.
 */
template <typename To, typename From>
To pun_cast(From* from) noexcept {
  static_assert(std::is_pointer<To>::value, "To must be a pointer type.");
  //   static_assert(
  //       alignof(remove_pointer_t<To>) <= alignof(From),
  //       "pun_cast error: alignment of To must be greater-than-or-equal-to "
  //       "alignment of From.");
  auto to = To{};
  std::memcpy(std::addressof(to), std::addressof(from), sizeof from);
  return to;
}

/** @brief Performs a bitwise conversion from From to To (a la "type punning").
 *
 * Does not violate strict aliasing rules, which reinterpret_cast might do.
 * @tparam To A reference to the type that this function should pun to.
 * @tparam From A reference to the type that this function should pun from.
 * @param from A reference to the object to pun.
 * @return A reference to from, but with the type To.
 */
template <typename To, typename From>
To pun_cast(From&& from) noexcept {
  static_assert(
      (std::is_lvalue_reference<To>::value &&
       std::is_lvalue_reference<From>::value) ||
          (std::is_rvalue_reference<To>::value &&
           std::is_rvalue_reference<From>::value),
      "pun_cast error: To and From must both be the same reference type.");
  static_assert(
      sizeof(remove_reference_t<To>) == sizeof(remove_reference_t<From>),
      "pun_cast error: both types have the same size.");
  //   static_assert(
  //       alignof(remove_reference_t<To>) <= alignof(remove_reference_t<From>),
  //       "pun_cast error: alignment of To must be greater-than-or-equal-to "
  //       "alignment of From.");
  return *pun_cast<remove_reference_t<To>*>(&from);
}
}  // namespace gsl
}  // namespace computecpp
#endif  // RUNTIME_INCLUDE_GSL_DETAIL_CAST_H_
