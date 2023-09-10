//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_PRE_CPP17_CONSTEXPR_GLOBAL_H_
#define RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_PRE_CPP17_CONSTEXPR_GLOBAL_H_

namespace computecpp {
namespace gsl {

/** @brief Used to emulate inline constexpr auto for default-constructible
 * literal types.
 */
template <typename T>
struct pre_cpp17_constexpr_global {
  static constexpr T value{};
};

#if __cplusplus < 201703L
template <typename T>
constexpr T pre_cpp17_constexpr_global<T>::value;
#endif  // __cplusplus < 201703L

}  // namespace gsl
}  // namespace computecpp

#endif  // RUNTIME_INCLUDE_COMPUTECPP_GSL_IMPL_PRE_CPP17_CONSTEXPR_GLOBAL_H_
