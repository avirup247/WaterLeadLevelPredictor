/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_math_common_builtins.h

  @brief Defines the public interface for the SYCL common math built-in
  functions. See
  Table 4.108 of SYCL Specification Version 1.2.1 for more details.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FP_NATIVE_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FP_NATIVE_H_

#include "SYCL/builtins/math_symbols.h"
#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/type_traits.h"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
namespace native {
/** @brief Compute cosine.
 *
 * x must be in the range -216 to +216.
 * @tparam F must model genfloatf.
 * @pre -216 <= x <= 216.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
F cos(F x) noexcept {
  // Expects(-216 <= x && x <= 216);
  return COMPUTECPP_BUILTIN_INVOKE1(native_cos, F, x);
}

/** @brief Compute x / y.
 * @tparam F must model genfloatf.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
F divide(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(native_divide, F, x, y);
}

/** @brief Compute the base-e exponential of x.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t exp(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_exp, return_t, x);
}

/** @brief Compute the base-2 exponential of x.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t exp2(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_exp2, return_t, x);
}

/** @brief Compute the base-10 exponential of x.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t exp10(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_exp10, return_t, x);
}

/** @brief Compute natural logarithm.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t log(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_log, return_t, x);
}

/** @brief Compute a base 2 logarithm.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t log2(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_log2, return_t, x);
}

/** @brief Compute a base 10 logarithm.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t log10(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_log10, return_t, x);
}

/** @brief Compute x to the power y, where x >= 0.
 * @tparam F must model genfloatf.
 * @pre x >= 0.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
F powr(F x, F y) noexcept {
  // Expects(x >= 0);
  return COMPUTECPP_BUILTIN_INVOKE2(native_powr, F, x, y);
}

/** @brief Compute reciprocal.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t recip(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_recip, return_t, x);
}

/** @brief Compute inverse square root.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t rsqrt(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_rsqrt, return_t, x);
}

/** @brief Compute sine.
 *
 * x must be in the range -216 to +216.
 * @tparam F must model genfloatf.
 * @pre -216 <= x <= 216.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
F sin(F x) noexcept {
  // Expects(-216 <= x && x <= 216);
  return COMPUTECPP_BUILTIN_INVOKE1(native_sin, F, x);
}

/** @brief Compute square root.
 * @tparam F must model genfloatf.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
return_t sqrt(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(native_sqrt, return_t, x);
}

/** @brief Compute tangent.
 *
 * x must be in the range -216 to +216.
 * @tparam F must model genfloatf.
 * @pre -216 <= x <= 216.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloatf<F>::value)>
F tan(F x) noexcept {
  // Expects(-216 <= x && x <= 216);
  return COMPUTECPP_BUILTIN_INVOKE1(native_tan, F, x);
}
}  // namespace native
}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FP_NATIVE_H_
