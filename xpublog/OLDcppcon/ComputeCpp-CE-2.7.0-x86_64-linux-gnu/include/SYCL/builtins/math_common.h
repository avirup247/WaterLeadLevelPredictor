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
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_COMMON_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_COMMON_H_

#include "SYCL/builtins/math_floating_point.h"
#include "SYCL/builtins/math_symbols.h"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus::detail::common
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
/** @breif Returns `fmin(fmax(x, minval), maxval)`.
 *
 * Results are undefined if `minval > maxval`.
 *
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of F1.
 * @pre `minval <= maxval`
 */
template <typename F1, typename F2,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               std::is_same<F1, F2>::value) ||
                              (detail::builtin::is_genfloath<F1>::value &&
                               std::is_same<F2, half>::value) ||
                              (detail::builtin::is_genfloatf<F1>::value &&
                               std::is_same<F2, float>::value) ||
                              (detail::builtin::is_genfloatd<F1>::value &&
                               std::is_same<F2, double>::value))>
F1 clamp(F1 x, F2 minval, F2 maxval) noexcept {
  // Expects(minval <= maxval);
  return COMPUTECPP_BUILTIN_INVOKE3(clamp, F1, x, minval, maxval);
}

/** @brief Converts radians to degrees, i.e. `(180 / π) * x`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t degrees(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(degrees, return_t, x);
}

/** @brief Compute absolute value of a floating-point number. Redirects to
 * `fabs(x, y)`.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of F1.
 */
template <typename F1, typename F2,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               std::is_same<F1, F2>::value) ||
                              (detail::builtin::is_genfloath<F1>::value &&
                               std::is_same<F2, half>::value) ||
                              (detail::builtin::is_genfloatf<F1>::value &&
                               std::is_same<F2, float>::value) ||
                              (detail::builtin::is_genfloatd<F1>::value &&
                               std::is_same<F2, double>::value))>
F1 abs(F1 x, F2 y) noexcept {
  return ::cl::sycl::fabs(x, y);
}

/** @brief Returns y if `x < y`, otherwise it returns x. If x or y are infinite
 * or NaN, the return values are undefined.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of F1.
 */
template <
    typename F1, typename F2, typename F = detail::common_return_t<F1, F2>,
    COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                         detail::builtin::is_genfloat<F2>::value) ||
                        (detail::builtin::is_genfloath<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, half>::value) ||
                        (detail::builtin::is_genfloatf<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, float>::value) ||
                        (detail::builtin::is_genfloatd<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, double>::value))>
F max(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(max, F, x, y);
}

/** @brief Returns y if `x > y`, otherwise it returns x. If x or y are infinite
 * or NaN, the return values are undefined.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of F1.
 */
template <
    typename F1, typename F2, typename F = detail::common_return_t<F1, F2>,
    COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                         detail::builtin::is_genfloat<F2>::value) ||
                        (detail::builtin::is_genfloath<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, half>::value) ||
                        (detail::builtin::is_genfloatf<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, float>::value) ||
                        (detail::builtin::is_genfloatd<F1>::value &&
                         detail::builtin::is_sgenfloat<F2, double>::value))>
F min(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(min, F, x, y);
}

/** @brief Returns the linear blend of x&y implemented as: x + (y - x) * a. a
 * must be a value in the range `0.0 ... 1.0`. If a is not in the range 0.0 ...
 * 1.0, the return values are undefined.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of T1.
 * @pre 0.0 <= a <= 1.0.
 */
template <typename F1, typename F2,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               std::is_same<F1, F2>::value) ||
                              (detail::builtin::is_genfloath<F1>::value &&
                               std::is_same<F2, half>::value) ||
                              (detail::builtin::is_genfloatf<F1>::value &&
                               std::is_same<F2, float>::value) ||
                              (detail::builtin::is_genfloatd<F1>::value &&
                               std::is_same<F2, double>::value))>
F1 mix(F1 x, F1 y, F2 a) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(mix, F1, x, y, a);
}

/** @brief Converts degrees to radians, i.e. `(π / 180) * x`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t radians(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(radians, return_t, x);
}

/** @brief Returns 0.0 if `x < edge`, otherwise it returns 1.0.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of T1.
 */
template <typename F1, typename F2,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               std::is_same<F1, F2>::value) ||
                              (std::is_same<F1, half>::value &&
                               detail::builtin::is_genfloath<F2>::value) ||
                              (std::is_same<F1, float>::value &&
                               detail::builtin::is_genfloatf<F2>::value) ||
                              (std::is_same<F1, double>::value &&
                               detail::builtin::is_genfloatd<F2>::value))>
F2 step(F1 edge, F2 x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(step, F2, edge, x);
}

/** @brief Returns 0.0 if `x <= edge0` and 1.0 if `x >= edge1` and performs
 * smooth Hermite interpolation between 0 and 1 when `edge0 < x < edge1`.
 *
 * This is useful in cases where you would want a threshold function with a
 * smooth transition. This is equivalent to:

```cpp
gentype t;
t = clamp ((x <= edge0)/ (edge1 >= edge0), 0, 1);
return t * t * (3 - 2 * t);
```

 * Results are undefined if `edge0 >= edge1` or if x, edge0 or edge1 is a NaN.
 * @tparam F1 must model genfloat.
 * @tparam F2 must model either genfloat or be the scalar equivalent of T1.
 */
template <typename F1, typename F2,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               std::is_same<F1, F2>::value) ||
                              (std::is_same<F1, half>::value &&
                               detail::builtin::is_genfloath<F2>::value) ||
                              (std::is_same<F1, float>::value &&
                               detail::builtin::is_genfloatf<F2>::value) ||
                              (std::is_same<F1, double>::value &&
                               detail::builtin::is_genfloatd<F2>::value))>
F2 smoothstep(F1 edge0, F1 edge1, F2 x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(smoothstep, F2, edge0, edge1, x);
}

/** @brief Returns 1.0 if x > 0, -0.0 if x = -0.0, +0.0 if x = +0.0, or -1.0 if
 * x < 0. Returns 0.0 if x is a NaN.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t sign(const F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(sign, return_t, x);
}
}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_COMMON_H_
