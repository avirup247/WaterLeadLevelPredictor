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
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FLOATING_POINT_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FLOATING_POINT_H_

#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/meta.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/type_traits.h"
#include "computecpp/gsl/gsl"

#include "SYCL/builtins/math_symbols.h"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
/** @brief Inverse cosine function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t acos(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(acos, return_t, x);
}

/** @brief Inverse hyperbolic cosine.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t acosh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(acosh, return_t, x);
}

/** @brief Compute `acos(x)/π`
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t acospi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(acospi, return_t, x);
}

/** @brief Inverse sine function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t asin(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(asin, return_t, x);
}

/** @brief Inverse hyperbolic sine.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t asinh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(asinh, return_t, x);
}

/** @brief Compute `asin(x)/π`
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t asinpi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(asinpi, return_t, x);
}

/** @brief Inverse tangent function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t atan(F y_over_x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(atan, return_t, y_over_x);
}

/** @brief Compute `atan(y/x)`.
 * @tparam F must model genfloat.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F atan2(F y, F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(atan2, F, y, x);
}

/** @brief Inverse hyperbolic tangent.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t atanh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(atanh, return_t, x);
}

/** @brief Compute `atan(x)/π`
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t atanpi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(atanpi, return_t, x);
}

/** @brief Compute `atan(y/x)/π`.
 * @tparam F must model genfloat.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F atan2pi(F y, F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(atan2pi, F, y, x);
}

/** @brief Compute cube-root.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t cbrt(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(cbrt, return_t, x);
}

/** @brief Round to integral value using the round to positive infinity rounding
 * mode.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t ceil(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(ceil, return_t, x);
}

/** @brief Returns x with its sign changed to match the sign of y.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F copysign(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(copysign, F, x, y);
}

/** @brief Compute cosine
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t cos(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(cos, return_t, x);
}

/** @brief Compute hyperbolic cosine
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t cosh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(cosh, return_t, x);
}

/** @brief Compute cos(πx)
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t cospi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(cospi, return_t, x);
}

/** @brief Complementary error function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t erfc(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(erfc, return_t, x);
}

/** @brief Error function encountered in integrating the normal distribution.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t erf(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(erf, return_t, x);
}

/** @brief Compute the base-e exponential of x.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t exp(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(exp, return_t, x);
}

/** @brief Exponential base 2 function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t exp2(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(exp2, return_t, x);
}

/** @brief Exponential base 10 function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t exp10(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(exp10, return_t, x);
}

/** @brief Compute `exp(x) - 1.0`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t expm1(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(expm1, return_t, x);
}

/** @brief Compute absolute value of a floating-point number.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t fabs(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(fabs, return_t, x);
}

/** @brief `x - y` if `x > y`, +0 if x is less than or equal to y.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F fdim(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fdim, F, x, y);
}

/** @brief Round to integral value using the round to negative infinity rounding
 * mode.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t floor(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(floor, return_t, x);
}

/** @brief Returns the correctly rounded floating-point representation of the
 * sum of c with the infinitely precise product of a and b.
 *
 * Rounding of intermediate products shall not occur. Edge case behavior is per
 * the IEEE 754-2008 standard.
 * @tparam F must model genfloat.
 */
template <typename F1, typename F2, typename F3,
          typename F = detail::common_return_t<F1, F2, F3>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value &&
                               detail::builtin::is_genfloat<F3>::value))>
F fma(F1 a, F2 b, F3 c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(fma, F, a, b, c);
}

/** @brief Returns y if `x < y`, otherwise it returns x.
 *
 * If one argument is a NaN, fmax() returns the other argument. If both
 * arguments are NaNs, fmax() returns a NaN.
 * @tparam F must model genfloat.
 */
template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES(
              detail::builtin::is_genfloat<F1>::value &&
              (detail::builtin::is_genfloat<F2>::value ||
               detail::builtin::is_sgenfloat<F2, detail::scalar_t<F>>::value))>
F fmax(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fmax, F, x, y);
}

/** @brief Returns y if y < x, otherwise it returns x.
 *
 * If one argument is a NaN, fmin() returns the other argument. If both
 * arguments are NaNs, fmin() returns a NaN.
 * @tparam F must model genfloat.
 */
template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES(
              detail::builtin::is_genfloat<F1>::value &&
              (detail::builtin::is_genfloat<F2>::value ||
               detail::builtin::is_sgenfloat<F2, detail::scalar_t<F>>::value))>
F fmin(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fmin, F, x, y);
}

/** @brief Modulus. Returns `x * y ∗ trunc(x/y)`.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F fmod(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fmod, F, x, y);
}

namespace detail {
/** @brief Constrains multi_ptr to not be a constant pointer (which is not
 * allowed for built-in functions).
 * @tparam T The pointer type.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 * @note For internal use only.
 */
template <typename T, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES(AddressSpace !=
                              access::address_space::constant_space)>
using builtin_ptr = multi_ptr<T, AddressSpace>;
}  // namespace detail

/** @brief Returns `fmin(x - floor (x), 0x1.fffffep-1f)`. `floor(x)` is returned
 * in iptr.
 * @tparam F must model genfloat.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 */
template <typename F, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F fract(F x, detail::builtin_ptr<F, AddressSpace> iptr) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fract, F, x, iptr);
}

/** @brief Extract mantissa and exponent from x.
 *
 * For each component the mantissa returned is a float with magnitude in the
 * interval [1/2, 1) or 0. Each component of x equals mantissa returned * 2exp.
 * @tparam F must model genfloat.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 */
template <typename F, typename Integral, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F>::value &&
                               detail::builtin::is_genint<Integral>::value))>
F frexp(F x, detail::builtin_ptr<Integral, AddressSpace> exp) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(frexp, F, x, exp);
}

/** @brief Compute the value of the square root of `x2 + y2` without undue
 * overflow or underflow.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F hypot(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(hypot, F, x, y);
}

namespace detail {
template <typename T>
struct correct_int {
  using type = cl_int;
};

template <typename T, int N>
struct correct_int<cl::sycl::vec<T, N>> {
  using type = cl::sycl::vec<cl_int, N>;
};

template <typename T>
using correct_int_t = typename correct_int<T>::type;
}  // namespace detail

/** @brief Return the exponent as an integer value.
 * @tparam F must model genfloat.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::correct_int_t<F> ilogb(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(ilogb, detail::correct_int_t<F>, x);
}

/** @brief Multiply x by 2 to the power k.
 * @tparam F must model genfloat.
 * @tparam Integral must model genint.
 */
template <typename F, typename Integral,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F>::value &&
                               detail::builtin::is_genint<Integral>::value))>
F ldexp(F x, Integral k) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(ldexp, F, x, k);
}

/** @brief Multiply x by 2 to the power k.
 * @tparam F must model genfloat.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F ldexp(F x, const int k) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(ldexp, F, x, k);
}

/** @brief Log gamma function.
 *
 * Returns the natural logarithm of the absolute value of the gamma function.
 * The sign of the gamma function is returned in the signp argument of lgamma_r.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t lgamma(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(lgamma, return_t, x);
}

/** @brief Log gamma function.
 *
 * Returns the natural logarithm of the absolute value of the gamma function.
 * The sign of the gamma function is returned in the signp argument of lgamma_r.
 * @tparam F must model genfloat.
 */
template <typename F, typename Integral, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F>::value &&
                               detail::builtin::is_genint<Integral>::value))>
F lgamma_r(F x, detail::builtin_ptr<Integral, AddressSpace> signp) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(lgamma_r, F, x, signp);
}

/** @brief Compute natural logarithm.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t log(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(log, return_t, x);
}

/** @brief Compute a base 2 logarithm.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t log2(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(log2, return_t, x);
}

/** @brief Compute a base 10 logarithm.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t log10(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(log10, return_t, x);
}

/** @brief Compute `loge(1.0 + x)`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t log1p(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(log1p, return_t, x);
}

/** @brief Compute the exponent of x, which is the integral part of logr (|x|).
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t logb(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(logb, return_t, x);
}

/** @brief mad approximates `a * b + c`.
 *
 * Whether or how the product of `a * b` is rounded and how supernormal or
 * subnormal intermediate products are handled is not defined. mad is intended
 * to be used where speed is preferred over accuracy.
 * @tparam F must model genfloat.
 */
template <typename F1, typename F2, typename F3,
          typename F = detail::common_return_t<F1, F2, F3>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value &&
                               detail::builtin::is_genfloat<F3>::value))>
F mad(F1 a, F2 b, F3 c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(mad, F, a, b, c);
}

/** @brief Returns x if |x| > |y|, y if |y| > |x|, otherwise `fmax(x, y)`.
 * @tparam F must model genfloat.
 */
template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F maxmag(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(maxmag, F, x, y);
}

/** @brief Returns x if |x| < |y|, y if |y| < |x|, otherwise `fmin(x, y)`.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F minmag(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(minmag, F, x, y);
}

/** @brief Decompose a floating-point number.
 *
 * The modf function breaks the argument x into integral and fractional parts,
 * each of which has the same sign as the argument. It stores the integral part
 * in the object pointed to by iptr.
 * @tparam F must model genfloat.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 */
template <typename F, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F modf(F x, detail::builtin_ptr<F, AddressSpace> iptr) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(modf, F, x, iptr);
}

namespace detail {
/** @brief Determines the matching integral type for a given genfloat.
 * @tparam F must model genfloat.
 * @note For internal use only.
 */
template <typename T, int = sizeof(scalar_t<T>)>
struct matching_float {};

template <typename T>
struct matching_float<T, 2> {
  using type = half;
};

template <typename T, int N>
struct matching_float<::cl::sycl::vec<T, N>, 2> {
  using type = ::cl::sycl::vec<cl_half, N>;
};

template <typename T>
struct matching_float<T, 4> {
  using type = cl_float;
};

template <typename T, int N>
struct matching_float<::cl::sycl::vec<T, N>, 4> {
  using type = ::cl::sycl::vec<cl_float, N>;
};

template <typename T>
struct matching_float<T, 8> {
  using type = cl_double;
};

template <typename T, int N>
struct matching_float<::cl::sycl::vec<T, N>, 8> {
  using type = ::cl::sycl::vec<cl_double, N>;
};

template <typename T>
using matching_float_t = typename matching_float<T>::type;
}  // namespace detail

/** @brief Returns x if |x| < |y|, y if |y| < |x|, otherwise `fmin(x, y)`.
 * @tparam F must model genfloatf or genfloatd.
 * @tparam Integral If F models genfloatf, then Integral must
 * model ugenint. If F models genfloatd, then Integral must model
 * ugenlonginteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(
              computecpp::gsl::or_<
                  detail::builtin::is_ugenint<Integral>::value,
                  detail::builtin::is_ugenlonginteger<Integral>::value>::value)>
auto nan(Integral nancode) noexcept -> detail::matching_float_t<Integral> {
  return COMPUTECPP_BUILTIN_INVOKE1(nan, detail::matching_float_t<Integral>,
                                    nancode);
}

template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_ushortn<Integral>::value)>
detail::matching_float_t<Integral> nan(Integral nancode) noexcept {
  return ::cl::sycl::detail::halve_width_cast(
      nan(::cl::sycl::detail::double_width_cast(nancode)));
}

/** @brief Computes the next representable single-precision floating-point value
 * following x in the direction of y. Thus, if y is less than x, nextafter()
 * returns the largest representable floating-point number less than x.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F nextafter(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(nextafter, F, x, y);
}

/** @brief Compute x to the power y.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F pow(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(pow, F, x, y);
}

/** @brief Compute x to the power y, where y is an integer.
 * @tparam F must model genfloat.
 * @tparam Integral must model genint.
 */
template <typename F, typename Integral,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F>::value &&
                               detail::builtin::is_genint<Integral>::value))>
F pown(F x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(pown, F, x, y);
}

/** @brief Compute x to the power y, where x >= 0.
 * @tparam F must model genfloat.
 * @pre x >= 0
 */
template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F powr(F1 x, F2 y) noexcept {
  // Expects(x >= 0);
  return COMPUTECPP_BUILTIN_INVOKE2(powr, F, x, y);
}

/** @brief Compute the value r such that `r = x - n * y`, where n is the integer
 * nearest the exact value of x / y.
 *
 * If there are two integers closest to x / y, n shall be the even one. If r is
 * zero, it is given the same sign as x.
 * @tparam F must model genfloat.
 */

template <typename F1, typename F2,
          typename F = detail::common_return_t<F1, F2>,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                               detail::builtin::is_genfloat<F2>::value))>
F remainder(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(remainder, F, x, y);
}

/** @brief The remquo function computes the value r such that `r = x - k * y`,
 * where k is the integer nearest the exact value of x/y.
 *
 * If there are two integers closest to x/y, k shall be the even one. If r is
 * zero, it is given the same sign as x. This is the same value that is returned
 * by the remainder function. remquo also calculates the lower seven bits of the
 * integral quotient x/y, and gives that value the same sign as x/y. It stores
 * this signed value in the object pointed to by quo.
 * @tparam F must model genfloat.
 * @tparam Integral must model genint.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 */
template <typename F, typename Integral, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F>::value &&
                               detail::builtin::is_genint<Integral>::value))>
F remquo(F x, F y, detail::builtin_ptr<Integral, AddressSpace> quo) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(remquo, F, x, y, quo);
}

/** @brief Round to integral value (using round to nearest even rounding mode)
 * in floating-point format. Refer to section 7.1 of the OpenCL 1.2
 * specification document for description of rounding modes.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t rint(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(rint, return_t, x);
}

/** @brief Compute x to the power 1/y.
 * @tparam F must model genfloat.
 * @tparam Integral must model genint.
 */
template <typename F, typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value,
                              detail::builtin::is_genint<Integral>::value)>
F rootn(F x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(rootn, F, x, y);
}

/** @brief Return the integral value nearest to x rounding halfway cases away
 * from zero, regardless of the current rounding direction.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t round(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(round, return_t, x);
}

/** @brief Compute inverse square root.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t rsqrt(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(rsqrt, return_t, x);
}

/** @brief Compute sine.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t sin(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(sin, return_t, x);
}

/** @brief Compute sine and cosine of x.
 *
 * The computed sine is the return value and computed cosine is returned in
 * cosval.
 * @tparam F must model geninteger.
 * @tparam AddressSpace The address space for which the pointer may address.
 * Cannot be constant_space.
 */
template <typename F, access::address_space AddressSpace,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
F sincos(F x, detail::builtin_ptr<F, AddressSpace> cosval) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(sincos, F, x, cosval);
}

/** @brief Compute hyperbolic sine.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t sinh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(sinh, return_t, x);
}

/** @brief Compute `sin(π * x)`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t sinpi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(sinpi, return_t, x);
}

/** @brief Compute square root.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t sqrt(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(sqrt, return_t, x);
}

/** @brief Compute tangent.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t tan(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(tan, return_t, x);
}

/** @brief Compute hyperbolic tangent.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t tanh(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(tanh, return_t, x);
}

/** @brief Compute `tan(π * x)`.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t tanpi(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(tanpi, return_t, x);
}

/** @brief Compute the gamma function.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t tgamma(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(tgamma, return_t, x);
}

/** @brief Round to integral value using the round to zero rounding mode.
 * @tparam F must model genfloat.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t trunc(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(trunc, return_t, x);
}
}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_FLOATING_POINT_H_
