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
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_INTEGRAL_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_INTEGRAL_H_

#include "SYCL/builtins/math_symbols.h"
#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/meta.h"
#include "SYCL/type_traits.h"
#include "computecpp/gsl/gsl"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus
#endif  // __SYCL_DEVICE_ONLY__

#if defined(__SYCL_DEVICE_ONLY__)
#define COMPUTECPP_BUILTIN_INTEGER_INVOKE2(...)                                \
  COMPUTECPP_BUILTIN_INVOKE2(__VA_ARGS__)
#else
#define COMPUTECPP_BUILTIN_INTEGER_INVOKE2(f, ...)                             \
  COMPUTECPP_BUILTIN_INVOKE2(detail::integer::f, __VA_ARGS__)
#endif  // __SYCL_DEVICE_ONLY_

namespace cl {
namespace sycl {
namespace detail {
template <typename Integral,
          COMPUTECPP_REQUIRES((std::is_integral<Integral>::value))>
make_unsigned_t<Integral> make_genuint_impl(Integral);

template <typename Integral, int N,
          COMPUTECPP_REQUIRES((std::is_integral<Integral>::value))>
vec<make_unsigned_t<Integral>, N> make_genuint_impl(vec<Integral, N>);

template <typename T>
using make_genuint_t = decltype(make_genuint_impl(std::declval<T>()));
}  // namespace detail

/** @brief Returns `|x|`
 * @tparam Integral must model geninteger.
 * @param x
 * @return `x` if `0 <= x`, `-x` otherwise.
 */
template <
    typename Integral,
    COMPUTECPP_REQUIRES((detail::builtin::is_geninteger<Integral>::value))>
auto abs(Integral x) noexcept -> detail::make_genuint_t<Integral> {
  return COMPUTECPP_BUILTIN_INVOKE1(abs, detail::make_genuint_t<Integral>, x);
}

/** @brief Returns `|x - y|` without modulo overflow.
 * @tparam Integral must model geninteger.
 */
template <
    typename Integral,
    COMPUTECPP_REQUIRES((detail::builtin::is_geninteger<Integral>::value))>
auto abs_diff(Integral x, Integral y) noexcept
    -> detail::make_genuint_t<Integral> {
// TODO(Gordon): Temporary fix to get past error with macros, to be resolved
// before merging!
#if defined(__SYCL_DEVICE_ONLY__)
  return COMPUTECPP_BUILTIN_INVOKE2(abs_diff, detail::make_genuint_t<Integral>,
                                    x, y);
#else
  return COMPUTECPP_BUILTIN_INVOKE2(detail::integer::abs_diff,
                                    detail::make_genuint_t<Integral>, x, y);
#endif  // __SYCL_DEVICE_ONLY_
}

/** @brief Returns `x + y` and saturates the result.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral add_sat(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(add_sat, Integral, x, y);
}

/** @brief Returns `(x + y) >> 1`. The intermediate sum does not modulo
 * overflow.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral hadd(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(hadd, Integral, x, y);
}

/** @brief Returns `(x + y + 1) >> 1`. The intermediate sum does not modulo
 * overflow.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral rhadd(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(rhadd, Integral, x, y);
}

/** @brief Returns `min(max(x, minval), maxval)`. Results are undefined if
 * `minval > maxval`.
 * @tparam Integral must model geninteger.
 * @pre minval <= maxval.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral clamp(Integral x, Integral minval, Integral maxval) noexcept {
  // TODO(@cjdb, CPP-673) Uncomment
  // Expects(minval <= maxval);
  return COMPUTECPP_BUILTIN_INVOKE3(clamp, Integral, x, minval, maxval);
}

/** @brief Returns `min(max(x, minval), maxval)`. Results are undefined if
 * `minval > maxval`.
 * @tparam Integral must model geninteger
 * @tparam S must model sgeninteger.
 * @pre minval <= maxval.
 */
template <
    typename Integral, typename S,
    COMPUTECPP_REQUIRES((detail::builtin::is_geninteger<Integral>::value &&
                         detail::builtin::is_sgeninteger<S>::value))>
Integral clamp(Integral x, S minval, S maxval) noexcept {
  // TODO(@cjdb, CPP-673) Uncomment
  // Expects(minval <= maxval);
  return COMPUTECPP_BUILTIN_INVOKE3(clamp, Integral, x, minval, maxval);
}

/** @brief Returns the number of leading 0-bits in `x`, starting at the most
 * significant bit position.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          typename return_t = detail::collapse_swizzled_vec_t<Integral>,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
return_t clz(Integral x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(clz, return_t, x);
}

/** @brief Returns `mul_hi(a, b) + c`.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral mad_hi(Integral a, Integral b, Integral c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(mad_hi, Integral, a, b, c);
}

/** @brief Returns `a * b + c` and saturates the result.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral mad_sat(Integral a, Integral b, Integral c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(mad_sat, Integral, a, b, c);
}

/** @brief Returns y if `x < y`, otherwise it returns x.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral max(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(max, Integral, x, y);
}

/** @brief Returns y if `x < y`, otherwise it returns x.
 * @tparam Integral must model geninteger.
 * @tparam S must model sgeninteger.
 */
template <
    typename Integral, typename S,
    COMPUTECPP_REQUIRES((detail::builtin::is_geninteger<Integral>::value &&
                         detail::builtin::is_sgeninteger<S>::value))>
Integral max(Integral x, S y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(max, Integral, x, y);
}

/** @brief Returns y if `y < x`, otherwise it returns x.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral min(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(min, Integral, x, y);
}

/** @brief Returns y if `y < x`, otherwise it returns x.
 * @tparam Integral must model geninteger.
 * @tparam S must model sgeninteger.
 */
template <
    typename Integral, typename S,
    COMPUTECPP_REQUIRES((detail::builtin::is_geninteger<Integral>::value &&
                         detail::builtin::is_sgeninteger<S>::value))>
Integral min(Integral x, S y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(min, Integral, x, y);
}

/** @brief Computes `x * y` and returns the high half of the product of `x` and
 * `y`.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral mul_hi(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(mul_hi, Integral, x, y);
}

/** @brief For each element in v, the bits are shifted left by the number of
 * bits given by the corresponding element in i (subject to usual shift modulo
 * rules described in section 6.3).
 *
 * Bits shifted off the left side of the element are shifted back in from the
 * right.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral rotate(Integral v, Integral i) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(rotate, Integral, v, i);
}

/** @brief Returns `x - y` and saturates the result.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
Integral sub_sat(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(sub_sat, Integral, x, y);
}

/** @brief Returns `result[i] = (hi[i] << 8) | lo[i]`
 * @tparam I1 must model geninteger. The integral size of I1 must be double the
 * integral size of I2.
 * @tparam I2 must model geninteger.
 */
template <typename I1, typename I2,
          COMPUTECPP_REQUIRES(
              computecpp::gsl::or_<
                  detail::builtin::is_geninteger8bit<I1>::value &&
                      detail::builtin::is_ugeninteger8bit<I2>::value,
                  detail::builtin::is_geninteger16bit<I1>::value &&
                      detail::builtin::is_ugeninteger16bit<I2>::value,
                  detail::builtin::is_geninteger32bit<I1>::value &&
                      detail::builtin::is_ugeninteger32bit<I2>::value>::value)>
auto upsample(I1 hi, I2 lo) noexcept
    -> decltype(::cl::sycl::detail::double_width_cast(std::declval<I1>())) {
  return COMPUTECPP_BUILTIN_INVOKE2(
      upsample,
      decltype(::cl::sycl::detail::double_width_cast(std::declval<I1>())), hi,
      lo);
}

/** @brief Returns the number of non-zero bits in `x`.
 * @tparam Integral must model geninteger.
 */
template <typename Integral,
          typename return_t = detail::collapse_swizzled_vec_t<Integral>,
          COMPUTECPP_REQUIRES(detail::builtin::is_geninteger<Integral>::value)>
return_t popcount(Integral x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(popcount, return_t, x);
}

/** @brief Multipy two 24-bit integer values `x` and `y` and add the 32-bit
 * integer result to the 32-bit integer `z`.
 *
 * Refer to definition of `mul24` to see how the 24-bit integer multiplication
 * is performed.
 * @tparam Integral must model geninteger32bit.
 */
template <
    typename Integral,
    COMPUTECPP_REQUIRES(detail::builtin::is_geninteger32bit<Integral>::value)>
Integral mad24(Integral x, Integral y, Integral z) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(mad24, Integral, x, y, z);
}

/** @brief Multiply two 24-bit integer values `x` and `y`. `x` and `y` are
 * 32-bit integers but only the low 24-bits are used to perform the
 * multiplication.
 *
 * `mul24` should only be used when values in `x` and `y` are in the range
 * [-223, 223 - 1] if `x` and `y` are signed integers and in the range [0, 224 -
 * 1] if `x` and `y` are unsigned integers. If `x` and `y` are not in this
 * range, the multiplication result is implementation-defined.
 * @tparam Integral must model geninteger32bit.
 */
template <
    typename Integral,
    COMPUTECPP_REQUIRES(detail::builtin::is_geninteger32bit<Integral>::value)>
Integral mul24(Integral x, Integral y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(mul24, Integral, x, y);
}
}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_INTEGRAL_H_
