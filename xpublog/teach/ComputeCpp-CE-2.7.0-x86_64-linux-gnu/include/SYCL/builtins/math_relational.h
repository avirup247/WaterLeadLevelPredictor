/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_math_relational_builtins.h

  @brief Defines the public interface for the SYCL common math built-in
  functions. See
  Table 4.108 of SYCL Specification Version 1.2.1 for more details.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_RELATIONAL_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_RELATIONAL_H_

#include "SYCL/builtins/math_symbols.h"
#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/meta.h"
#include "SYCL/type_traits.h"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus::detail::relational
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
/** @brief Returns the component-wise compare of `x == y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isequal(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isequal, detail::matching_integral_t<F>, x,
                                    y);
}

/** @brief Returns the component-wise compare of `x != y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isnotequal(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isnotequal, detail::matching_integral_t<F>,
                                    x, y);
}

/** @brief Returns the component-wise compare of `x > y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <
    typename F1, typename F2,
    typename F = detail::matching_integral_t<detail::common_return_t<F1, F2>>,
    COMPUTECPP_REQUIRES((detail::builtin::is_genfloat<F1>::value &&
                         detail::builtin::is_genfloat<F2>::value))>
F isgreater(F1 x, F2 y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isgreater, F, x, y);
}

/** @brief Returns the component-wise compare of `x >= y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isgreaterequal(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isgreaterequal,
                                    detail::matching_integral_t<F>, x, y);
}

/** @brief Returns the component-wise compare of `x < y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isless(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isless, detail::matching_integral_t<F>, x,
                                    y);
}

/** @brief Returns the component-wise compare of `x <= y`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> islessequal(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(islessequal, detail::matching_integral_t<F>,
                                    x, y);
}

/** @brief Returns the component-wise compare of `(x < y) || (x > y)`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> islessgreater(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(islessgreater,
                                    detail::matching_integral_t<F>, x, y);
}

/** @brief Test for finite value.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isfinite(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(isfinite, detail::matching_integral_t<F>,
                                    x);
}

/** @brief Test for infinity value (positive or negative).
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          typename return_t =
              detail::matching_integral_t<detail::collapse_swizzled_vec_t<F>>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t isinf(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(isinf, return_t, x);
}

/** @brief Test for a NaN.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          typename return_t =
              detail::matching_integral_t<detail::collapse_swizzled_vec_t<F>>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t isnan(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(isnan, return_t, x);
}

/** @brief Test for a normal value.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isnormal(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(isnormal, detail::matching_integral_t<F>,
                                    x);
}

/** @brief Test if arguments are ordered.
 *
 * isordered() takes arguments x and y, and returns the result `isequal(x, x) &&
 * isequal(y, y)`.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isordered(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isordered, detail::matching_integral_t<F>,
                                    x, y);
}

/** @brief Test if arguments are unordered.
 *
 * isunordered() takes arguments x and y, returning non-zero if x or y is NaN,
 * and zero otherwise.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
detail::matching_integral_t<F> isunordered(F x, F y) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(isunordered, detail::matching_integral_t<F>,
                                    x, y);
}

/** @brief Test for sign bit. The scalar version of the function returns a 1 if
 * the sign bit in the float is set else returns 0.
 *
 * The vector version of the function returns the following for each component
 * in floatn: -1 (i.e all bits set) if the sign bit in the float is set else
 * returns 0.
 * @tparam F must model genfloatf or genfloatd.
 */
template <typename F,
          typename return_t =
              detail::matching_integral_t<detail::collapse_swizzled_vec_t<F>>,
          COMPUTECPP_REQUIRES(detail::builtin::is_genfloat<F>::value)>
return_t signbit(F x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(signbit, return_t, x);
}

/** @brief Returns 1 if the most significant bit in any component of x is set;
 * otherwise returns 0.
 * @tparam Integral must model igeninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_igeninteger<Integral>::value)>
int any(Integral x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(any, int, x);
}

/** @brief Returns 1 if the most significant bit in all components of x is set;
 * otherwise returns 0.
 * @tparam Integral must model igeninteger.
 */
template <typename Integral,
          COMPUTECPP_REQUIRES(detail::builtin::is_igeninteger<Integral>::value)>
int all(Integral x) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(all, int, x);
}

/** Each bit of the result is the corresponding bit of a if the corresponding
 * bit of c is 0.
 *
 * Otherwise it is the corresponding bit of b.
 * @tparam T must model gentype.
 */
template <typename T,
          COMPUTECPP_REQUIRES(detail::builtin::is_gentype<T>::value)>
T bitselect(T a, T b, T c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(bitselect, T, a, b, c);
}

/** @brief For each component of a vector type: `result[i] = (MSB of c[i] is
 * set)? b[i] : a[i]`
 * For a scalar type: `result = c ? b : a`.
 * geninteger must have the same number of elements and bits as gentype.
 *
 * @tparam T1 must model geninteger, genfloatf, or genfloatd.
 * @tparam T2 If:
 *
 * - T1 models geninteger, then T2 must model either igeninteger or ugeninteger.
 * - T1 models genfloatf, then T2 must model either igenint or ugenint.
 * - T1 models genfloatd, then T2 must model either igeninteger64bit or
 * ugeninteger64bit.
 */
template <typename T1, typename T2,
          COMPUTECPP_REQUIRES(
              computecpp::gsl::or_<
                  detail::builtin::is_geninteger<T1>::value &&
                      detail::builtin::is_igeninteger<T2>::value,
                  detail::builtin::is_geninteger<T1>::value &&
                      detail::builtin::is_ugeninteger<T2>::value,
                  detail::builtin::is_genfloatf<T1>::value &&
                      detail::builtin::is_genint<T2>::value,
                  detail::builtin::is_genfloatf<T1>::value &&
                      detail::builtin::is_ugenint<T2>::value,
                  detail::builtin::is_genfloatd<T1>::value &&
                      detail::builtin::is_igeninteger64bit<T2>::value,
                  detail::builtin::is_genfloatd<T1>::value &&
                      detail::builtin::is_ugeninteger64bit<T2>::value>::value)>
T1 select(T1 a, T1 b, T2 c) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE3(select, T1, a, b, c);
}

template <
    typename T1, typename T2,
    COMPUTECPP_REQUIRES(computecpp::gsl::or_<
                        detail::builtin::is_genfloath<T1>::value &&
                            detail::builtin::is_genshort<T2>::value,
                        detail::builtin::is_genfloath<T1>::value &&
                            detail::builtin::is_ugenshort<T2>::value>::value)>
T1 select(T1 a, T1 b, T2 c) noexcept {
  return ::cl::sycl::detail::halve_width_cast(
      ::cl::sycl::select(::cl::sycl::detail::double_width_cast(a),
                         ::cl::sycl::detail::double_width_cast(b),
                         ::cl::sycl::detail::double_width_cast(c)));
}

}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_RELATIONAL_H_
