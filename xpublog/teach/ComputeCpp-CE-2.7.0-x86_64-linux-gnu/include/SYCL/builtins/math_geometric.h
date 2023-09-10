/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_math_geometric_builtins.h

  @brief Defines the public interface for the SYCL geometric math built-in
  functions. See
  Table 4.108 of SYCL Specification Version 1.2.1 for more details.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_GEOMETRIC_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_GEOMETRIC_H_

#include "SYCL/builtins/math_symbols.h"
#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/half_type.h"
#include "SYCL/type_traits.h"
#include "computecpp/gsl/gsl"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus::detail::geometric
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
/** @brief Returns the cross product of p0.xyz and p1.xyz.
 *
 * The w component of float4 result returned will be 0.0.
 * @tparam F must be half, float, or double.
 * @tparam N N == 3 or N == 4
 */
template <typename F, int N,
          COMPUTECPP_REQUIRES((detail::is_custom_half_type<F>::value ||
                               std::is_same<F, float>::value ||
                               std::is_same<F, double>::value) &&
                              (N == 3 || N == 4))>
vec<F, N> cross(const vec<F, N> p0, const vec<F, N> p1) noexcept {
  using vec_t = vec<F, N>;
  return COMPUTECPP_BUILTIN_INVOKE2(cross, vec_t, p0, p1);
}

namespace detail {
template <typename F>
using select_float_t = detail::conditional_t<
    detail::builtin::is_gengeohalf<F>::value, ::cl::sycl::half,
    detail::conditional_t<detail::builtin::is_gengeofloat<F>::value, float,
                          double>>;
}

/** @brief Compute dot product.
 * @tparam F must model gengeohalf, gengeofloat, gengeodouble.
 */
template <
    typename F1, typename F2,
    typename F = detail::select_float_t<detail::common_return_t<F1, F2>>,
    COMPUTECPP_REQUIRES((detail::builtin::is_gen_geo_anyfloat<F1>::value &&
                         detail::builtin::is_gen_geo_anyfloat<F2>::value))>
F dot(F1 p0, F2 p1) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(dot, F, p0, p1);
}

/** @brief Returns the distance between p0 and p1.
 *
 * This is calculated as `length(p0 - p1)`.
 * @tparam F must model gengeohalf, gengeofloat, gengeodouble.
 */
template <
    typename F1, typename F2,
    typename F = detail::select_float_t<detail::common_return_t<F1, F2>>,
    COMPUTECPP_REQUIRES((detail::builtin::is_gen_geo_anyfloat<F1>::value &&
                         detail::builtin::is_gen_geo_anyfloat<F2>::value))>
F distance(F1 p0, F2 p1) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(distance, F, p0, p1);
}

/** @brief Return the length of vector p, i.e. `sqrt((p.x * p.x) + (p.y * p.y) +
 * ...)`
 * @tparam F must model gengeohalf, gengeofloat, gengeodouble.
 */
template <typename F,
          typename return_t =
              detail::select_float_t<detail::collapse_swizzled_vec_t<F>>,
          COMPUTECPP_REQUIRES(detail::builtin::is_gen_geo_anyfloat<F>::value)>
return_t length(F p) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(length, return_t, p);
}

/** @brief Returns a vector in the same direction as p but with a length of 1.
 * @tparam F must model gengeohalf, gengeofloat, gengeodouble.
 */
template <typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
          COMPUTECPP_REQUIRES(detail::builtin::is_gen_geo_anyfloat<F>::value)>
return_t normalize(F p) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(normalize, return_t, p);
}

/** @brief Returns f ast `length(p0 - p1)`.
 * @tparam F must model gengeofloat.
 */
template <typename F1, typename F2,
          typename F = detail::scalar_t<detail::common_return_t<F1, F2>>,
          COMPUTECPP_REQUIRES(
              (computecpp::gsl::or_<
                  detail::builtin::is_gengeohalf<F1>::value,
                  detail::builtin::is_gengeofloat<F1>::value>::value) &&
              (computecpp::gsl::or_<
                  detail::builtin::is_gengeohalf<F2>::value,
                  detail::builtin::is_gengeofloat<F2>::value>::value))>
F fast_distance(F1 p0, F2 p1) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE2(fast_distance, F, p0, p1);
}

/** @brief Returns the length of vector p computed as: `sqrt((half)(pow(p.x,2) +
 * pow(p.y,2)+ ...))`.
 * @tparam F must model gengeofloat.
 */
template <
    typename F,
    typename return_t = detail::scalar_t<detail::collapse_swizzled_vec_t<F>>,
    COMPUTECPP_REQUIRES(
        computecpp::gsl::or_<detail::builtin::is_gengeohalf<F>::value,
                             detail::builtin::is_gengeofloat<F>::value>::value)>
return_t fast_length(F p) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(fast_length, return_t, p);
}

/** @brief Returns a vector in the same direction as p but with a length of 1.
 * fast_normalize is computed as: `p * rsqrt((half)(pow(p.x, 2) + pow(p.y, 2) +
 * ...))`.
 *
 * The result shall be within 8192 ulps error from the infinitely precise
 * result of
 *
```
if (all (p == 0.0f))
   result = p;
else
result = p / sqrt (pow(p.x,2)+ pow(p.y,2)+ ... );
```

 * with the following exceptions:
 *
 * 1. If the sum of squares is greater than FLT_MAX then the value of the
floating-point values in the result vector are undefined.
 * 2. If the sum of squares is less than FLT_MIN then the implementation may
return back p.
 * 3. If the device is in "denorms are flushed to zero" mode, individual operand
elements with magnitude less than sqrt(FLT_MIN) may be flushed to zero before
proceeding with the calculation.
 * @tparam F must model gengeofloat.
 */
template <
    typename F, typename return_t = detail::collapse_swizzled_vec_t<F>,
    COMPUTECPP_REQUIRES(
        computecpp::gsl::or_<detail::builtin::is_gengeohalf<F>::value,
                             detail::builtin::is_gengeofloat<F>::value>::value)>
return_t fast_normalize(F p) noexcept {
  return COMPUTECPP_BUILTIN_INVOKE1(fast_normalize, return_t, p);
}

}  // namespace sycl
}  // namespace cl

#if !defined(__SYCL_DEVICE_ONLY__)
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // !defined(__SYCL_DEVICE_ONLY__)

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_MATH_GEOMETRIC_H_
