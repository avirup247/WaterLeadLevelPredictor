/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_math_builtin_symbols.h

  @brief Provides symbols that are common to all maths built-ins.
*/
#ifndef RUNTIME_INCLUDE_SYCL_SYCL_MATH_BUILTIN_SYMBOLS_H_
#define RUNTIME_INCLUDE_SYCL_SYCL_MATH_BUILTIN_SYMBOLS_H_

#include "SYCL/cl_to_cpp_cast.h"
#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/predefines.h"
#include "SYCL/type_traits.h"
#include "computecpp/gsl/gsl"

#define COMPUTECPP_REQUIRES_IMPL(B) detail::enable_if_t<B, int> = 0
#define COMPUTECPP_REQUIRES(...) COMPUTECPP_REQUIRES_IMPL((__VA_ARGS__))

#if defined(__SYCL_DEVICE_ONLY__)
#include "SYCL/builtins/device_builtins.h"
#define COMPUTECPP_DETAIL_NAMESPACE ::cl::sycl::detail
#define COMPUTECPP_CPP_TO_CL(x) ::cl::sycl::detail::cpp_to_cl_cast(x)
#else
#include "SYCL/abacus_all.h"
#define COMPUTECPP_CPP_TO_CL(x)                                                \
  ::cl::sycl::detail::sycl_to_abacus(::cl::sycl::detail::cpp_to_cl_cast(x))

namespace cl {
namespace sycl {
namespace detail {

template <typename T>
constexpr abacus::sycl_to_abacus_t<T>* sycl_to_abacus(T* p) {
  return ::computecpp::gsl::pun_cast<abacus::sycl_to_abacus_t<T>*>(p);
}

/** @brief Converts a SYCL type to the corresponding Abacus type.
 * @tparam The type to be converted.
 * @param The object to be converted.
 * @return The Abacus equivalent of t.
 * @note For internal use only.
 */
template <typename T>
constexpr abacus::sycl_to_abacus_t<T> sycl_to_abacus(const T& t) noexcept {
  return static_cast<abacus::sycl_to_abacus_t<T>>(t);
}

/** @brief Converts a SYCL type to the corresponding Abacus type.
 * @tparam The type to be converted.
 * @param The object to be converted.
 * @return The Abacus equivalent of t.
 * @note For internal use only.
 */
template <typename T, int N>
abacus::sycl_to_abacus_t<cl::sycl::vec<T, N>> sycl_to_abacus(
    const cl::sycl::vec<T, N>& v) noexcept {
  return computecpp::gsl::pun_cast<
      abacus::sycl_to_abacus_t<cl::sycl::vec<T, N>>&>(v);
}

/** @brief Converts a SYCL type to the corresponding Abacus type.
 * @tparam The type to be converted.
 * @param The object to be converted.
 * @return The Abacus equivalent of t.
 * @note For internal use only.
 */
template <int N>
constexpr abacus::sycl_to_abacus_t<cl::sycl::vec<cl::sycl::cl_float, N>>
sycl_to_abacus(const cl::sycl::vec<cl::sycl::half, N>& v) noexcept {
  return sycl_to_abacus(
      v.template convert<cl::sycl::cl_float,
                         cl::sycl::rounding_mode::automatic>());
}
}  // namespace detail
}  // namespace sycl
}  // namespace cl
#endif  // defined(__SYCL_DEVICE_ONLY__)

namespace cl {
namespace sycl {
namespace detail {
/** @brief Determines the matching integral type for a given genfloat.
 * @tparam F must model genfloat.
 * @note For internal use only.
 */
template <typename>
struct matching_integral;

template <>
struct matching_integral<cl_half> {
  using type = cl_short;
};

template <int N>
struct matching_integral<::cl::sycl::vec<cl_half, N>> {
  using type = ::cl::sycl::vec<cl_short, N>;
};

template <>
struct matching_integral<cl_float> {
  using type = cl_int;
};

template <int N>
struct matching_integral<::cl::sycl::vec<cl_float, N>> {
  using type = ::cl::sycl::vec<cl_int, N>;
};

template <>
struct matching_integral<cl_double> {
  using type = cl_long;
};

template <int N>
struct matching_integral<::cl::sycl::vec<cl_double, N>> {
  using type = ::cl::sycl::vec<cl_long, N>;
};

template <typename T>
using matching_integral_t = typename matching_integral<T>::type;

template <typename T>
struct scalar {
  using type = T;
};

template <typename T, int N>
struct scalar<cl::sycl::vec<T, N>> {
  using type = T;
};

template <typename T, int kElems, int... Indexes>
struct scalar<cl::sycl::swizzled_vec<T, kElems, Indexes...>> {
  using type = T;
};

template <typename T>
using scalar_t = typename scalar<T>::type;
}  // namespace detail
}  // namespace sycl
}  // namespace cl

/** @brief Implements invocation for SYCL built-in functions.
 *
 * Due to the fact that the host functions are contained in a separate namespace
 * to the SYCL device functions, but the remainder of the functionality is
 * consistent for each function call, this is kept as a macro.
 *
 * @param f The name of the SYCL built-in function to be invoked.
 * @param T The expected return type of the built-in function being invoked.
 * @param __VA_ARGS__ The parameters for f.
 * @return The return value for f.
 * @note For internal use only.
 */
#define COMPUTECPP_BUILTIN_INVOKE_IMPL(f, T, ...)                              \
  ::cl::sycl::detail::cl_to_cpp_cast<T>(                                       \
      COMPUTECPP_DETAIL_NAMESPACE::f(__VA_ARGS__))

/** @brief Implements invocation for SYCL built-in functions with exactly one
 * parameter.
 *
 * Dispatch is made to the correct host or device function from here. Conversion
 * from the C++ type to the corresponding SYCL type is performed automatically.
 *
 * @param f The name of the SYCL built-in function to be invoked.
 * @param T The expected return type of the built-in function being invoked.
 * @param x The parameter for f.
 * @return The return value for f.
 * @note For internal use only.
 */
#define COMPUTECPP_BUILTIN_INVOKE1(f, T, x)                                    \
  COMPUTECPP_BUILTIN_INVOKE_IMPL(f, T, COMPUTECPP_CPP_TO_CL(x))

/** @brief Implements invocation for SYCL built-in functions with exactly two
 * parameters.
 *
 * Dispatch is made to the correct host or device function from here. Conversion
 * from the C++ type to the corresponding SYCL type is performed automatically.
 *
 * @param f The name of the SYCL built-in function to be invoked.
 * @param T The expected return type of the built-in function being invoked.
 * @param x The first parameter for f.
 * @param y The second parameter for f.
 * @return The return value for f.
 * @note For internal use only.
 */
#define COMPUTECPP_BUILTIN_INVOKE2(f, T, x, y)                                 \
  COMPUTECPP_BUILTIN_INVOKE_IMPL(f, T, COMPUTECPP_CPP_TO_CL(x),                \
                                 COMPUTECPP_CPP_TO_CL(y))

/** @brief Implements invocation for SYCL built-in functions with exactly three
 * parameters.
 *
 * Dispatch is made to the correct host or device function from here. Conversion
 * from the C++ type to the corresponding SYCL type is performed automatically.
 *
 * @param f The name of the SYCL built-in function to be invoked.
 * @param T The expected return type of the built-in function being invoked.
 * @param x The first parameter for f.
 * @param y The second parameter for f.
 * @param z The third parameter for f.
 * @return The return value for f.
 * @note For internal use only.
 */
#define COMPUTECPP_BUILTIN_INVOKE3(f, T, x, y, z)                              \
  COMPUTECPP_BUILTIN_INVOKE_IMPL(f, T, COMPUTECPP_CPP_TO_CL(x),                \
                                 COMPUTECPP_CPP_TO_CL(y),                      \
                                 COMPUTECPP_CPP_TO_CL(z))

#endif  // RUNTIME_INCLUDE_SYCL_SYCL_MATH_BUILTIN_SYMBOLS_H_
