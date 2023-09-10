/*****************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/** @file addrspace_cast.h
 *
 * @brief This file contains the addrspace_cast function, for casting between
 * address spaces.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ADDRSPACE_CAST_H_
#define RUNTIME_INCLUDE_SYCL_ADDRSPACE_CAST_H_

/// @cond COMPUTECPP_DEV
#include "SYCL/base.h"
#include "SYCL/type_traits.h"
#include <type_traits>
#include <utility>

namespace cl {
namespace sycl {
namespace detail {
template <typename T>
struct strip_addrspace {
  using type = T;
};
#ifdef __SYCL_DEVICE_ONLY__
template <typename T>
struct strip_addrspace<__attribute__((opencl_private)) T> {
  using type = T;
};
template <typename T>
struct strip_addrspace<COMPUTECPP_CL_ASP_GLOBAL T> {
  using type = T;
};
template <typename T>
struct strip_addrspace<COMPUTECPP_CL_ASP_CONSTANT T> {
  using type = T;
};
template <typename T>
struct strip_addrspace<COMPUTECPP_CL_ASP_LOCAL T> {
  using type = T;
};
template <typename T>
struct strip_addrspace<COMPUTECPP_CL_ASP_SUBGROUP_LOCAL T> {
  using type = T;
};
#endif  // __SYCL_DEVICE_ONLY__

template <typename T>
using strip_addrspace_t = typename strip_addrspace<T>::type;

#ifdef __SYCL_DEVICE_ONLY__
COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
COMPUTECPP_HOST_CXX_DIAGNOSTIC(
    ignored "-Wincompatible-pointer-types-discards-qualifiers")
template <typename T, typename S>
inline T addrspace_cast(S src) {
  static_assert(std::is_pointer<S>::value,
                "Can only address space cast between pointers");
  static_assert(
      std::is_same<strip_addrspace_t<remove_pointer_t<T>>,
                   strip_addrspace_t<remove_pointer_t<S>>>::value,
      "Tried to do an addrspace cast between two different types in device "
      "code");
  return (T)src;
}

template <typename T, typename S>
inline T reinterpret_addrspace_cast(S src) {
  static_assert(std::is_pointer<S>::value,
                "Can only reintepret address space cast between pointers");
  return (T)src;
}
COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
#else   // Non-device code
/** Warning-safe way to cast between address spaces of pointers to the same val
 *
 * T and S must be pointers to the same type, but in possibly different address
 * spaces. This method returns `src` converted to the type of `T` without
 * changing its value.
 *
 * Both `S` and `T` must be pointers. In non-device code, `S` and `T` must be
 * the same type.
 *
 * This method does not work in Offload.
 *
 * @tparam S Source type
 * @tparam T Destination type
 */
template <typename T, typename S>
inline T addrspace_cast(S src) {
  static_assert(std::is_pointer<S>::value,
                "Can only address space cast between pointers");
  static_assert(std::is_same<S, T>::value,
                "Tried to address space cast between two different types in "
                "non-device code");
  return src;
}

/** Less strict version of addrspace_cast that also does a reinterpret_cast
 *
 * T and S must be pointers, but can be different types and address spaces. This
 * method returns `src` converted to the type of `T` without changing its value.
 *
 * This method does not work in Offload.
 *
 * @tparam S Source type
 * @tparam T Destination type
 */
template <typename T, typename S>
inline T reinterpret_addrspace_cast(S src) {
  static_assert(std::is_pointer<S>::value,
                "Can only reintepret address space cast between pointers");
  return reinterpret_cast<T>(src);
}
#endif  // __SYCL_DEVICE_ONLY__
}  // namespace detail
}  // namespace sycl
}  // namespace cl

/// COMPUTECPP_DEV @endcond

#endif  // RUNTIME_INCLUDE_SYCL_ADDRSPACE_CAST_H_

////////////////////////////////////////////////////////////////////////////////
