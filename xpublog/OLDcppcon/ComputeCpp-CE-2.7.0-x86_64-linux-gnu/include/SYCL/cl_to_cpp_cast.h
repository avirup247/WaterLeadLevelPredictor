/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/
#ifndef RUNTIME_INCLUDE_SYCL_CL_TO_CPP_CAST_H_
#define RUNTIME_INCLUDE_SYCL_CL_TO_CPP_CAST_H_

#include "SYCL/type_traits.h"
#include "SYCL/vec.h"
#include "computecpp/gsl/gsl"

namespace cl {
namespace sycl {
namespace detail {
/** @brief Converts an OpenCL type to a C++ type.
 * @tparam T The type to be converted to.
 * @tparam FromT The type to be converted from.
 * @param from The object to be converted.
 * @returns An object equivalent to `from` with type `T`.
 * @note Overload for cl::sycl::vec.
 */
template <typename T, typename FromT, int N>
T cl_to_cpp_cast(const ::cl::sycl::vec<FromT, N>& from) noexcept {
  return from.template as<T>();
}

/** @brief Converts an OpenCL type to a C++ type.
 * @tparam T The type to be converted to.
 * @tparam FromT The type to be converted from.
 * @param from The object to be converted.
 * @returns An object equivalent to `from` with type `T`.
 */
template <typename T, typename FromT>
T cl_to_cpp_cast(const FromT& from) {
  return static_cast<T>(from);
}
}  // namespace detail
}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_CL_TO_CPP_CAST_H_
