/*****************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/**
  @file backend.h

  @brief This file contains the interface for enumerating backends
  and backend-specific functionality.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BACKEND_H_
#define RUNTIME_INCLUDE_SYCL_BACKEND_H_

#include "SYCL/include_opencl.h"

#define SYCL_BACKEND_OPENCL
#define SYCL_BACKEND_HOST

namespace cl {
namespace sycl {

/// Defines backends available in ComputeCpp
enum class backend {
  host,    ///< Native C++ device
  opencl,  ///< OpenCL device
};

/** Defines backend-specific type traits
 * @tparam Backend The backend to inspect
 * @note Added in SYCL 2020
 */
template <backend Backend>
class backend_traits {
 public:
  /** Used when converting from a backend specific object to a SYCL object
   * @tparam SyclType Type of the SYCL object
   */
  template <class SyclType>
  using input_type = void;

  /** Used when converting from a SYCL object to a backend specific object
   * @tparam SyclType Type of the SYCL object
   */
  template <class SyclType>
  using return_type = void;
};

namespace detail {

/** Used for defining @ref backend_traits for the OpenCL backend
 * @tparam SyclType Type of the SYCL object
 */
template <typename SyclType>
struct opencl_backend_traits {
  // Don't define any types by default
};

}  // namespace detail

/** Specializes backend traits for the OpenCL backend
 *
 * @tparam
 */
template <>
class backend_traits<backend::opencl> {
 public:
  /** Used when converting from an OpenCL object to a SYCL object
   * @tparam SyclType Type of the SYCL object
   */
  template <class SyclType>
  using input_type =
      typename detail::opencl_backend_traits<SyclType>::input_type;

  /** Used when converting from a SYCL object to an OpenCL object
   * @tparam SyclType Type of the SYCL object
   */
  template <class SyclType>
  using return_type =
      typename detail::opencl_backend_traits<SyclType>::return_type;
};

/** Shorthand for retrieving the input_type for a specific backend
 * @tparam Backend The backend to inspect
 * @tparam SyclType Type of the SYCL object
 * @note Added in SYCL 2020
 */
template <backend Backend, typename SyclType>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SyclType>;

/** Shorthand for retrieving the return_type for a specific backend
 * @tparam Backend The backend to inspect
 * @tparam SyclType Type of the SYCL object
 * @note Added in SYCL 2020
 */
template <backend Backend, typename SyclType>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SyclType>;

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_BACKEND_H_
