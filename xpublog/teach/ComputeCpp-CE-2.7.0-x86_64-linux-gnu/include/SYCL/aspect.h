/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp.

*******************************************************************************/

/**
  @file aspect.h

  @brief This file contains the aspect enum class.
*/
#ifndef RUNTIME_INCLUDE_SYCL_ASPECT_H_
#define RUNTIME_INCLUDE_SYCL_ASPECT_H_

#include "SYCL/predefines.h"

namespace cl {
namespace sycl {

/** @brief Enumerates the aspects which can be queried on a @ref platform or
 * @ref device.
 */
enum class aspect_impl {
  host COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use host_debuggable instead"),
  cpu,
  gpu,
  accelerator,
  custom,
  fp16,
  fp64,
  int64_base_atomics COMPUTECPP_DEPRECATED_BY_SYCL_202001(
      "Use atomic64 instead"),
  int64_extended_atomics COMPUTECPP_DEPRECATED_BY_SYCL_202001(
      "Use atomic64 instead"),
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_shared_allocations,
  usm_restricted_shared_allocations,
  usm_system_allocator COMPUTECPP_DEPRECATED_BY_SYCL_202001(
      "Use usm_system_allocations instead"),
  emulated,
  host_debuggable,
  atomic64,
  usm_atomic_host_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations,
};

#if SYCL_LANGUAGE_VERSION >= 202001

/** @brief Enumerates the aspects which can be queried on a @ref platform or
 * @ref device.
 */
using aspect = aspect_impl;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ASPECT_H_
