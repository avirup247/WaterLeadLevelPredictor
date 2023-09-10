/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file device_selector_globals.h

  @brief This file contains the API for the @ref cl::sycl::device_selector class
*/
#ifndef RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_GLOBALS_H_
#define RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_GLOBALS_H_

#include "SYCL/device_selector.h"
#include "SYCL/host_compiler_macros.h"
#include "SYCL/predefines.h"

namespace cl {
namespace sycl {

#if SYCL_LANGUAGE_VERSION >= 202001

COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wexit-time-destructors")

/** @brief Global instance of default_selector.
 */
inline default_selector default_selector_v{};

/** @brief Global instance of host_selector.
 */
inline host_selector host_selector_v{};

/** @brief Global instance of host_selector.
 */
inline cpu_selector cpu_selector_v{};

/** @brief Global instance of gpu_selector.
 */
inline gpu_selector gpu_selector_v{};

/** @brief Global instance of accelerator_selector.
 */
inline accelerator_selector accelerator_selector_v{};

/** @brief Global instance of intel_selector.
 */
inline intel_selector intel_selector_v{};

/** @brief Global instance of amd_selector.
 */
inline amd_selector amd_selector_v{};

COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_GLOBALS_H_
