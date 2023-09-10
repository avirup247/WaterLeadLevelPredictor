/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_DEFINITIONS_H_
#define RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_DEFINITIONS_H_

namespace cl {
namespace sycl {

COMPUTECPP_INLINE_EXPERIMENTAL
namespace experimental {

namespace usm {

/** @brief Enum specifying the type of a USM allocation
 */
enum class alloc {
  host,     //< The memory is available on host
  device,   //< The memory is available on a device
  shared,   //< The memory is shared between host and device
  unknown,  //< Couldn't determine the type of allocation
};
}  // namespace usm

}  // namespace experimental
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_DEFINITIONS_H_
