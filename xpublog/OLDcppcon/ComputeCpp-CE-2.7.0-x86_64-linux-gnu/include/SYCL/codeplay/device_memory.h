////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2002-2019 Codeplay Software Limited
//  All Rights Reserved.
//
//  ComputeCpp : SYCL 1.2.1 Implementation
//
////////////////////////////////////////////////////////////////////////////////

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_DEVICE_MEMORY_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_DEVICE_MEMORY_H_

#include "SYCL/cl_types.h"

namespace cl {
namespace sycl {

class device;

namespace codeplay {

/** @brief Represents different types of device memory
 */
enum class memory_type : size_t {
  global = 0,
  local = 1,
  onchip = 2,
};

/** @brief Retrieves the amount of memory
 *        that has been allocated on the device
 *        for the specified memory type
 * @param dev Device to query
 * @param memoryType Which memory pool to query
 * @return Amount of allocated memory in bytes
 */
COMPUTECPP_EXPORT cl::sycl::cl_ulong get_allocated_memory_size(
    const cl::sycl::device& dev, memory_type memoryType);

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_DEVICE_MEMORY_H_
