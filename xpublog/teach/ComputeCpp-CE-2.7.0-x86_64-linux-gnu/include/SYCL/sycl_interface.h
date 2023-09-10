/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_interface.h

  @brief This file is an unified header file which includes all required
  header files for the sycl runtime interface.
 */

#ifndef RUNTIME_INCLUDE_SYCL_INTERFACE_H_
#define RUNTIME_INCLUDE_SYCL_INTERFACE_H_

// This has to be included first
#include "SYCL/common.h"

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/compiler_hooks.h"
#endif

#include "SYCL/accessor.h"
#include "SYCL/accessor/accessor_args.h"
#include "SYCL/accessor/accessor_base.h"
#include "SYCL/accessor/accessor_ops.h"
#include "SYCL/accessor/buffer_accessor.h"
#include "SYCL/accessor/host_accessor.h"
#include "SYCL/accessor/host_image_accessor.h"
#include "SYCL/accessor/image_accessor.h"
#include "SYCL/accessor/image_array_accessor.h"
#include "SYCL/accessor/local_accessor.h"
#include "SYCL/allocator.h"
#include "SYCL/apis.h"
#include "SYCL/atomic.h"
#include "SYCL/atomic_device.h"
#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/bit_cast.h"
#include "SYCL/buffer.h"
#include "SYCL/cl_types.h"
#include "SYCL/cl_vec_types.h"
#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/device_info.h"
#include "SYCL/device_selector.h"
#include "SYCL/device_selector_globals.h"
#include "SYCL/error.h"
#include "SYCL/event.h"
#include "SYCL/experimental/usm.h"
#include "SYCL/experimental/usm_definitions.h"
#include "SYCL/experimental/usm_wrapper.h"
#include "SYCL/feature_test_macros.h"
#include "SYCL/group.h"
#include "SYCL/group_base.h"
#include "SYCL/group_functions.h"
#include "SYCL/half_type.h"
#include "SYCL/id.h"
#include "SYCL/image.h"
#include "SYCL/index_array.h"
#include "SYCL/info.h"
#include "SYCL/item.h"
#include "SYCL/item_base.h"
#include "SYCL/kernel.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/nd_range_base.h"
#include "SYCL/platform.h"
#include "SYCL/private_memory.h"
#include "SYCL/program.h"
#include "SYCL/property.h"
#include "SYCL/queue.h"
#include "SYCL/range.h"
#include "SYCL/reduction.h"
#include "SYCL/sampler.h"
#include "SYCL/storage_mem.h"
#include "SYCL/stream.h"
#include "SYCL/sycl_language_version.h"
#include "SYCL/task.h"
#include "SYCL/vec.h"
#include "SYCL/vec_common.h"
#include "SYCL/vec_load_store.h"
#include "SYCL/vec_macros.h"
#include "SYCL/vec_swizzles.h"
#include "SYCL/vec_swizzles_impl.h"

// usm_wrapper needs to be tightly coupled with the runtime
// in order to work publicly
#include "SYCL/experimental/usm_wrapper.h"

// This has to be included last
#include "SYCL/postdefines.h"

#if defined(COMPUTECPP_DISABLE_SYCL_NAMESPACE_ALIAS) ||                        \
    ((SYCL_LANGUAGE_VERSION < 202000) &&                                       \
     ((defined(__GNUC__) && (__GNUC__ < 8) && !defined(__llvm__) &&            \
       !defined(__INTEL_COMPILER)) ||                                          \
      (defined(__clang_major__) && (__clang_major__ < 4) &&                    \
       (__clang_minor__ < 8)) ||                                               \
      (defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1900))))
// There is a bug in older compilers that introduces namespace ambiguities
#else
// Alias to allow easy porting to future versions of SYCL
namespace sycl = ::cl::sycl;
#endif  // __GNUC__ < 8

#if SYCL_LANGUAGE_VERSION >= 202001
// Global instance of init_sycl_language_version which will initialize the value
// of sycl_language_version with SYCL_LANGUAGE_VERSION to propogate the version
// of SYCL that the application is targeting.
inline cl::sycl::detail::init_sycl_language_version initSYCLLanguageVersion{};
#endif  // SYCL_LANGUAGE_VERSION

#endif  // RUNTIME_INCLUDE_SYCL_INTERFACE_H_
