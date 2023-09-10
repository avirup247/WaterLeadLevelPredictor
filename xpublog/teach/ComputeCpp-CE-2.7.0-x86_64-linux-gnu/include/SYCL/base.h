/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file base.h

  @brief This file contains functions and forward declarations of functions used
  throughout the runtime.
*/
#ifndef RUNTIME_INCLUDE_SYCL_BASE_H_
#define RUNTIME_INCLUDE_SYCL_BASE_H_

#include "SYCL/common.h"

namespace cl {
namespace sycl {

///////////////////////////////////////////////////////////////////////////////
// Forward declarations
///////////////////////////////////////////////////////////////////////////////

class image_mem;

class sampler;

namespace detail {
class nd_range_base;
class nd_item_base;
class group_base;
}  // namespace detail

template <int dimensions>
class nd_range;

template <int dimensions, bool with_offset>
class item;

template <int dimensions>
class nd_item;

template <int dimensions>
class h_item;

template <int dimensions>
class group;

class task_functor;

class queue;

class context;

class kernel;

class storage_mem;

class image_storage;

class program;

////////////////////////////////////////////////////////////////////////////////
// Convenience aliases

namespace detail {
class kernel;
class program;
class context;
class event;
class queue;
class storage_mem;
class sampler;
class device;
class accessor;
class transaction;
class platform;
class property_base;
class device_storage;
class usm_allocator_detail;
}  // namespace detail

/// @cond COMPUTECPP_DEV
using dkernel_shptr = shared_ptr_class<detail::kernel>;
using dprogram_shptr = shared_ptr_class<detail::program>;
using dcontext_shptr = shared_ptr_class<detail::context>;
using dqueue_shptr = shared_ptr_class<detail::queue>;
using dmem_shptr = shared_ptr_class<detail::storage_mem>;
using dsampler_shptr = shared_ptr_class<detail::sampler>;
using ddevice_shptr = shared_ptr_class<detail::device>;
using dplatform_shptr = shared_ptr_class<detail::platform>;
using daccessor_shptr = shared_ptr_class<detail::accessor>;
using devent_shptr = shared_ptr_class<detail::event>;
using dprogram_wkptr = std::weak_ptr<detail::program>;
using dqueue_wkptr = std::weak_ptr<detail::queue>;
using dcontext_wkptr = std::weak_ptr<detail::context>;
using ddevice_wkptr = std::weak_ptr<detail::device>;
using dproperty_shptr = shared_ptr_class<detail::property_base>;
using dtrans_uptr = unique_ptr_class<detail::transaction>;
using dusm_alloc_shptr = shared_ptr_class<detail::usm_allocator_detail>;
namespace detail {
using ddev_storage_shptr = shared_ptr_class<detail::device_storage>;
}
/// COMPUTECPP_DEV @endcond

////////////////////////////////////////////////////////////////////////////////
// Common header for the arguments
namespace detail {

template <class T, bool value>
struct enable_if;

template <class T>
struct enable_if<T, true> {
  using type = T;
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
namespace access {
/** @brief Memory fence descriptor.
 * Values are taken from SPIR specification.
 */
enum class fence_space : cl_uint {
  local_space = 1,  /**< Preform all initiated memory operation on local memory
                   before the operation */
  global_space = 2, /**< Preform all initiated memory operation on global memory
                   before the operation */
  global_and_local = 3 /**< Preform all initiated memory operation on local and
                     global memory before the operation */
};                     // enum class fence_space

/** @brief Address space descriptors
 */
enum class address_space : int {
  private_space = 0,        /**< OpenCL Private memory */
  global_space = 1,         /**< OpenCL Global memory */
  constant_space = 2,       /**< OpenCL Constant memory */
  local_space = 3,          /**< OpenCL Local memory */
  subgroup_local_space = 9, /**< Sub-group local memory extension */
};                          // enum class address_space
}  // namespace access

/// @cond COMPUTECPP_DEV
#ifdef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_CL_ASP_PRIVATE
#define COMPUTECPP_CL_ASP_GLOBAL __attribute__((opencl_global))
#define COMPUTECPP_CL_ASP_CONSTANT __attribute__((opencl_constant))
#define COMPUTECPP_CL_ASP_LOCAL __attribute__((opencl_local))

// ASP has a special attribute for subgroup local
#ifdef __SYCL_COMPUTECPP_ASP__
#define COMPUTECPP_CL_ASP_SUBGROUP_LOCAL                                       \
  __attribute__((codeplay_sycl_subgroup_local))
#else  // Offload
#define COMPUTECPP_CL_ASP_SUBGROUP_LOCAL COMPUTECPP_CL_ASP_ADDRESS(9)
#endif

#define COMPUTECPP_CL_ASP_ADDRESS(space) __attribute__((address_space(space)))
#else  // __SYCL_DEVICE_ONLY__
#define COMPUTECPP_CL_ASP_PRIVATE
#define COMPUTECPP_CL_ASP_GLOBAL
#define COMPUTECPP_CL_ASP_CONSTANT
#define COMPUTECPP_CL_ASP_LOCAL
#define COMPUTECPP_CL_ASP_SUBGROUP_LOCAL
#define COMPUTECPP_CL_ASP_ADDRESS(space)
#endif  // __SYCL_DEVICE_ONLY__
/// COMPUTECPP_DEV @endcond

namespace detail {

/** @brief Performs a work-group barrier on the host synchronizing with all
 * work-items in the current work-group.
 * Note that this relies on having a barrier per global thread,
 * and each global thread is accessed via the static variable
 * in host_runtime.
 * @param itm The nd_item describing the currently executing work-group, to
 * which the barrier will be aplied.
 */
COMPUTECPP_EXPORT void host_barrier(detail::nd_item_base itm);

/** @brief Performs a work-group barrier on the host synchronizing with all
 * work-items in the current work-group.
 * Note that this relies on having a barrier per global thread,
 * and each global thread is accessed via the static variable
 * in host_runtime.
 * @param grp The group describing the currently executing work-group, to which
 * the barrier will be aplied.
 */
COMPUTECPP_EXPORT void host_barrier(detail::group_base grp);

/** @brief Executes a mem_fence operation on the host
 * @param accessMode Specifies whether all load (access::mode::read),
 *         store (access::mode::write) or both load and store memory accesses
 *         (access::mode::read_write) in the specified address space issued
 *         before the mem-fence should complete before those issued
 *         after the mem-fence.
 */
COMPUTECPP_EXPORT void host_mem_fence(access::mode accessMode);

/** @brief Converts a fence space enum object into the low-level SPIR value
 * @param fenceSpace Value specifying the type of fence to apply
 * @return Equivalent SPIR 1.2 value for the given enum class entry.
 */
inline constexpr cl_uint get_cl_mem_fence_flag(access::fence_space fenceSpace) {
  return static_cast<cl_uint>(fenceSpace);
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_BASE_H_

////////////////////////////////////////////////////////////////////////////////
