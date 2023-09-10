/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp


*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_COMPILER_HOOKS_H_
#define RUNTIME_INCLUDE_SYCL_COMPILER_HOOKS_H_

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/builtins/device_builtins.h"
#endif  // __SYCL_DEVICE_ONLY__

#include "SYCL/common.h"
#include "SYCL/group_base.h"
#include "SYCL/host_compiler_macros.h"
#include "SYCL/id.h"
#include "SYCL/item.h"

/**
  @file compiler_hooks.h
  @brief Internal file used by runtime to implement the kernel invoking APIs for
  the device.
*/
namespace cl {
namespace sycl {
namespace detail {

/** Type of the kernel interop get_*_id function pointers
 */
using get_id_f = detail::size_type (*)(cl::sycl::cl_uint);

/** Type of the kernel interop get_*_size function pointers
 */
using get_range_f = get_id_f;

/** @brief Helper for constructing an id or range
 *        based on the number of dimensions.
 * @tparam dims Number of dimensions
 */
template <int dims>
struct index_array_generator;

/** @brief Helper for constructing a 1D id or range
 */
template <>
struct index_array_generator<1> {
  template <get_id_f getter>
  static id<1> get_id() noexcept {
    return {getter(0)};
  }
  template <get_range_f getter>
  static range<1> get_range() noexcept {
    return {getter(0)};
  }
};

/** @brief Helper for constructing a 2D id or range
 * @note The 1st and 2nd dimensions are flipped
 *       in order to align the OpenCL iteration space
 *       with the row major data layout.
 */
template <>
struct index_array_generator<2> {
  template <get_id_f getter>
  static id<2> get_id() noexcept {
    return {getter(1), getter(0)};
  }
  template <get_range_f getter>
  static range<2> get_range() noexcept {
    return {getter(1), getter(0)};
  }
};

/** @brief Helper for constructing a 3D id or range
 * @note The 1st and 3nd dimensions are flipped
 *       in order to align the OpenCL iteration space
 *       with the row major data layout.
 */
template <>
struct index_array_generator<3> {
  template <get_id_f getter>
  static id<3> get_id() noexcept {
    return {getter(2), getter(1), getter(0)};
  }
  template <get_range_f getter>
  static range<3> get_range() noexcept {
    return {getter(2), getter(1), getter(0)};
  }
};

#ifdef __SYCL_DEVICE_ONLY__

#ifdef __SYCL_COMPUTECPP_ASP__
#define COMPUTECPP_SYCL_DEVFUNC
#else  // Offload
#define COMPUTECPP_SYCL_DEVFUNC __attribute__((__offload__))
#endif

/** @brief Wraps the get_local_id kernel interop function
 * @param dim Dimension to query
 * @return Local ID of the `dim` dimension
 * @note Required because of a symbol conflict
 */
inline detail::size_type get_local_id_helper(cl::sycl::cl_uint dim) {
  return detail::get_local_id(dim);
}

// Disable some warnings in case -Werror is used
// These warnings only appear in this file
COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wsign-compare")

/** Define the device_call functions for ComputeCpp device compiler.
 * In this case, we are defining the functions purely to be
 * the root of the kernel call-stack.
 */

/** Kernel generation for the single task API entry.
 * @param functorT Functor containing the kernel.
 */
template <typename kernelT, typename functorT>
__attribute__((sycl_kernel(kernelT))) COMPUTECPP_SYCL_DEVFUNC void
kernelgen_single_task(functorT functor) {
  functor();
}

template <class kernelT, class functorT, int dims>
__attribute__((sycl_kernel(kernelT))) COMPUTECPP_SYCL_DEVFUNC void
kernelgen_parallel_for_nd(functorT functor) {
  const auto globalID =
      index_array_generator<dims>::template get_id<get_global_id>();
  const auto localID =
      index_array_generator<dims>::template get_id<get_local_id_helper>();
  const auto groupID =
      index_array_generator<dims>::template get_id<get_group_id>();

  const auto globalRange =
      index_array_generator<dims>::template get_range<get_global_size>();
  const auto localRange =
      index_array_generator<dims>::template get_range<get_local_size>();
  const auto groupRange =
      index_array_generator<dims>::template get_range<get_num_groups>();

  const auto globalOffset =
      index_array_generator<dims>::template get_id<get_global_offset>();

  auto ndItemID = nd_item<dims>{
      detail::nd_item_base(localID, globalID, localRange, globalRange,
                           globalOffset, groupID, groupRange)};
  functor(ndItemID);
}

/** Kernel generation for the parallel_for_id API entry.
 * @param functorT Functor containing the kernel.
 */
template <class kernelT, class functorT, int dims>
__attribute__((sycl_kernel(kernelT))) COMPUTECPP_SYCL_DEVFUNC void
kernelgen_parallel_for_id(functorT functor) {
  const auto globalID =
      index_array_generator<dims>::template get_id<get_global_id>();
  const auto globalRange =
      index_array_generator<dims>::template get_range<get_global_size>();
  auto itemID = item<dims>{detail::item_base(globalID, globalRange)};
  functor(itemID);
}

/**
Check functions that are inserted by the compiler before and after local and
global stores in functions qualified with the address_space_of_locals attribute.
*/
extern "C" {
/**
@brief Function called before entering a hierarchical critical region. Returns
 true if the linear local id is 0.
@return boolean specifying if the linear local id is 0.
*/
COMPUTECPP_SYCL_DEVFUNC inline bool __computecpp_access_hierarchical_region_() {
  return !(detail::get_local_id(0) | detail::get_local_id(1) |
           detail::get_local_id(2));
}

/** @brief Hierarchical critical region merge function requiring a local mem
 * fence.
 */
COMPUTECPP_SYCL_DEVFUNC inline void
__computecpp_merge_hierarchical_local_region_() {
  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(
      access::fence_space::local_space));
}

/** @brief Hierarchical critical region merge function requiring a global mem
 * fence.
 */
COMPUTECPP_SYCL_DEVFUNC inline void
__computecpp_merge_hierarchical_global_region_() {
  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(
      access::fence_space::global_space));
}

/** @brief Hierarchical critical region merge function requiring a global and
 * local mem fence.
 */
COMPUTECPP_SYCL_DEVFUNC inline void
__computecpp_merge_hierarchical_global_local_region_() {
  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(
      access::fence_space::global_and_local));
}
}

#define COMPUTECPP_ASP_OPENCL_LOCAL 2

/** Kernel generation for the parallel_for_work_group API entry.
 * @param functorT Functor containing the kernel.
 */
#if __SYCL_COMPUTECPP_ASP__
// ASP does not yet support hierarchical, put a placeholder here until it does
// SYCLE-2257 for hierarchical support
template <class kernelT, class functorT, int dims>
__attribute__((sycl_kernel(kernelT))) void
#else  // Offload
template <class kernelT, class functorT, int dims>
__attribute__((sycl_kernel(kernelT, 2))) __attribute__((__offload__(2)))
__attribute__((address_space_of_locals(COMPUTECPP_ASP_OPENCL_LOCAL, 1, 0, 0, 1,
                                       1))) void
#endif
kernelgen_parallel_for_work_group(functorT functor) {
  const auto groupID =
      index_array_generator<dims>::template get_id<get_group_id>();

  const auto globalRange =
      index_array_generator<dims>::template get_range<get_global_size>();
  const auto localRange =
      index_array_generator<dims>::template get_range<get_local_size>();
  const auto workGroups =
      index_array_generator<dims>::template get_range<get_num_groups>();

  auto groupObj = group<dims>{
      detail::group_base(groupID, workGroups, globalRange, localRange)};
  functor(groupObj);
}

#undef COMPUTECPP_ASP_OPENCL_LOCAL

/** Kernel generation for the parallel_for_work_item.
 * @param groupP groupID Group identification
 * @param functorT functor containing the code for the work-item
 */
template <int dims, typename functorT>
COMPUTECPP_SYCL_DEVFUNC void kernelgen_parallel_for_work_item(
    group<dims> groupP, functorT functor) {
  (void)groupP;
  const auto globalID =
      index_array_generator<dims>::template get_id<get_global_id>();
  const auto localID =
      index_array_generator<dims>::template get_id<get_local_id_helper>();

  const auto globalRange = groupP.get_global_range();
  const auto localRange = groupP.get_local_range();

  auto itemID = h_item<dims>{
      detail::h_item_base(detail::item_base(localID, localRange),
                          detail::item_base(localID, localRange),
                          detail::item_base(globalID, globalRange))};

  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(
      access::fence_space::global_and_local));
  functor(itemID);
  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(
      access::fence_space::global_and_local));
}

/** Kernel generation for the parallel_for_work_item.
 * @param groupP groupID Group identification
 * @param localRange user provided local range
 * @param functorT functor containing the code for the work-item
 */
template <int dims, typename functorT>
COMPUTECPP_SYCL_DEVFUNC void kernelgen_parallel_for_work_item(
    group<dims> groupP, range<dims> localRange, functorT functor) {
  (void)groupP;
  const auto globalID =
      index_array_generator<dims>::template get_id<get_global_id>();
  const auto phyLocalID =
      index_array_generator<dims>::template get_id<get_local_id_helper>();
  const auto globalRange = groupP.get_global_range();
  const auto phyLocalRange = groupP.get_local_range();

  const auto localRange_3d = range<3>(detail::index_array{localRange});
  const auto phyLocalID_3d = id<3>{detail::index_array{phyLocalID}};
  const auto phyLocalRange_3d = range<3>{detail::index_array{phyLocalRange}};

  detail::barrier(
      detail::get_cl_mem_fence_flag(access::fence_space::global_and_local));
  for (int item_x = phyLocalID_3d[0]; item_x < localRange_3d[0];
       item_x += phyLocalRange_3d[0]) {
    for (int item_y = phyLocalID_3d[1]; item_y < localRange_3d[1];
         item_y += phyLocalRange_3d[1]) {
      for (int item_z = phyLocalID_3d[2]; item_z < localRange_3d[2];
           item_z += phyLocalRange_3d[2]) {
        auto localID = id<dims>{detail::index_array(item_x, item_y, item_z)};
        auto itemID = h_item<dims>{
            detail::h_item_base(detail::item_base(localID, localRange),
                                detail::item_base(phyLocalID, phyLocalRange),
                                detail::item_base(globalID, globalRange))};
        functor(itemID);
      }
    }
  }
  detail::barrier(
      detail::get_cl_mem_fence_flag(access::fence_space::global_and_local));
}

COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)

#undef COMPUTECPP_SYCL_DEVFUNC
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_COMPILER_HOOKS_H_
