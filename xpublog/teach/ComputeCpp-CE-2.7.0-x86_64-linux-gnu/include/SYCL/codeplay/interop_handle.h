/******************************************************************************

    Copyright (C) 2002-2019 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_INTEROP_HANDLE_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_INTEROP_HANDLE_H_

/** @file interop_handle.h
 * @brief This file contains the interop_handle class
 */

#include "SYCL/base.h"
#include "SYCL/common.h"
#include "SYCL/include_opencl.h"

#include "computecpp_export.h"

namespace cl {
namespace sycl {
#ifndef __SYCL_DEVICE_ONLY__
class accessor_base;
#else
template <typename T>
class accessor_device_base;
using accessor_base = accessor_device_base<void*>;
#endif
template <typename dataT, int dimensions, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;

namespace codeplay {
class interop_handle;
}
namespace detail {

/** Helper struct used to distinguish between codeplay::handle and sycl::handle
 * @tparam interop_handle_t Type of the handle
 */
template <class interop_handle_t>
struct interop_handle_tag {};

/** Constructs an instance of codeplay::interop_handle,
 *  which is not user constructible
 * @param syclQueue Queue to associate the handle with
 * @return Instance of the handle
 */
codeplay::interop_handle make_interop_handle(
    interop_handle_tag<codeplay::interop_handle>, dqueue_shptr syclQueue);

}  // namespace detail

namespace codeplay {
class handler;

#ifndef __SYCL_DEVICE_ONLY__

/** @brief Handle that allows access to OpenCL interop objects
 *        on a specific SYCL queue
 */
class COMPUTECPP_EXPORT interop_handle {
 public:
  friend class codeplay::handler;

  friend codeplay::interop_handle detail::make_interop_handle(
      detail::interop_handle_tag<codeplay::interop_handle>,
      dqueue_shptr syclQueue);

  /** @brief Retrieves an OpenCL memory object (buffer/image)
   *        associated with a device accessor
   * @tparam elemT Underlying type of the accessor data
   * @tparam kDims Number of data dimensions
   * @tparam kMode Access mode
   * @tparam kTarget Access target
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @param acc Accessor to retrieve the OpenCL memory object from
   * @return OpenCL memory object
   */
  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget, access::placeholder isPlaceholder>
  cl_mem get(
      const accessor<elemT, kDims, kMode, kTarget, isPlaceholder>& acc) const {
    return get_mem_object_impl(acc);
  }

  /** @brief Retrieves the OpenCL command queue from the SYCL one
   * @return OpenCL queue
   */
  cl_command_queue get_queue() const;

  /** @brief Retrieves the OpenCL context associated with the SYCL queue
   * @return OpenCL context
   */
  cl_context get_context() const;

  /** @brief Retrieves the OpenCL device associated with the SYCL queue
   * @return OpenCL device
   */
  cl_device_id get_device() const;

 protected:
  /** @brief Constructs an interop object from a SYCL queue
   * @param syclQueue Detail SYCL queue object
   */
  explicit interop_handle(dqueue_shptr syclQueue) : m_queue(syclQueue) {}

  /** @brief Helper function for retrieving the OpenCL memory object
   *        from an accessor
   * @param acc Accessor to retrieve the OpenCL memory object from
   * @return OpenCL memory object
   */
  cl_mem get_mem_object_impl(const accessor_base& acc) const;

 private:
  /** @brief SYCL queue used to provide OpenCL interop objects
   */
  dqueue_shptr m_queue;

};  // class interop_handle

#else  // __SYCL_DEVICE_ONLY__

class interop_handle {
 public:
  explicit interop_handle(dqueue_shptr);

  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget, access::placeholder isPlaceholder>
  cl_mem get(
      const accessor<elemT, kDims, kMode, kTarget, isPlaceholder>& acc) const;
  cl_command_queue get_queue() const;
  cl_device_id get_device() const;
  cl_context get_context() const;
};  // class interop_handle

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace codeplay

namespace detail {

inline codeplay::interop_handle make_interop_handle(
    interop_handle_tag<codeplay::interop_handle>, dqueue_shptr syclQueue) {
  return codeplay::interop_handle{std::move(syclQueue)};
}

}  // namespace detail

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_INTEROP_HANDLE_H_
