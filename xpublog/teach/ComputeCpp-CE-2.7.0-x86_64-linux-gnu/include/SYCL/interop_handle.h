/******************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_INTEROP_HANDLE_H_
#define RUNTIME_INCLUDE_SYCL_INTEROP_HANDLE_H_

/** @file interop_handle.h
 * @brief This file contains the interop_handle class
 */

#include "SYCL/backend.h"
#include "SYCL/codeplay/interop_handle.h"
#include "SYCL/error_log.h"
#include "SYCL/type_traits.h"

namespace cl {
namespace sycl {

class handler;

namespace detail {

class interop_handle;

/** Constructs an instance of sycl::interop_handle,
 *  which is not user constructible
 * @param syclQueue Queue to associate the handle with
 * @return Instance of the handle
 */
interop_handle make_interop_handle(interop_handle_tag<interop_handle>,
                                   dqueue_shptr syclQueue);

/** Handle that can be used to retrieve backend-specific objects
 *  inside a host_task
 * @note For OpenCL uses the same functionality as codeplay::interop_handle
 */
class interop_handle : protected ::cl::sycl::codeplay::interop_handle {
 private:
  using base_t = ::cl::sycl::codeplay::interop_handle;

 protected:
  /** Constructs a new instance
   * @param syclQueue Queue to associate the handle with
   */
  explicit interop_handle(dqueue_shptr syclQueue)
      : base_t{std::move(syclQueue)} {}

 public:
  friend interop_handle make_interop_handle(interop_handle_tag<interop_handle>,
                                            dqueue_shptr syclQueue);

  /// Not user constructible
  interop_handle() = delete;

  /** Retrieves the backend associated with the queue this handle uses
   * @return Backend enum value
   */
  COMPUTECPP_EXPORT backend get_backend() const noexcept;

  /** Retrieves the native memory object from an accessor
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @tparam dataT Accessor data type
   * @tparam dims Number of accessor dimensions
   * @tparam accessMode Accessor mode
   * @tparam accessTarget Accessor target
   * @tparam isPlaceholder Whether the accessor is a placeholder
   * @param bufferAccessor Accessor object
   * @return Backend-specific memory object
   */
  template <backend Backend, typename dataT, int dims, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  backend_return_t<Backend, buffer<dataT, dims>> get_native_mem(
      const sycl::accessor<dataT, dims, accessMode, accessTarget,
                           isPlaceholder>& bufferAccessor) const {
    COMPUTECPP_IF_CONSTEXPR(Backend == sycl::backend::opencl) {
      return {base_t::get(bufferAccessor)};
    }
    else {
      COMPUTECPP_NOT_IMPLEMENTED(
          "Only the OpenCL backend is supported for interop_handle")
    }
  }

#ifdef COMPUTECPP_SYCL_2020_IMAGES

  /** Retrieves the native image object from an unsampled image accessor
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @tparam dataT Accessor data type
   * @tparam dims Number of accessor dimensions
   * @tparam accessMode Accessor mode
   * @param imageAcc Accessor object
   * @return Backend-specific image
   */
  template <backend Backend, typename dataT, int dims, access::mode accessMode>
  backend_return_t<Backend, unsampled_image<dims>> get_native_mem(
      const unsampled_image_accessor<dataT, dims, accessMode,
                                     image_target::device>& imageAcc) const {
    COMPUTECPP_NOT_IMPLEMENTED(
        "Image accessors not supported yet for interop_handle")
  }

  /** Retrieves the native image object from a sampled image accessor
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @tparam dataT Accessor data type
   * @tparam dims Number of accessor dimensions
   * @param imageAcc Accessor object
   * @return Backend-specific image
   */
  template <backend Backend, typename dataT, int dims>
  backend_return_t<Backend, sampled_image<dims>> get_native_mem(
      const sampled_image_accessor<dataT, dims, image_target::device>& imageAcc)
      const {
    COMPUTECPP_NOT_IMPLEMENTED(
        "Image accessors not supported yet for interop_handle")
  }

#endif  // COMPUTECPP_SYCL_2020_IMAGES

  /** Retrieves the native queue object from a SYCL queue
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @return Backend-specific queue
   */
  template <backend Backend>
  backend_return_t<Backend, sycl::queue> get_native_queue() const {
    COMPUTECPP_IF_CONSTEXPR(Backend == sycl::backend::opencl) {
      return base_t::get_queue();
    }
    else {
      COMPUTECPP_NOT_IMPLEMENTED(
          "Only the OpenCL backend is supported for interop_handle")
    }
  }

  /** Retrieves the native device object from a SYCL device
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @return Backend-specific device
   */
  template <backend Backend>
  backend_return_t<Backend, sycl::device> get_native_device() const {
    COMPUTECPP_IF_CONSTEXPR(Backend == sycl::backend::opencl) {
      return base_t::get_device();
    }
    else {
      COMPUTECPP_NOT_IMPLEMENTED(
          "Only the OpenCL backend is supported for interop_handle")
    }
  }

  /** Retrieves the native context object from a SYCL context
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @return Backend-specific context
   */
  template <backend Backend>
  backend_return_t<Backend, sycl::context> get_native_context() const {
    COMPUTECPP_IF_CONSTEXPR(Backend == sycl::backend::opencl) {
      return base_t::get_context();
    }
    else {
      COMPUTECPP_NOT_IMPLEMENTED(
          "Only the OpenCL backend is supported for interop_handle")
    }
  }

  /** Retrieves the native platform object from a SYCL platform
   * @tparam Backend Which backend to retrieve information from.
   *         Must match the backend of the associated queue.
   * @return Backend-specific platform
   */
  template <backend Backend>
  backend_return_t<Backend, sycl::platform> get_native_platform() const {
    COMPUTECPP_IF_CONSTEXPR(Backend == sycl::backend::opencl) {
      return base_t::get_context();
    }
    else {
      COMPUTECPP_NOT_IMPLEMENTED(
          "Only the OpenCL backend is supported for interop_handle")
    }
  }
};

inline interop_handle make_interop_handle(interop_handle_tag<interop_handle>,
                                          dqueue_shptr syclQueue) {
  return interop_handle{std::move(syclQueue)};
}

}  // namespace detail

#if SYCL_LANGUAGE_VERSION >= 202002

using interop_handle = detail::interop_handle;

#endif  // SYCL_LANGUAGE_VERSION >= 202002

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_INTEROP_HANDLE_H_
