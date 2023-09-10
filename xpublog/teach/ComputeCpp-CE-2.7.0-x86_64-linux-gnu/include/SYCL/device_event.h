/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file device_event.h
 *
 * @brief Declaration of the device event
 */

#ifndef RUNTIME_INCLUDE_SYCL_DEVICE_EVENT_H_
#define RUNTIME_INCLUDE_SYCL_DEVICE_EVENT_H_

#include "SYCL/addrspace_cast.h"
#include "SYCL/backend.h"

namespace cl {
namespace sycl {

class device_event;

namespace detail {
class nd_item_base;

template <>
struct opencl_backend_traits<sycl::device_event> {
 public:
  // input_type not defined
#ifdef __SYCL_DEVICE_ONLY__
  using return_type = __sycl_event_t;
#else
// return_type not defined on host
#endif  // __SYCL_DEVICE_ONLY__
};

}  // namespace detail

/** @brief Event for asynchronous operations on the device side.
 * Currently implemented as a no-op.
 *
 * On device it has to be located in private memory - by default,
 * objects are placed in local memory when created within
 * a parallel_for_work_group invocation.
 */
class device_event {
  friend class nd_item<1>;
  friend class nd_item<2>;
  friend class nd_item<3>;
  friend class group<1>;
  friend class group<2>;
  friend class group<3>;

 public:
#ifdef __SYCL_DEVICE_ONLY__
  device_event(__sycl_event_t internalEvent) : m_kernelEvent(internalEvent) {}
#else   // __SYCL_DEVICE_ONLY__
  device_event() = default;
#endif  // __SYCL_DEVICE_ONLY__

  device_event(const device_event&) = default;
  device_event& operator=(const device_event&) = default;

  device_event(device_event&&) = default;
  device_event& operator=(device_event&&) = default;

  /** @brief Wait for the event to finalize
   */
  void wait() const {
#ifdef __SYCL_DEVICE_ONLY__
    __sycl_event_t internalEvent = m_kernelEvent;
#ifdef __SYCL_COMPUTECPP_ASP__
    ::cl::sycl::detail::wait_group_events(
        1, ::cl::sycl::detail::addrspace_cast<__attribute__((opencl_private))
                                              __sycl_event_t*>(&internalEvent));
#else   // Offload
    ::cl::sycl::detail::wait_group_events(
        1, reinterpret_cast<__attribute__((opencl_private)) __sycl_event_t*>(
               &internalEvent));
#endif  // __SYCL_COMPUTECPP_ASP__
#else
// Barrier is not needed on the host because the operation is
// synchronous
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Destroys the device event. Performs a wait.
   */
  ~device_event() { wait(); }

 private:
#ifdef __SYCL_DEVICE_ONLY__
  __sycl_event_t m_kernelEvent;
#endif  // __SYCL_DEVICE_ONLY__

} COMPUTECPP_PRIVATE_MEMORY_ATTR;

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEVICE_EVENT_H_
