/******************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file apis.h

  @brief This file contains Codeplay specific extensions to the SYCL API
*/
#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_APIS_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_APIS_H_

#include "SYCL/apis.h"
#include "SYCL/codeplay/interop_handle.h"
#include "SYCL/common.h"
#include "SYCL/queue.h"

namespace cl {
namespace sycl {

namespace codeplay {

#ifndef __SYCL_DEVICE_ONLY__

/** @brief Command group handler that implements Codeplay specific API
 *        extensions
 */
class COMPUTECPP_EXPORT handler {
 public:
  friend class queue;
  friend class detail::queue;

  /** @brief Conversion operator
   * @return Reference to normal SYCL handler
   */
  operator cl::sycl::handler&() { return m_cgh; }

  /** @brief Conversion operator, const overload
   * @return Const reference to normal SYCL handler
   */
  operator const cl::sycl::handler&() const { return m_cgh; }

  /// @cond COMPUTECPP_DEV

  /** @brief Returns an internal transaction
   */
  inline detail::transaction* get_transaction() const {
    return m_cgh.get_transaction();
  }

  /// COMPUTECPP_DEV @endcond

  /** @brief Registers a placeholder accessor with the handler
   * @param acc Placeholder accessor
   */
  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget>
  void require(const accessor<elemT, kDims, kMode, kTarget,
                              access::placeholder::true_t>& acc) {
    m_cgh.require(acc);
  }

  /////////////// API : Interop Task

  /** @brief Launches a single host task that
   *        allows access to OpenCL interop objects
   * @tparam FunctorT Type of the user functions. Determined by the compiler.
   * @param functor User function being enqueued
   */
  template <typename FunctorT>
  void interop_task(const FunctorT& functor) {
    this->interop_task_impl(functor);
  }

 protected:
  /** @brief Creates a handler for a specific queue
   * @param q Queue to run operations on
   * @param fallbackQueue Queue used in case of a failure on the main queue
   */
  explicit handler(const dqueue_shptr& q,
                   const dqueue_shptr& fallbackQueue = nullptr)
      : m_cgh(q, fallbackQueue) {}

  /** @brief Launches the interop task
   * @param interopTaskPtr User function
   */
  void interop_task_impl(
      const detail::codeplay_interop_task_ptr& interopTaskPtr);

 private:
  /** @brief Actual handler object
   */
  cl::sycl::handler m_cgh;

};  // class handler

/**
  @brief Command group host handler that implements Codeplay specific API
  extensions
*/
class COMPUTECPP_EXPORT host_handler {
 public:
  friend class queue;
  friend class detail::queue;

 protected:
  /** @brief Creates a handler for a specific queue.
   */
  explicit host_handler(const dqueue_shptr& q,
                        const dqueue_shptr& fallbackQueue = nullptr)
      : cgh(q, fallbackQueue) {}

 public:
  operator cl::sycl::handler&() { return cgh; }
  operator const cl::sycl::handler&() const { return cgh; }

  /// @cond COMPUTECPP_DEV

  /** @brief Returns an internal transaction
   */
  detail::transaction* get_transaction() const;

  /// COMPUTECPP_DEV @endcond

  /////////////// API : Host Task

  /**
    @brief This function effectively just launches a single thread
    to execute the kernel in serial asynchronously to the host execution.
    @tparam functorT this is the type of the kernel. It will be automatically
    deduced by the compiler
    @param functor the kernel being enqueued
  */
  template <typename functorT>
  void host_task(const functorT& functor) {
    this->host_task_impl(functor);
  }
  /**
      @brief Register a single event that this handler should wait for before
      running.
      @param e The event that the handler should wait for before running.
    */
  void experimental_depends_on(cl::sycl::event e) {
    cgh.experimental_depends_on(e);
  }

  /** @brief Register a set of events that this handler should wait for before
  running.
   * @param v a vector of events.
   */
  void experimental_depends_on(std::vector<cl::sycl::event> v) {
    cgh.experimental_depends_on(v);
  }

 protected:
  void host_task_impl(const detail::single_task_ptr& singleTaskPtr);

  cl::sycl::handler cgh;

};  // class host_handler

#else  // __SYCL_DEVICE_ONLY__

class handler {
 public:
  explicit handler(dqueue_shptr q);

  operator cl::sycl::handler&();
  operator const cl::sycl::handler&();

  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget>
  void require(const accessor<elemT, kDims, kMode, kTarget,
                              access::placeholder::true_t>& acc);

  /////////////// API : Interop Task

  template <typename functorT>
  void interop_task(const functorT&) {
    // No work for the device compiler
  }

};  // class handler

class host_handler {
 public:
  explicit host_handler(dqueue_shptr q) : cgh(q) {}

  operator cl::sycl::handler&() { return cgh; }
  operator const cl::sycl::handler&() const { return cgh; }

  /////////////// API : Host Task

  template <typename functorT>
  void host_task(const functorT&) {
    // No work for the device compiler
  }

  void experimental_depends_on(cl::sycl::event) {
    // No work for the device compiler
  }

  void experimental_depends_on(const std::vector<cl::sycl::event>&) {
    // No work for the device compiler
  }

 protected:
  cl::sycl::handler cgh;

};  // class host_handler

#endif  // __SYCL_DEVICE_ONLY__

/** Codeplay extension.
 * Flushes all command groups that have been submited to a queue. Synchronous
 * errors are reported through C++ exceptions.
 */
COMPUTECPP_EXPORT void flush(cl::sycl::queue& syclQueue);

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_APIS_H_
