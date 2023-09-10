/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file task.h

  @brief This file contains internal classes and functions used to implement
  kernel invocation APIs.
  */

#ifndef RUNTIME_INCLUDE_SYCL_TASK_H_
#define RUNTIME_INCLUDE_SYCL_TASK_H_

#include "SYCL/common.h"
#include "SYCL/group_base.h"
#include "SYCL/item_base.h"

namespace cl {
namespace sycl {

namespace codeplay {
class interop_handle;
}

namespace detail {

class base_task;
class event;
class event_list;
class interop_handle;

/** @brief Type of function_class pointer to a function without parameters,
 * following the singleTask execution model from the SYCL specification
 *
 */
using single_task_ptr = function_class<void()>;

/** @brief Type of function_class pointer to a function without parameters,
 * following the parallel_for execution model from the SYCL specification
 *
 */
using parallel_for_ptr = function_class<void(cl::sycl::detail::nd_item_base&)>;

/** @brief Type of function_class pointer to a function without parameters,
 * following the parallel_for execution model from the SYCL specification
 *
 */
using parallel_for_id_ptr = function_class<void(cl::sycl::detail::item_base&)>;

/** @brief Type of function_class pointer to a function without parameters,
 * following the Hierarchical execution model from the SYCL specification
 */
using parallel_for_work_group_ptr =
    function_class<void(cl::sycl::detail::group_base&)>;

/** @brief Type of function_class pointer to a function that takes a queue,
 *        following the host_task execution model of the Codeplay extension
 */
using host_command_task_ptr = function_class<void(const dqueue_shptr&)>;

/** @brief Type of function_class pointer to a function that takes a queue and a
 *        predecessor list, used for host commands that enqueue an operation on
 *        a queue
 */
using enqueue_task_ptr =
    function_class<devent_shptr(const dqueue_shptr&, event_list*)>;

/** @brief Type of function_class pointer to a function
 *        that takes an interop_handle, required by the interop_task
 */
using codeplay_interop_task_ptr =
    function_class<void(const codeplay::interop_handle&)>;

/** @brief Type of function_class pointer to a function
 *        that takes an interop_handle, required by host_task
 */
using interop_task_ptr = function_class<void(const detail::interop_handle&)>;

/** @brief Unique pointer to a base task
 */
using baseTask_uptr = unique_ptr_class<detail::base_task>;

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_TASK_H_
