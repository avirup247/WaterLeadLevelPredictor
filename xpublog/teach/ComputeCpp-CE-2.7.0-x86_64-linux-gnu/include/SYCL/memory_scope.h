/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file memory_scope.h
 *
 * @brief This file implements the memory_scope enum class.
 */

#ifndef RUNTIME_INCLUDE_SYCL_MEMORY_SCOPE_H_
#define RUNTIME_INCLUDE_SYCL_MEMORY_SCOPE_H_

#include "computecpp_export.h"

namespace cl {
namespace sycl {

#if SYCL_LANGUAGE_VERSION >= 202001

/** @brief Indicates the scope of memory operations.
 */
enum class memory_scope {
  work_item,
  sub_group,
  work_group,
  device,
  system,
};

/** @brief Shortcut to memory_scope::work_item.
 */
inline constexpr memory_scope memory_scope_work_item = memory_scope::work_item;

/** @brief Shortcut to memory_scope::sub_group.
 */
inline constexpr memory_scope memory_scope_sub_group = memory_scope::sub_group;

/** @brief Shortcut to memory_scope::work_group.
 */
inline constexpr memory_scope memory_scope_work_group =
    memory_scope::work_group;

/** @brief Shortcut to memory_scope::device.
 */
inline constexpr memory_scope memory_scope_device = memory_scope::device;

/** @brief Shortcut to memory_scope::system.
 */
inline constexpr memory_scope memory_scope_system = memory_scope::system;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_MEMORY_SCOPE_H_

////////////////////////////////////////////////////////////////////////////////
