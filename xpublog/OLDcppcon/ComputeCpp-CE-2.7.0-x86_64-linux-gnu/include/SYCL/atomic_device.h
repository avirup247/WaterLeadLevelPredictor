/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

    * atomic_device.h *

*******************************************************************************/

/** @file atomic_device.h
 * @brief This file contains an implementation of the member functions of the
 *        atomic class for the device.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ATOMIC_DEVICE_H_
#define RUNTIME_INCLUDE_SYCL_ATOMIC_DEVICE_H_

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/builtins/device_builtins.h"
#endif  // __SYCL_DEVICE_ONLY__

#include "SYCL/atomic.h"
#include "SYCL/common.h"
#include "SYCL/type_traits.h"

#include <type_traits>
#include <utility>

namespace cl {
namespace sycl {

#ifdef __SYCL_DEVICE_ONLY__

/** @cond COMPUTECPP_DEV */

namespace detail {
/** @brief helper function for atomic_compare_exchange on device
 * @param old the old value of *m_data
 * @param expected the expected value of *m_data
 * @return old == expected
 */
template <typename T>
cl_bool cmpxchg_helper(T old, T& expected) {
  if (old == expected) {
    return true;
  } else {
    expected = old;
    return false;
  }
}

/** @brief Helper struct to deduce whether 32-bit atomics or 64-bit atomics are
 *        to be used. The primary template handles 32-bit atomics only.
 * @tparam T The type of the atomic operation.
 * @tparam Size The size of T.
 */
template <typename T, std::size_t Size = sizeof(T)>
struct atomic_helper {
  atomic_helper() = delete;

  /** @brief Helper function for calling 32-bit atomic exchange built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_xchg.
   */
  template <typename MultiPtr>
  static T xchg(MultiPtr& data, T& operand) {
    return ::cl::sycl::detail::atomic_xchg(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic compare-and-exchange
   * built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param expected The expression's expected value.
   * @param desired The expression's desired value.
   * @return Result of atomic_cmpxchg.
   */
  template <typename MultiPtr>
  static T cmpxchg(MultiPtr& data, T& expected, T desired) {
    return ::cl::sycl::detail::atomic_cmpxchg(
        ::cl::sycl::detail::cpp_to_cl_cast(data), expected, desired);
  }

  /** @brief Helper function for calling 32-bit atomic add built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_add.
   */
  template <typename MultiPtr>
  static T add(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_add(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic subtract built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_sub.
   */
  template <typename MultiPtr>
  static T sub(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_sub(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic logical or built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_or.
   */
  template <typename MultiPtr>
  static T logical_or(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_or(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic logical and built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_and.
   */
  template <typename MultiPtr>
  static T logical_and(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_and(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic logical xor built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_xor.
   */
  template <typename MultiPtr>
  static T logical_xor(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_xor(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic min built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_min.
   */
  template <typename MultiPtr>
  static T min(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_min(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 32-bit atomic max built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atomic_max.
   */
  template <typename MultiPtr>
  static T max(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atomic_max(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }
};

/** @brief Helper struct to deduce whether 32-bit atomics or 64-bit atomics are
 *        to be used.
 * @tparam T The type of the atomic operation.
 * @note This specialisation targets 64-bit atomics only.
 */
template <typename T>
struct atomic_helper<T, 8> {
  atomic_helper() = delete;

  /** @brief Helper function for calling 64-bit atomic exchange built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_xchg.
   */
  template <typename MultiPtr>
  static T xchg(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_xchg(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic compare-and-exchange
   * built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param expected The expression's expected value.
   * @param desired The expression's desired value.
   * @return Result of atom_cmpxchg.
   */
  template <typename MultiPtr>
  static T cmpxchg(MultiPtr& data, T& expected, T desired) {
    return ::cl::sycl::detail::atom_cmpxchg(
        ::cl::sycl::detail::cpp_to_cl_cast(data), expected, desired);
  }

  /** @brief Helper function for calling 64-bit atomic add built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_add.
   */
  template <typename MultiPtr>
  static T add(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_add(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic subtract built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_sub.
   */
  template <typename MultiPtr>
  static T sub(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_sub(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic logical or built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_or.
   */
  template <typename MultiPtr>
  static T logical_or(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_or(::cl::sycl::detail::cpp_to_cl_cast(data),
                                       operand);
  }

  /** @brief Helper function for calling 64-bit atomic logical and built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_and.
   */
  template <typename MultiPtr>
  static T logical_and(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_and(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic logical xor built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_xor.
   */
  template <typename MultiPtr>
  static T logical_xor(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_xor(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic min built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_min.
   */
  template <typename MultiPtr>
  static T min(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_min(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }

  /** @brief Helper function for calling 64-bit atomic max built-in.
   * @tparam MultiPtr The multi_ptr type.
   * @param data The multi_ptr to be operated on.
   * @param operand The expression's operand.
   * @return Result of atom_max.
   */
  template <typename MultiPtr>
  static T max(MultiPtr& data, T operand) {
    return ::cl::sycl::detail::atom_max(
        ::cl::sycl::detail::cpp_to_cl_cast(data), operand);
  }
};
}  // namespace detail

/* General implementation
 * --------------------------------------------------------------------------*/

// OpenCL 1.2 has no store operation, so swap with the desired value, then
// discard the old value (i.e. *m_data).
template <typename T, access::address_space AddressSpace>
inline void atomic<T, AddressSpace>::store(T operand, memory_order) const
    noexcept {
  detail::atomic_helper<T>::xchg(m_data, operand);
}

// OpenCL 1.2 has no load operation, so add zero to obtain the "old" value.
template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::load(memory_order) const {
  return detail::atomic_helper<T>::add(m_data, 0);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::exchange(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::xchg(m_data, operand);
}

// OpenCL 1.2 does not implement compare_and_exchange in the same way as C++11,
// so this function emulates the behavior on the device. It attempts to figure
// out if the call to cmpxchg actually altered the atomic value by capturing the
// old value then comparing it to expected. If they are the same, then the
// comparison will have succeeded, so we can return true. If they aren't, then
// comparison failed, so we need to update expected manually then return false.
// The helper function performs this part of the logic, since it is shared by
// all compare_exchange device implementations.
template <typename T, access::address_space AddressSpace>
inline cl_bool atomic<T, AddressSpace>::compare_exchange_strong(
    T& expected, T desired, memory_order /*success*/,
    memory_order /*fail*/) const noexcept {
  auto old = detail::atomic_helper<T>::cmpxchg(m_data, expected, desired);
  return ::cl::sycl::detail::cmpxchg_helper(old, expected);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_add(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::add(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_sub(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::sub(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_and(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::logical_and(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_or(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::logical_or(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_xor(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::logical_xor(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_min(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::min(m_data, operand);
}

template <typename T, access::address_space AddressSpace>
inline T atomic<T, AddressSpace>::fetch_max(T operand, memory_order) const
    noexcept {
  return detail::atomic_helper<T>::max(m_data, operand);
}

/* <cl_float, global> specialization
 *
 --------------------------------------------------------------------------*/

// Unlike store, there is no way to non-destructively update m_data and return
// the old value, so a simple load is issued. This should be atomic regardless.
template <>
inline cl_float atomic<cl_float, access::address_space::global_space>::load(
    memory_order) const {
  return *m_data;
}

/* <cl_float, local> specialization
 *
 --------------------------------------------------------------------------*/

template <>
inline cl_float atomic<cl_float, access::address_space::local_space>::load(
    memory_order) const {
  return *m_data;
}

/** COMPUTECPP_DEV @endcond  */

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ATOMIC_DEVICE_H_
