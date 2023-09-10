/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

    * atomic.h *

*******************************************************************************/

/** @file atomic.h
 * This file contains an implementation of the atomic class as described
 * in the SYCL specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ATOMIC_H_
#define RUNTIME_INCLUDE_SYCL_ATOMIC_H_

#include <algorithm>
#include <atomic>

#include "memory_scope.h"

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor_buffer_interface;

}  // namespace detail

/** Only relaxed memory order is supported in SYCL 1.2.1
 */
enum class memory_order : int {
  relaxed = static_cast<int>(std::memory_order_relaxed),
};

#if SYCL_LANGUAGE_VERSION >= 202001
/** Compile-time traits for memory orders. Gives separate read and write orders
 * for a read-modify-write order.
 * @tparam ReadModifyWriteOrder The read-modify-write order for which a read
 * and write order are defined.
 */
template <memory_order ReadModifyWriteOrder>
struct memory_order_traits;

/** Specialization of memory_order_traits for memory_order::relaxed. Gives
 * a read and write order.
 */
template <>
struct memory_order_traits<memory_order::relaxed> {
  static constexpr memory_order read_order = memory_order::relaxed;
  static constexpr memory_order write_order = memory_order::relaxed;
};
#endif  // SYCL_LANGUAGE_VERSION >= 202001

/** Atomic class template
 *
 * This template class specifies the interface and internal data of atomics
 * as specified by SYCL. It offers several different atomic operations,
 * including min/max which are not otherwise available in C++ 11 code.
 * Most of the file is visible to the device compiler only;
 * this is so that the runtime can call the appropriate atomic function based
 * on the type of the elements. A portion is visible to both (class declaration
 * and global functions) with a small section for the host-only implementation.
 * The device compiler section has separate specializations for each pair of
 * template parameters. They are organized primarily by type (cl_int, cl_uint
 * etc.) and secondarily by address space (global then local). It is done like
 * this because the SPIR function to be called is different based on the type
 * and address space of the atomic.
 */
template <typename T, access::address_space addressSpace =
                          access::address_space::global_space>
struct atomic;

/** Atomic int object with default global address space
 */
using atomic_int = atomic<cl_int>;
/** Atomic unsigned int object with default global address space
 */
using atomic_uint = atomic<cl_uint>;
/** Atomic float object with default global address space
 */
using atomic_float = atomic<cl_float>;

#if SYCL_LANGUAGE_VERSION >= 202001
/** SYCL-2020 atomic_ref, implementing atomic operations common to all types.
 * An atomic_ref allows a referenced value to be acted on atomically.
 * Referencing 64bit types requires aspect::atomic64.
 *  @tparam T Type referenced by atomic_ref
 *  @tparam DefaultOrder the memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope the memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace the address space of the reference.
 *  @tparam Enable variable to allow type-specific specializations.
 */
namespace detail {
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space addressSpace = access::address_space::
              global_space>  // TODO: access::address_space::generic_space
class atomic_ref_base;
}

/** SYCL-2020 atomic_ref, implementing atomic operations on a referenced value.
 * Referencing 64bit types requires aspect::atomic64.
 *  @tparam T Type referenced by atomic_ref
 *  @tparam DefaultOrder the memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope the memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace the address space of the reference.
 *  @tparam Enable variable to allow type-specific specializations.
 */
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space addressSpace =
              access::address_space::global_space,
          class Enable = void>  // TODO: access::address_space::generic_space
class atomic_ref;
#endif  // SYCL_LANGUAGE_VERSION

template <typename elemT, access::address_space addressSpace>
struct device_type {
  /** Underlying type of the device pointer
   */
  using underlying_t = elemT;

  /** Pointer type used on device
   */
  using ptr_t = multi_ptr<underlying_t, addressSpace>;
};

/** Implementation of the SYCL atomic class according to 1.2 spec.
 * (section 3.8). On host, calls C++ atomic functions on an
 * std::atomic; on device uses SPIR-mangled OpenCL 1.2 functions
 * to achieve same result.
 */
template <typename T, access::address_space addressSpace>
struct atomic {
 private:
  /** Pointer type used on the device
   */
  using device_ptr_t = typename device_type<T, addressSpace>::ptr_t;

  /* @cond COMPUTECPP_DEV */
  /* Host has a single C++ 11 atomic, device simply has a pointer in the
   * global address space */

#ifndef __SYCL_DEVICE_ONLY__
  /** Pointer to std::atomic<T>, host only.
   */
  std::atomic<T>* m_data;
#else
  /** Pointer decorated with address space. Device only.
   */
  device_ptr_t m_data;
#endif

  /** Factory function only visible to accessors. Stores the address
   * provided internally and operates on that location atomically.
   * @param datum The address to be operated on atomically, obtained
   * from an atomic accessor.
   */
  static atomic make_from_device_ptr(device_ptr_t datum) {
    atomic result;
#ifndef __SYCL_DEVICE_ONLY__
    memcpy(&result.m_data, &datum, sizeof(m_data));
#else
    result.m_data = datum;
#endif
    return result;
  }

  /** Private default constructor that is meant to be used in
   * make_from_device_ptr only.
   */
  atomic() : m_data{nullptr} {}

  /* @endcond  */

#ifndef __SYCL_DEVICE_ONLY__
#if SYCL_LANGUAGE_VERSION >= 202001
  /// Return true if the atomic operations provided by this sycl::atomic are
  /// always lock-free.
  static constexpr bool is_always_lock_free =
      std::atomic<T>::is_always_lock_free;
#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /// The value is true if the operations provided by sycl::atomic are
  /// lock-free.
  bool is_lock_free() const noexcept { return m_data->is_lock_free(); }
#else
  // is_always_lock_free is not implemented for devices.
  // is_lock_free() is not implemented for devices.
#endif  // __SYCL_DEVICE_ONLY__

  /** Atomically compare and optionally exchange
   * expected with *m_data. Calls C++11 equivalent on host,
   * has to be implemented "by hand" on device because
   * OpenCL 1.2 and C++ 11 have different semantics for
   * compare and exchange. If *m_data == expected, may perform
   * *m_data = desired and returns true. Otherwise, performs
   * expected = *m_data and returns false. Weak indicates that the
   * latter may sometimes occur even when *m_data == expected.
   * @param expected The value to compare against *m_data.
   * @param desired The value to store in *m_data on success.
   * @param success the ordering to use when comparison
   * succeeds. Can only be memory_order_relaxed.
   * @param fail the ordering to use when comparison fails.
   * Can only be memory_order_relaxed.
   * @return True if comparison succeeds, false if it fails.
   */
  cl_bool compare_exchange_weak(T& expected, T desired,
                                memory_order success = memory_order::relaxed,
                                memory_order fail = memory_order::relaxed)  //
      const noexcept;

 public:
  /** The accessors are friends so that it can access the constructor but
   * user code can't. */
  template <typename elemT, int kDims, access::mode kMode,
            access::target kTarget, access::placeholder isPlaceholder>
  friend class detail::accessor_buffer_interface;

#if SYCL_LANGUAGE_VERSION >= 202001
  /// The atomic_ref_base class needs access to compare_exchange_weak.
  template <typename TyRef, memory_order MOrderRef, memory_scope MScopeRef,
            access::address_space AddrSpaceRef>
  friend class detail::atomic_ref_base;
  /// The atomic_ref class is a friend to allow direct access to m_data.
  /// for fetch_add/sub specialization for pointers.
  template <typename TyRef, memory_order MOrderRef, memory_scope MScopeRef,
            access::address_space AddrSpaceRef, class EnableRef>
  friend class atomic_ref;
#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** Constructs an instance of SYCL atomic which is associated with the
   *        pointer ptr, converted to a pointer of data type T.
   *
   *        Permitted data types for pointerT is any valid scalar data type
   *        which is the same size in bytes as T.
   *
   * @tparam pointerT Underlying type of the pointer ptr
   * @param ptr Pointer to be used in an atomic manner
   */
  template <typename pointerT>
  atomic(multi_ptr<pointerT, addressSpace> ptr)
      : atomic(make_from_device_ptr(ptr.get())) {}

  /* Functions as mandated by specification. Global functions simply
   * forward on to these function calls. */

  /** Atomically store operand in m_data. Calls C++11 equivalent on host,
   * on device it calls exchange, discarding the result.
   * @param operand the value to store in m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   */
  void store(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically load from m_data. Calls C++11 equivalent on host,
   * on device it either calls atomic_add with operand = 0, discarding
   * the result.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return The value loaded from m_data.
   */
  T load(memory_order mem_order = memory_order::relaxed) const;

  /** Atomically exchange operand with *m_data.
   * @param operand the value to store in *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return The old value of *m_data.
   */
  T exchange(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically compare and optionally exchange expected with *m_data.
   * Calls C++11 equivalent on host, has to be implemented "by hand" on device
   * because OpenCL 1.2 and C++ 11 have different semantics for compare and
   * exchange.
   * If *m_data == expected, performs *m_data = desired and returns true.
   * Otherwise, performs expected = *m_data and returns false.
   * @param expected The value to compare against *m_data.
   * @param desired The value to store in *m_data on success.
   * @param success the ordering to use when comparison succeeds. Can only
   * be memory_order_relaxed.
   * @param fail the ordering to use when comparison fails. Can only
   * be memory_order_relaxed.
   * @return True if comparison succeeds, false if it fails.
   */
  cl_bool compare_exchange_strong(
      T& expected, T desired, memory_order success = memory_order::relaxed,
      memory_order fail = memory_order::relaxed) const noexcept;

  /** Atomically add operand to *m_data.
   * param operand the value to add to *m_data.
   * param mem_order the ordering to use. Can only be memory_order_relaxed.
   * return the old value of *m_data.
   */
  T fetch_add(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically subtract operand from *m_data.
   * @param operand the value to subtract from *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of m_data.
   */
  T fetch_sub(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically bitwise-and operand with *m_data.
   * @param operand the value to and with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_and(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically bitwise-or operand with *m_data.
   * @param operand the value to or with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_or(T operand, memory_order = memory_order::relaxed) const noexcept;

  /** Atomically bitwise-XOR operand with *m_data.
   * @param operand the value to XOR with *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_xor(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically compare operand to *m_data, storing the smaller of the
   * two
   * in *m_data.
   * @param operand the value to compare to *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_min(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;

  /** Atomically compare operand to *m_data, storing the larger of the
   * two in *m_data.
   * @param operand the value to compare to *m_data.
   * @param mem_order the ordering to use. Can only be memory_order_relaxed.
   * @return the old value of *m_data.
   */
  T fetch_max(T operand, memory_order mem_order = memory_order::relaxed) const
      noexcept;
};

#ifndef __SYCL_DEVICE_ONLY__
/** @cond COMPUTECPP_DEV */

template <typename T, access::address_space addressSpace>
inline void atomic<T, addressSpace>::store(T operand,
                                           memory_order mem_order) const
    noexcept {
  m_data->store(operand, static_cast<std::memory_order>(mem_order));
  return;
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::load(memory_order mem_order) const {
  return m_data->load(static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::exchange(T operand,
                                           memory_order mem_order) const
    noexcept {
  return m_data->exchange(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline cl_bool atomic<T, addressSpace>::compare_exchange_strong(
    T& expected, T desired, memory_order success, memory_order fail) const
    noexcept {
  return m_data->compare_exchange_strong(
      expected, desired, static_cast<std::memory_order>(success),
      static_cast<std::memory_order>(fail));
}

template <typename T, access::address_space addressSpace>
inline cl_bool atomic<T, addressSpace>::compare_exchange_weak(
    T& expected, T desired, memory_order success, memory_order fail) const
    noexcept {
  return m_data->compare_exchange_weak(expected, desired,
                                       static_cast<std::memory_order>(success),
                                       static_cast<std::memory_order>(fail));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_add(T operand,
                                            memory_order mem_order) const
    noexcept {
  return m_data->fetch_add(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_sub(T operand,
                                            memory_order mem_order) const
    noexcept {
  return m_data->fetch_sub(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_and(T operand,
                                            memory_order mem_order) const
    noexcept {
  return m_data->fetch_and(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_or(T operand,
                                           memory_order mem_order) const
    noexcept {
  return m_data->fetch_or(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_xor(T operand,
                                            memory_order mem_order) const
    noexcept {
  return m_data->fetch_xor(operand, static_cast<std::memory_order>(mem_order));
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_min(const T operand,
                                            const memory_order memOrder) const
    noexcept {
  /* Standard C++ 11 defines no "min" operation, so this function emulates the
   * behavior by executing a loop. First, the value is loaded, then compared
   * against the operand. After this a compare_exchange is used to check that
   * the value hasn't been updated sneakily (by another thread say); if it has
   * (i.e. m_data doesn't equal its old value) store the new value of the atomic
   * and try again. */
  T old = m_data->load(std::memory_order_relaxed);
  do {
    if (old < operand) {
      break;
    }
  } while (!compare_exchange_weak(old, operand, memOrder, memOrder));
  return old;
}

template <typename T, access::address_space addressSpace>
inline T atomic<T, addressSpace>::fetch_max(const T operand,
                                            const memory_order memOrder) const
    noexcept {
  /* Standard C++ 11 defines no "max" operation, so this function emulates the
   * behavior by executing a loop. First, the value is loaded, then compared
   * against the operand. After this a compare_exchange is used to check that
   * the value hasn't been updated sneakily (by another thread say); if it has
   * (i.e. m_data doesn't equal its old value) store the new value of the atomic
   * and try again. */
  T old = m_data->load(std::memory_order_relaxed);
  do {
    if (operand < old) {
      break;
    }
  } while (!compare_exchange_weak(old, operand, memOrder, memOrder));
  return old;
}

/** COMPUTECPP_DEV @endcond  */

#endif  // __SYCL_DEVICE_ONLY__

/* global function definitions
 * --------------------------------------------------------------------------*/

/* For each of these global functions f(atomic * a, operands...), the code is
 * simply:
 * a->f(operands) */

/** @function Global function atomic_load. Calls load on SYCL atomic object.
 * @param object The atomic object to load from
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_load(atomic<T, addressSpace> object,
                     memory_order mem_order = memory_order::relaxed) {
  return object.load(mem_order);
}

/** Global function atomic_store. Calls store on SYCL atomic object.
 * @param object The atomic object to store to
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 */
template <typename T, access::address_space addressSpace>
inline void atomic_store(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.store(operand, mem_order);
}

/** Global function atomic_exchange. Calls exchange on SYCL atomic
 * object.
 * @param object The atomic object to exchange with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return the old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_exchange(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.exchange(operand, mem_order);
}

/** Global function atomic_compare_exchange. Calls compare_exchange on
 * SYCL
 * atomic object.
 * @param object The atomic object to compare_exchange with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return Whether comparison succeeds or fails
 */
template <typename T, access::address_space addressSpace>
inline cl_bool atomic_compare_exchange_strong(
    atomic<T, addressSpace> object, T& expected, T desired,
    memory_order success = memory_order::relaxed,
    memory_order fail = memory_order::relaxed) {
  return object.compare_exchange_strong(expected, desired, success, fail);
}

/** Global function atomic_add. Calls add on SYCL atomic object.
 * @param object The atomic object to add to
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_add(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_add(operand, mem_order);
}

/** Global function atomic_sub. Calls sub on SYCL atomic object.
 * @param object The atomic object to subtract from
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_sub(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_sub(operand, mem_order);
}

/** Global function atomic_and. Calls and on SYCL atomic object.
 * @param object The atomic object to and with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_and(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_and(operand, mem_order);
}

/** Global function atomic_or. Calls or on SYCL atomic object.
 * @param object The atomic object to or with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_or(atomic<T, addressSpace> object, T operand,
                         memory_order mem_order = memory_order::relaxed) {
  return object.fetch_or(operand, mem_order);
}

/** Global function atomic_xor. Calls XOR on SYCL atomic object.
 * @param object The atomic object to XOR with
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_xor(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_xor(operand, mem_order);
}

/** Global function atomic_min. Calculates min(object, operand), storing
 * the
 * result in object.
 * @param object The atomic object to perform min on
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_min(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_min(operand, mem_order);
}

/** Global function atomic_max. Calculates max(object, operand), storing
 * the
 * result in object.
 * @param object The atomic object to perform max on
 * @param mem_order The memory ordering to use. Only memory_order_relaxed
 * is supported.
 * @return The old value of *object
 */
template <typename T, access::address_space addressSpace>
inline T atomic_fetch_max(atomic<T, addressSpace> object, T operand,
                          memory_order mem_order = memory_order::relaxed) {
  return object.fetch_max(operand, mem_order);
}

#if SYCL_LANGUAGE_VERSION >= 202001
#ifndef __SYCL_DEVICE_ONLY__

namespace detail {
/** SYCL-2020 atomic_ref, implementing atomic operations common to all types.
 * An atomic_ref allows a referenced value to be acted on atomically.
 * Referencing 64bit types requires aspect::atomic64.
 *  @tparam T Type referenced by atomic_ref
 *  @tparam DefaultOrder The memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope The memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace The address space of the reference.
 */
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space addressSpace>
class atomic_ref_base {
 protected:
  static_assert(sizeof(T) <= 8, "Types larger than 64 bits are not supported.");

  /// Uses sycl::atomic to provide underlying functionality.
  atomic<T, addressSpace> m_data;

 public:
  using value_type = T;

  static constexpr size_t required_alignment = alignof(T);

  /// Return true if the atomic operations provided by this sycl::atomic are
  /// always lock-free.
  static constexpr bool is_always_lock_free =
      atomic<T, addressSpace>::is_always_lock_free;

  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;

  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;

  static constexpr memory_order default_read_modify_write_order = DefaultOrder;

  static constexpr memory_scope default_scope = DefaultScope;

  /// The value is true if the operations provided by sycl::atomic are
  /// lock-free.
  bool is_lock_free() const noexcept { return m_data.is_lock_free(); }

  /** Constructs an atomic reference to the provided variable.
   * @param target The value to operate atomically on.
   */
  explicit atomic_ref_base(T& target)
      : m_data{static_cast<multi_ptr<T, addressSpace>>(&target)} {}

  /** Constructs an atomic reference to the variable referenced by another
   * atomic reference.
   * @param other atomic_ref_base to copy.
   */
  atomic_ref_base(const atomic_ref_base& other) noexcept
      : m_data{other.m_data} {}

  atomic_ref_base& operator=(const atomic_ref_base&) = delete;

  /** Atomically store operand to the object referenced by atomic_ref.
   * @param operand The value to be stored.
   * @param order The memory ordering to use. Must be memory_order::relaxed,
   * memory_order::release or memory_order::seq_cst.
   * @param scope the memory scope to use.
   */
  void store(T operand, memory_order order = default_write_order,
             memory_scope scope = default_scope) const noexcept {
    (void)scope;
    m_data.store(operand, order);
    return;
  }

  /** Stores a value to the referenced location.
   * Equivalent to store(desired).
   * @param desired The value to store.
   * @return input parameter desired.
   */
  T operator=(T desired) const noexcept {
    m_data.store(desired, default_write_order);
    return desired;
  }

  /** Atomically loads the value of the referenced variable.
   * @param order The memory order to use. Must be memory_order::relaxed,
   * memory_order::release or memory_order::seq_cst.
   * @param scope the memory scope to use.
   * @return the value of the referenced variable.
   */
  T load(memory_order order = default_read_order,
         memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return m_data.load(order);
  }

  /** Equivalent to load().
   */
  operator T() const noexcept { return load(); }

  /** Atomically replace the value of the object referenced by this
   * atomic_ref with a given value. Only supported for 64-bit data types on
   * devices supporting aspect::atomic64.
   * @param operand The value to store at the referenced location.
   * @param order The memory order to use.
   * @param scope The memory scope to use.
   * Returns the original value stored.
   */
  T exchange(T operand, memory_order order = default_read_modify_write_order,
             memory_scope scope = default_scope) const noexcept {
    (void)order;
    (void)scope;
    return m_data.exchange(operand, default_read_modify_write_order);
  }

  /** Atomically compare the value referenced by this atomic_ref to an
   * expected value. If equal, attempt to replace the referenced value with a
   * desired value and return true. Otherwise, the referenced value is loaded
   * into expected and the function returns false. Weak indicates that the
   * function is allowed to fail, returning false, even when the expected value
   * matches the referenced value.
   * @param expected Value to compare to referenced value. On failure, the
   * referenced value is loaded into this.
   * @param desired The value to set the referenced value to on success.
   * @param success The memory order to use on success.
   * @param failure The memory order to use on failure.
   * @param scope The memory scope to use. Defaults to default_scope.
   * @return True on exchange of variables. False otherwise.
   */
  bool compare_exchange_weak(T& expected, T desired, memory_order success,
                             memory_order failure,
                             memory_scope scope = default_scope) const
      noexcept {
    (void)scope;
    return m_data.compare_exchange_weak(expected, desired, success, failure);
  }

  /** Atomically compare the value referenced by this atomic_ref to an
   * expected value. If equal, attempt to replace the referenced value with a
   * desired value and return true. Otherwise, the referenced value is loaded
   * into expected and the function returns false. Weak indicates that the
   * function is allowed to fail, returning false, even when the expected value
   * matches the referenced value.
   * @param expected Value to compare to referenced value. On failure, the
   * referenced value is loaded into this.
   * @param desired The value to set the referenced value to on success.
   * @param order The memory order to use. Defaults to
   * default_read_modify_write_order.
   * @param scope The memory scope to use. Defaults to default_scope.
   * @return True on exchange of variables. False otherwise.
   */
  bool compare_exchange_weak(
      T& expected, T desired,
      memory_order order = default_read_modify_write_order,
      memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return m_data.compare_exchange_weak(expected, desired, order, order);
  }

  /** Atomically compare the value referenced by this atomic_ref to an
   * expected value. If equal, replaces the referenced value with a
   * desired value and return true. Otherwise, the referenced value is loaded
   * into expected and the function returns false.
   * @param expected Value to compare to referenced value. On failure, the
   * referenced value is loaded into this.
   * @param desired The value to set the referenced value to on success.
   * @param success The memory order to use on success.
   * @param failure The memory order to use on failure.
   * @param scope The memory scope to use. Defaults to default_scope.
   * @return True on exchange of variables. False otherwise.
   */
  bool compare_exchange_strong(T& expected, T desired, memory_order success,
                               memory_order failure,
                               memory_scope scope = default_scope) const
      noexcept {
    (void)scope;
    return this->m_data.compare_exchange_strong(expected, desired, success,
                                                failure);
  }

  /** Atomically compare the value referenced by this atomic_ref to an
   * expected value. If equal, attempt to replace the referenced value with a
   * desired value and return true. Otherwise, the referenced value is loaded
   * into expected and the function returns false.
   * @param expected Value to compare to referenced value. On failure, the
   * referenced value is loaded into this.
   * @param desired The value to set the referenced value to on success.
   * @param order The memory order to use. Defaults to
   * default_read_modify_write_order.
   * @param scope The memory scope to use. Defaults to default_scope.
   * @return True on exchange of variables. False otherwise.
   */
  bool compare_exchange_strong(
      T& expected, T desired,
      memory_order order = default_read_modify_write_order,
      memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.compare_exchange_strong(expected, desired, order,
                                                order);
  }
};
}  // namespace detail

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space addressSpace, class Enable>
class atomic_ref : public detail::atomic_ref_base<T, DefaultOrder, DefaultScope,
                                                  addressSpace> {
 protected:
  using base_t =
      detail::atomic_ref_base<T, DefaultOrder, DefaultScope, addressSpace>;

 public:
  static constexpr memory_order default_read_order = base_t::default_read_order;

  static constexpr memory_order default_write_order =
      base_t::default_write_order;

  static constexpr memory_order default_read_modify_write_order =
      base_t::default_read_modify_write_order;

  static constexpr memory_scope default_scope = DefaultScope;

  explicit atomic_ref(T& target)
      : detail::atomic_ref_base<T, DefaultOrder, DefaultScope, addressSpace>(
            target) {}

  /** Atomically store to the reference value.
   * @param desired The value to store.
   * @return The input value desired.
   */
  T operator=(T desired) const noexcept {
    this->store(desired);
    return desired;
  }
};

/** SYCL-2020 atomic_ref specialized for integral types. It implements
 * atomic operations on referenced values. Referencing 64bit values requires
 * aspect::atomic64.
 *  @tparam Integral The integral type referenced by atomic_ref
 *  @tparam DefaultOrder The memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope The memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace The address space of the reference.
 */
template <typename Integral, memory_order DefaultOrder,
          memory_scope DefaultScope, access::address_space addressSpace>
class atomic_ref<Integral, DefaultOrder, DefaultScope, addressSpace,
                 typename std::enable_if_t<std::is_integral_v<Integral>>>
    : public detail::atomic_ref_base<Integral, DefaultOrder, DefaultScope,
                                     addressSpace> {
  /* All other members from detail::atomic_ref_base<Integral> are available */
 protected:
  using base_t = detail::atomic_ref_base<Integral, DefaultOrder, DefaultScope,
                                         addressSpace>;

 public:
  using difference_type = typename base_t::value_type;

  static constexpr memory_order default_read_order = base_t::default_read_order;

  static constexpr memory_order default_write_order =
      base_t::default_write_order;

  static constexpr memory_order default_read_modify_write_order =
      base_t::default_read_modify_write_order;

  static constexpr memory_scope default_scope = DefaultScope;

  /** Constructs an atomic_ref from given variable.
   * @param target The value to operate atomically on with atomic_ref.
   */
  explicit atomic_ref(Integral& target)
      : detail::atomic_ref_base<Integral, DefaultOrder, DefaultScope,
                                addressSpace>(target) {}

  /** Atomically store to the reference value.
   * @param desired The value to store.
   * @return the input variable desired.
   */
  Integral operator=(Integral desired) const noexcept {
    this->store(desired);
    return desired;
  }

  /** Atomically add the operand to the referenced value. Returns the
   * original value of the referenced value.
   * @param operand The value to add (addend) to the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced variable.
   */
  Integral fetch_add(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_add(operand, order);
  }

  /** Atomically subtract the operand to the referenced value. Returns the
   * original value of the referenced value.
   * @param operand The value to subtract from the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced value.
   */
  Integral fetch_sub(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_sub(operand, order);
  }

  /** Atomically perform a bitwise 'and' operation with the referenced
   * value and the operand. Returns the original value of the referenced value.
   * @param operand The value to and with the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced value.
   */
  Integral fetch_and(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_and(operand, order);
  }

  /** Atomically perform a bitwise 'or' operation with the referenced
   * value and the operand. Returns the original value of the referenced value.
   * @param operand The value to or with the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced value.
   */
  Integral fetch_or(Integral operand,
                    memory_order order = default_read_modify_write_order,
                    memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_or(operand, order);
  }

  /** Atomically perform a bitwise 'XOR' operation with the referenced
   * value and the operand. Returns the original value of the referenced value.
   * @param operand The value to XOR with the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced value.
   */
  Integral fetch_xor(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_xor(operand, order);
  }

  /** Atomically computes the minimum of the referenced value and the operand,
   * and stores it in the referenced value. Returns the original value of the
   * referenced value.
   * @param operand The value to compute the minimum of with the referenced
   * value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return the original value of the referenced value.
   */
  Integral fetch_min(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_min(operand, order);
  }

  /** Atomically computes the maximum of the referenced value and the operand,
   * and stores it in the referenced value. Returns the original value of the
   * referenced value.
   * @param operand The value to compute the maximum of with the referenced
   * value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced value.
   */
  Integral fetch_max(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_max(operand, order);
  }

  /// Postincrement the referenced value atomically.
  Integral operator++(int) const noexcept { return fetch_add(1); }

  /// Postdecrement the referenced value atomically.
  Integral operator--(int) const noexcept { return fetch_sub(1); }

  /// Preincrement the referenced value atomically.
  Integral operator++() const noexcept { return fetch_add(1) + 1; }

  /// Predecrement the referenced value atomically.
  Integral operator--() const noexcept { return fetch_sub(1) - 1; }

  /** Atomic addition assignment.
   *  @param operand The value to add to the referenced value.
   */
  Integral operator+=(Integral operand) const noexcept {
    return fetch_add(operand);
  }

  /** Atomic subtraction assignment.
   *  @param operand The value to subtract from the referenced value.
   */
  Integral operator-=(Integral operand) const noexcept {
    return fetch_sub(operand);
  }

  /** Atomic bitwise 'and' assignment.
   *  @param operand The value to 'and' with the referenced value.
   */
  Integral operator&=(Integral operand) const noexcept {
    return fetch_and(operand);
  }

  /** Atomic bitwise 'or' assignment.
   *  @param operand The value to 'or' with the referenced value.
   */
  Integral operator|=(Integral operand) const noexcept {
    return fetch_or(operand);
  }

  /** Atomic bitwise 'XOR' assignment.
   *  @param operand The value to XOR with the referenced value.
   */
  Integral operator^=(Integral operand) const noexcept {
    return fetch_xor(operand);
  }
};

/** SYCL-2020 atomic_ref specialized for floating-point types. It implements
 * atomic operations on referenced values. Referencing 64bit types requires
 * aspect::atomic64.
 *  @tparam Floating The floating-point type referenced by atomic_ref
 *  @tparam DefaultOrder The memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope The memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace The address space of the reference.
 */
template <typename Floating, memory_order DefaultOrder,
          memory_scope DefaultScope, access::address_space addressSpace>
class atomic_ref<Floating, DefaultOrder, DefaultScope, addressSpace,
                 typename std::enable_if_t<std::is_floating_point_v<Floating>>>
    : public detail::atomic_ref_base<Floating, DefaultOrder, DefaultScope,
                                     addressSpace> {
  /* All other members from detail::atomic_ref_base<Floating> are available */
 protected:
  using base_t = detail::atomic_ref_base<Floating, DefaultOrder, DefaultScope,
                                         addressSpace>;

 public:
  using difference_type = typename base_t::value_type;

  static constexpr memory_order default_read_order = base_t::default_read_order;

  static constexpr memory_order default_write_order =
      base_t::default_write_order;

  static constexpr memory_order default_read_modify_write_order =
      base_t::default_read_modify_write_order;

  static constexpr memory_scope default_scope = DefaultScope;

  /** Constructor.
   * @param target The value to operate atomically on.
   */
  explicit atomic_ref(Floating& target)
      : detail::atomic_ref_base<Floating, DefaultOrder, DefaultScope,
                                addressSpace>(target) {}

  /** Atomically store to the reference value.
   * @param desired The value to store.
   */
  Floating operator=(Floating desired) const noexcept {
    this->store(desired);
    return desired;
  }

  /** Atomically add the operand to the referenced value. Returns the
   * original value of the referenced value.
   * @param operand The value to add to the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced value.
   */
  Floating fetch_add(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    /* std::atomic<Floating>::fetch_add is C++ 2020 so fetch add must be
    emulated. */
    Floating old{this->load(order)};
    Floating sum;
    do {
      sum = old + operand;
    } while (!this->compare_exchange_weak(old, sum, order, scope));
    return old;
  }

  /** Atomically subtract the operand from the referenced value.
   * Returns the original value of the referenced value.
   * @param operand The value to subtract from the referenced value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced value.
   */
  Floating fetch_sub(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    /* std::atomic<Floating>::fetch_add is C++ 2020 so fetch add must be
    emulated. */
    Floating old{this->load(order)};
    Floating difference;
    do {
      difference = old - operand;
    } while (!this->compare_exchange_weak(old, difference, order, scope));

    return old;
  }

  /** Atomically computes the minimum of the referenced value and the
   * operand. Returns the original value of the referenced value.
   * @param operand The value to compute the minimum of with the referenced
   * value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced value.
   */
  Floating fetch_min(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_min(operand, order);
  }

  /** Atomically computes the maximum of the referenced value and the
   * operand. Returns the original value of the referenced value.
   * @param operand The value to compute the maximum of with the referenced
   * value.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced value.
   */
  Floating fetch_max(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.fetch_max(operand, order);
  }

  /** Atomic addition assignment.
   *  @param operand The value to add (addend) to the referenced value.
   */
  Floating operator+=(Floating operand) const noexcept {
    return fetch_add(operand);
  }

  /** Atomic subtraction assignment.
   *  @param operand The value to subtract from the referenced value.
   */
  Floating operator-=(Floating operand) const noexcept {
    return fetch_sub(operand);
  }
};

/** SYCL-2020 atomic_ref specialized for pointer types. It implements
 * atomic operations on referenced pointers. Referencing 64bit pointers requires
 * aspect::atomic64.
 *  @tparam T* The pointer type referenced by atomic_ref
 *  @tparam DefaultOrder The memory order to use in default arguments or where
 *  no ordering parameter can be given.
 *  @tparam DefaultScope The memory scope to use in default arguments or where
 *  no scope parameter can be given.
 *  @tparam addressSpace The address space of the reference.
 */
template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space addressSpace>
class atomic_ref<T*, DefaultOrder, DefaultScope, addressSpace>
    : public detail::atomic_ref_base<T*, DefaultOrder, DefaultScope,
                                     addressSpace> {
 protected:
  using base_t =
      detail::atomic_ref_base<T*, DefaultOrder, DefaultScope, addressSpace>;

 public:
  using difference_type = ptrdiff_t;

  static constexpr memory_order default_read_order = base_t::default_read_order;

  static constexpr memory_order default_write_order =
      base_t::default_write_order;

  static constexpr memory_order default_read_modify_write_order =
      base_t::default_read_modify_write_order;

  static constexpr memory_scope default_scope = DefaultScope;

  static constexpr size_t required_alignment = alignof(T*);

  static constexpr bool is_always_lock_free =
      std::atomic<T*>::is_always_lock_free;

  /** Constructor.
   * @param target The value to operate atomically on.
   */
  explicit atomic_ref(T*& target)
      : detail::atomic_ref_base<T*, DefaultOrder, DefaultScope, addressSpace>(
            target) {}

  /** Atomically store an address to the referenced pointer.
   * @param desired The address to store.
   */
  T* operator=(T* desired) const noexcept {
    this->store(desired);
    return desired;
  }

  /** Atomically add the operand to the referenced pointer. Returns the
   * original value of the referenced pointer.
   * @param operand The value to add (addend) to the referenced pointer.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced pointer.
   */
  T* fetch_add(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.m_data->fetch_add(
        operand, static_cast<std::memory_order>(order));
  }

  /** Atomically subtract the operand from the referenced pointer.
   * Returns the original value of the referenced pointer.
   * @param operand The value to subtract from the referenced pointer.
   * @param memory_order The memory order to use.
   * @param scope The memory scope to use.
   * @return The original value of the referenced pointer.
   */
  T* fetch_sub(difference_type operand,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {
    (void)scope;
    return this->m_data.m_data->fetch_sub(
        operand, static_cast<std::memory_order>(order));
  }

  /// Postincrement the referenced pointer atomically.
  T* operator++(int) const noexcept { return fetch_add(1); }

  /// Postdecrement the referenced pointer atomically.
  T* operator--(int) const noexcept { return fetch_sub(1); }

  /// Preincrement the referenced pointer atomically.
  T* operator++() const noexcept { return fetch_add(1) + 1; }

  /// Predecrement the referenced pointer atomically.
  T* operator--() const noexcept { return fetch_sub(1) - 1; }

  /** Atomic addition assignment.
   * @param operand The value to add (addend) to the referenced pointer.
   */
  T* operator+=(difference_type operand) const noexcept {
    return fetch_add(operand);
  }

  /** Atomic subtraction assignment.
   * @param operand The value to subtract from the referenced pointer.
   */
  T* operator-=(difference_type operand) const noexcept {
    return fetch_sub(operand);
  }
};

#endif  //__SYCL_DEVICE_ONLY__
#endif  // SYCL_LANGUAGE_VERSION >= 202001

namespace detail {

/** Retrieves the address space, suitable for use in an atomic,
 *        from the access target.
 *
 *        The general case is to use global_space - this value needs to be
 *        available even for cases that the atomic class doesn't support.
 *
 * @tparam accessTarget Access target to retrieve the address space for
 */
template <access::target accessTarget>
struct get_atomic_address_space {
  /** Most targets will correspond to the global address space,
   *        even though it's only valid for access::target::global_buffer
   */
  static constexpr auto value = access::address_space::global_space;
};

/** Retrieves the address space, suitable for use in an atomic,
 *        from the access::target::local access target
 */
template <>
struct get_atomic_address_space<access::target::local> {
  /** The local target corresponds to the local address space
   */
  static constexpr auto value = access::address_space::local_space;
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ATOMIC_H_
