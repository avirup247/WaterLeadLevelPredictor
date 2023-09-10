/*****************************************************************************
  Copyright (C) 2002-2020 Codeplay Software Limited

  All Rights Reserved.
  Codeplay's ComputeCpp
*****************************************************************************/

/**
  @file sub_group.h
  @brief This file contains the API for cl::sycl::experimental::sub_group.
*/

#ifndef RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_SUB_GROUP_H_
#define RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_SUB_GROUP_H_

#include "SYCL/common.h"
#include "SYCL/functional.h"
#include "SYCL/group_base.h"
#include "SYCL/id.h"
#include "SYCL/memory_scope.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/range.h"

#include <functional>
#include <limits>

namespace cl {
namespace sycl {

namespace detail {

/** @brief Implementation of subgroup barrier.
 * @param fenceSpace Unused.
 */
inline void sub_group_barrier_impl(access::fence_space fenceSpace) {
  (void)fenceSpace;
#ifdef __SYCL_DEVICE_ONLY__
  detail::sub_group_barrier();
#endif  // __SYCL_DEVICE_ONLY__
  // Subgroup barrier is a no-op on the host as the subgroup size is always 1.
}

}  // namespace detail

COMPUTECPP_INLINE_EXPERIMENTAL
namespace experimental {

/// @brief The sub_group class is an interface for subgroups.
struct sub_group {
 public:
#if SYCL_LANGUAGE_VERSION >= 202001

  static constexpr memory_scope fence_scope = memory_scope::sub_group;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  template <int dimensions>
  friend class ::cl::sycl::nd_item;

  /** @brief Get an id representing the index of the subgroup within the
   * work-group.
   * @return The subgroup id.
   */
  inline id<1> get_group_id() const noexcept { return id<1>(m_subGroupId); }

  /** @brief Get the number of subgroups within the work-group.
   * @return The subgroup range.
   */
  inline range<1> get_group_range() const noexcept {
    return range<1>(m_subGroupRange);
  }

  /** @brief Get the number of subgroups per work-group in the uniform region of
   * the nd-range.
   * @return The subgroup uniform range.
   */
  inline range<1> get_uniform_group_range() const noexcept {
    return range<1>(m_uniformSubGroupRange);
  }

  /** @brief Get an id representing the index of the work-item within the
   * subgroup.
   * @return The local id.
   */
  inline id<1> get_local_id() const noexcept { return id<1>(m_localId); }

  /** @brief Get the number of work-items in the subgroup.
   * @return The local range.
   */
  inline range<1> get_local_range() const noexcept {
    return range<1>(m_localRange);
  }

  /** @brief Get the maximum number of work-items in any subgroups within the
   * nd-range.
   * @return The maximum local range.
   */
  inline range<1> get_max_local_range() const noexcept {
    return range<1>(m_maxLocalRange);
  }

  /** @brief Synchronizes all work-items in a subgroup.
   * @param fenceSpace Barrier fence space.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(202001,
                                    "Use group_barrier(sub_group) instead.")
  inline void barrier(access::fence_space fenceSpace =
                          access::fence_space::global_and_local) const {
    detail::sub_group_barrier_impl(fenceSpace);
  }

  /** @brief Logical any function.
   * @param predicate Value of the predicate for the current work-item.
   * @return True iff @p predicate is true for any work-item in the subgroup.
   */
  inline bool any(bool predicate) const { return predicate; }

  /** @brief Logical all function.
   * @param predicate Value of the predicate for the current work-item.
   * @return True iff @p predicate is true for all work-items in the subgroup.
   */
  inline bool all(bool predicate) const { return predicate; }

  /** @brief Broadcast @p x from the specified @p localId to all work-items
   * within the subgroup.
   * @tparam T
   * @param x Value to broadcast.
   * @param localId Must be the same id for all work-items in the subgroup.
   * @return The value broadcasted by @p localId.
   */
  template <class T>
  inline T broadcast(T x, id<1> localId) const {
    (void)localId;
    return x;
  }

  /** @brief Reduce the values @p x from all work-items.
   * The initialization value is chosen to be the identity element of @p
   * binaryOp.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param binaryOp Reduce operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the reduction.
   */
  template <class T, class BinaryOp>
  inline T reduce(T x, BinaryOp binaryOp) const {
    (void)binaryOp;
    return x;
  }

  /** @brief Reduce the values @p x from all work-items.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param init Initialization value.
   * @param binaryOp Reduce operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the reduction.
   */
  template <class T, class BinaryOp>
  inline T reduce(T x, T init, BinaryOp binaryOp) const {
    return binaryOp(init, x);
  }

  /** @brief Exclusive scan of the values @p x from all work-items.
   * The value returned on the `i`-th work-item is the scan of the first `i`
   * work-items. The initialization value is chosen to be the identity element
   * of @p binaryOp.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param binaryOp Scan operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the exclusive scan of the first `i` work-items.
   */
  template <class T, class BinaryOp>
  inline T exclusive_scan(T x, BinaryOp binaryOp) const {
    (void)x;
    (void)binaryOp;
    return detail::identity_value<T, BinaryOp>::value;
  }

  /** @brief Exclusive scan of the values @p x from all work-items.
   * The value returned on the `i`-th work-item is the scan of the first `i`
   * work-items.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param init Initialization value.
   * @param binaryOp Scan operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the exclusive scan of the first `i` work-items.
   */
  template <class T, class BinaryOp>
  inline T exclusive_scan(T x, T init, BinaryOp binaryOp) const {
    (void)x;
    (void)binaryOp;
    return init;
  }

  /** @brief Inclusive scan of the values @p x from all work-items.
   * The value returned on the `i`-th work-item is the scan of the first `i`
   * work-items. The initialization value is chosen to be the identity element
   * of @p binaryOp.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param binaryOp Scan operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the inclusive scan of the first `i` work-items.
   */
  template <class T, class BinaryOp>
  inline T inclusive_scan(T x, BinaryOp binaryOp) const {
    (void)binaryOp;
    return x;
  }

  /** @brief Inclusive scan of the values @p x from all work-items.
   * The value returned on the `i`-th work-item is the scan of the first `i`
   * work-items.
   * @tparam T
   * @tparam BinaryOp
   * @param x Value to reduce for the current work-item.
   * @param init Initialization value.
   * @param binaryOp Scan operation, must be one of @ref cl::sycl::plus, @ref
   * cl::sycl::minimum, @ref cl::sycl::maximum.
   * @return The result of the inclusive scan of the first `i` work-items.
   */
  template <class T, class BinaryOp>
  inline T inclusive_scan(T x, BinaryOp binaryOp, T init) const {
    return binaryOp(x, init);
  }

  /** @brief Exchange values of @p x between work-items in a subgroup.
   * @tparam T
   * @param x Value to send.
   * @param localId Work-item id to retrieve.
   * @return The value sent by the work-item @p localId.
   */
  template <class T>
  inline T shuffle(T x, id<1> localId) const {
    (void)localId;
    return x;
  }

  /** @brief Exchange values of @p x between work-items in a subgroup.
   * @ref shuffle_down is a specialized version of @ref shuffle that may be
   * optimized. The return value is unspecified if `get_local_id() + delta >=
   * get_local_range()`.
   * @tparam T
   * @param x Value to send.
   * @param delta Offset added to the calling work-item's id.
   * @return The value sent by the work-item whose id is `get_local_id() +
   * delta`.
   */
  template <class T>
  inline T shuffle_down(T x, uint32_t delta) const {
    (void)delta;
    return x;
  }

  /** @brief Exchange values of @p x between work-items in a subgroup.
   * @ref shuffle_up is a specialized version of @ref shuffle that may be
   * optimized. The return value is unspecified if `get_local_id() - delta < 0`.
   * @tparam T
   * @param x Value to send.
   * @param delta Offset substracted to the calling work-item's id.
   * @return The value sent by the work-item whose id is `get_local_id() -
   * delta`.
   */
  template <class T>
  inline T shuffle_up(T x, uint32_t delta) const {
    (void)delta;
    return x;
  }

  /** @brief Exchange values of @p x between work-items in a subgroup.
   * @ref shuffle_xor is a specialized version of @ref shuffle that may be
   * optimized.
   * @tparam T
   * @param x Value to send.
   * @param mask Mask applied to the calling work-item's, must be constant
   * across the subgroup.
   * @return The value sent by the work-item whose id is `get_local_id() ^
   * mask`.
   */
  template <class T>
  inline T shuffle_xor(T x, id<1> mask) const {
    (void)mask;
    return x;
  }

  /** @brief Exchange values of @p x and @p y between work-items in a subgroup.
   * Two inputs shuffles can be thought as a one input shuffle on a virtual
   * subgroup twice as big.
   * @tparam T
   * @param x Value to send.
   * @param y Value to send.
   * @param localId Work-item id to retrieve, must be between 0 and twice the
   * subgroup size.
   * @return The value @p x sent by the work-item whose id is @p localId if @p
   * localId is between 0 and the subgroup size. Return the value @p y sent the
   * work-item whose id is `localId % get_local_range()` otherwise.
   */
  template <class T>
  inline T shuffle(T x, T y, id<1> localId) const {
    return localId.get(0) < get_local_range().get(0) ? x : y;
  }

  /** @brief Exchange values of @p x and @p y between work-items in a subgroup.
   * Two inputs shuffles can be thought as a one input shuffle on a virtual
   * subgroup twice as big. @ref shuffle_down is a specialized version of @ref
   * shuffle that may be optimized.
   * @tparam T
   * @param x Value to send.
   * @param y Value to send.
   * @param delta Offset added to the calling work-item's id, must be less than
   * the subgroup size.
   * @return The value @p x sent by the work-item whose id is `get_local_id() +
   * delta` if the result is between 0 and the subgroup size. Return the value
   * @p y sent by the work-item whose id is `(get_local_id() + delta) %
   * get_local_range()` otherwise.
   */
  template <class T>
  inline T shuffle_down(T x, T y, uint32_t delta) const {
    return (get_local_id().get(0) + delta) < get_local_range().get(0) ? x : y;
  }

  /** @brief Exchange values of @p x and @p y between work-items in a subgroup.
   * Two inputs shuffles can be thought as a one input shuffle on a virtual
   * subgroup twice as big. @ref shuffle_up is a specialized version of @ref
   * shuffle that may be optimized.
   * @tparam T
   * @param x Value to send.
   * @param y Value to send.
   * @param delta Offset added to the calling work-item's id, must be less than
   * the subgroup size.
   * @return The value @p x sent by the work-item whose id is `get_local_id() -
   * delta` if the result is between 0 and the subgroup size. Return the value
   * @p y sent by the work-item whose id is `(get_local_id() + delta) %
   * get_local_range()` otherwise.
   */
  template <class T>
  inline T shuffle_up(T x, T y, uint32_t delta) const {
    return (get_local_id().get(0) - delta) < get_local_range().get(0) ? x : y;
  }

  /** @brief Load contiguous data from @p src.
   * @tparam T
   * @tparam Space
   * @param src Pointer to the data to load, must be the same across all
   * work-items in the subgroup.
   * @return T Data corresponding to `src + get_local_id()`.
   */
  template <class T, access::address_space Space>
  inline T load(const multi_ptr<T, Space> src) const {
    return src.get()[0];
  }

  /** @brief Load contiguous data from @p src.
   * @tparam T
   * @tparam N Number of element to load per work-item.
   * @tparam Space
   * @param src Pointer to the data to load, must be the same across all
   * work-items in the subgroup.
   * @return T Data corresponding to `src + get_local_id() + i *
   * get_max_local_range()` for `i` between 0 and @p N.
   */
  template <class T, int N, access::address_space Space>
  inline vec<T, N> load(const multi_ptr<T, Space> src) const {
    vec<T, N> res;
    res.load(0, src);
    return res;
  }

  /** @brief Store contiguous data to @p dst.
   * @tparam T
   * @tparam Space
   * @param dst Pointer to the data to store, must be the same across all
   * work-items in the subgroup.
   * @param x Value to store at `dst + get_local_id()`.
   */
  template <class T, access::address_space Space>
  inline void store(multi_ptr<T, Space> dst, const T& x) const {
    dst.get()[0] = x;
  }

  /** @brief Store contiguous data to @p dst.
   * @tparam T
   * @tparam N  Number of element to store per work-item.
   * @tparam Space
   * @param dst Pointer to the data to store, must be the same across all
   * work-items in the subgroup.
   * @param x Values to store at `dst + get_local_id() + i *
   * get_max_local_range()` for `i` between 0 and @p N.
   */
  template <class T, int N, access::address_space Space>
  inline void store(multi_ptr<T, Space> dst, const vec<T, N>& x) const {
    x.store(0, dst);
  }

 private:
  /** @brief Construct a new subgroup object.
   * Constructor is private to enforce construction by
   * nd_item::get_sub_group().
   * @param subGroupId @see get_group_id.
   * @param subGroupRange @see get_group_range.
   * @param localId @see get_local_id.
   * @param localRange @see get_local_range.
   */
  sub_group(size_t subGroupId, size_t subGroupRange,
            size_t uniformSubGroupRange, size_t localId, size_t localRange,
            size_t maxLocalRange)
      : m_subGroupId(subGroupId),
        m_subGroupRange(subGroupRange),
        m_uniformSubGroupRange(uniformSubGroupRange),
        m_localId(localId),
        m_localRange(localRange),
        m_maxLocalRange(maxLocalRange) {}

  size_t m_subGroupId;
  size_t m_subGroupRange;
  size_t m_uniformSubGroupRange;
  size_t m_localId;
  size_t m_localRange;
  size_t m_maxLocalRange;
};

}  // namespace experimental

namespace codeplay {
// Alias for backward compatibility
COMPUTECPP_DEPRECATED_API("Use experimental::sub_group instead")
typedef experimental::sub_group sub_group;
}  // namespace codeplay

}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_SUB_GROUP_H_
