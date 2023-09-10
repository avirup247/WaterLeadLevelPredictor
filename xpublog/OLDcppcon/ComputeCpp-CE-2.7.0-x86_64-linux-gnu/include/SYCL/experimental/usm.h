/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_H_
#define RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_H_

#include "SYCL/base.h"
#include "SYCL/experimental/usm_definitions.h"
#include "SYCL/predefines.h"

#include <cstddef>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

class context;
class device;
class queue;

namespace detail {

class COMPUTECPP_EXPORT usm_allocator_base {
 protected:
  /** @brief Allows copy construction from a rebound derived allocator
   */
  struct rebind_copy_tag {};

 public:
  /** @brief Constructs an allocator using default selected device and context
   * @param allocationType Type of allocations to perform
   * @param alignment Requested alignment
   */
  usm_allocator_base(experimental::usm::alloc allocationType, size_t alignment);

  /** @brief Copy constructor from a derived allocator of rebound type
   * @param copy Other allocator
   */
  usm_allocator_base(const usm_allocator_base& copy, rebind_copy_tag);

  /** @brief Constructs an allocator using default selected device and context
   * @param allocationType Type of allocations to perform
   * @param alignment Requested alignment
   * @param ctx Allocation context
   * @param dev Device to allocate for. Must be associated with the context.
   */
  usm_allocator_base(experimental::usm::alloc allocationType, size_t alignment,
                     const sycl::context& ctx, const sycl::device& dev);

  /** @brief Constructs an allocator using default selected device and context
   * @param allocationType Type of allocations to perform
   * @param alignment Requested alignment
   * @param q Queue containing the allocation context and device
   */
  usm_allocator_base(experimental::usm::alloc allocationType, size_t alignment,
                     const sycl::queue& q);

  /** @brief Performs a USM allocation
   * @param size Number of bytes to allocate
   * @return A pointer to the memory block allocated by the function
   * @throw std::bad_alloc if allocation fails
   */
  void* allocate(size_t size);

  /** @brief Frees USM allocated memory
   * @param ptr Pointer to USM allocated memory.
   *        Passing other kinds of pointers is not guaranteed to work.
   * @param size The size of allocated memory
   */
  void deallocate(void* ptr, size_t size);

  /// @cond COMPUTECPP_DEV

  /** @brief Returns the allocator implementation object
   * @return Allocator PIMPL
   * @internal
   */
  const dusm_alloc_shptr& get_impl() const noexcept { return m_impl; }

  /** @brief Checks if the two allocators compare equally
   *        for all non-templated values w.r.t. the templated derived allocator
   * @return True if allocators are the same
   * @internal
   */
  bool core_equals(const usm_allocator_base& rhs) const noexcept;

  /// COMPUTECPP_DEV @endcond

 protected:
  /** @brief Throws an exception saying that the allocator function
   *        is not supported on a device allocation
   * @param functionName Unsupported function
   * @throw cl::sycl::feature_not_supported
   */
  void throw_not_supported_on_device(const char* functionName) const;

  /** @brief Throws an exception saying that the allocator function
   *        is not implemented yet
   * @param functionName Unimplemented function
   * @throw cl::sycl::feature_not_supported
   */
  void throw_not_implemented(const char* functionName) const;

 private:
  /// Allocator implementation object
  dusm_alloc_shptr m_impl;
};

}  // namespace detail

COMPUTECPP_INLINE_EXPERIMENTAL
namespace experimental {

/** @brief Performs a USM allocation on the device
 * @param size Number of bytes to allocate
 * @param dev Device to allocate on
 * @param ctx Context where the device is located
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_device(size_t size, const device& dev,
                                      const context& ctx);

/** @brief Performs a USM allocation on the device
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param dev Device to allocate on
 * @param ctx Context where the device is located
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_device(size_t count, const device& dev, const context& ctx) {
  return static_cast<T*>(malloc_device(count * sizeof(T), dev, ctx));
}

/** @brief Performs a USM allocation on the device
 * @param size Number of bytes to allocate
 * @param q Queue containing the context and device to allocate on
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_device(size_t size, const queue& q);

/** @brief Performs a USM allocation on the device
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param q Queue containing the context and device to allocate on
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_device(size_t count, const queue& q) {
  return static_cast<T*>(malloc_device(count * sizeof(T), q));
}

/** @brief Performs a USM allocation on the device with a specific alignment
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate
 * @param dev Device to allocate on
 * @param ctx Context where the device is located
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_device(size_t alignment, size_t size,
                                             const device& dev,
                                             const context& ctx);

/** @brief Performs a USM allocation on the device with a specific alignment
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param dev Device to allocate on
 * @param ctx Context where the device is located
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_device(size_t alignment, size_t count, const device& dev,
                        const context& ctx) {
  return static_cast<T*>(
      aligned_alloc_device(alignment, count * sizeof(T), dev, ctx));
}

/** @brief Performs a USM allocation on the device with a specific alignment
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate
 * @param q Queue containing the context and device to allocate on
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_device(size_t alignment, size_t size,
                                             const queue& q);

/** @brief Performs a USM allocation on the device with a specific alignment
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type
 * @param count Number of elements to allocate
 * @param q Queue containing the context and device to allocate on
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_device(size_t alignment, size_t count, const queue& q) {
  return static_cast<T*>(aligned_alloc_device(alignment, count * sizeof(T), q));
}

/** @brief Performs a host USM allocation.
 * @param size Number of bytes to allocate.
 * @param ctx Context whose devices can access the host allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_host(size_t size, const context& ctx);

/** @brief Performs a host USM allocation.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param ctx Context whose devices can access the host allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_host(size_t count, const context& ctx) {
  return static_cast<T*>(malloc_host(count * sizeof(T), ctx));
}

/** @brief Performs a host USM allocation.
 * @param size Number of bytes to allocate.
 * @param q Queue containing the context whose devices can access the host
 * allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_host(size_t size, const queue& q);

/** @brief Performs a host USM allocation.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context whose devices can access the host
 * allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_host(size_t count, const queue& q) {
  return static_cast<T*>(malloc_host(count * sizeof(T), q));
}

/** @brief Performs a host USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param ctx Context whose devices can access the host allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_host(size_t alignment, size_t size,
                                           const context& ctx);

/** @brief Performs a host USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param ctx Context whose devices can access the host allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_host(size_t alignment, size_t count, const context& ctx) {
  return static_cast<T*>(aligned_alloc_host(alignment, count * sizeof(T), ctx));
}

/** @brief Performs a host USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param q Queue containing the context whose devices can access the host
 * allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_host(size_t alignment, size_t size,
                                           const queue& q);

/** @brief Performs a host USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context whose devices can access the host
 * allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_host(size_t alignment, size_t count, const queue& q) {
  return static_cast<T*>(aligned_alloc_host(alignment, count * sizeof(T), q));
}

/** @brief Performs a shared USM allocation.
 * @param size Number of bytes to allocate.
 * @param dev Device to allocate on.
 * @param ctx Context whose devices can access the shared allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_shared(size_t size, const device& dev,
                                      const context& ctx);

/** @brief Performs a shared USM allocation.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param dev Device to allocate on.
 * @param ctx Context whose devices can access the shared allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_shared(size_t count, const device& dev, const context& ctx) {
  return static_cast<T*>(malloc_shared(count * sizeof(T), dev, ctx));
}

/** @brief Performs a shared USM allocation.
 * @param size Number of bytes to allocate.
 * @param q Queue the context and device to allocate on.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc_shared(size_t size, const queue& q);

/** @brief Performs a shared USM allocation.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc_shared(size_t count, const queue& q) {
  return static_cast<T*>(malloc_shared(count * sizeof(T), q));
}

/** @brief Performs a shared USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param dev Device to allocate on.
 * @param ctx Context whose devices can access the shared allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_shared(size_t alignment, size_t size,
                                             const device& dev,
                                             const context& ctx);

/** @brief Performs a shared USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param dev Device to allocate on.
 * @param ctx Context whose devices can access the shared allocation.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_shared(size_t alignment, size_t count, const device& dev,
                        const context& ctx) {
  return static_cast<T*>(
      aligned_alloc_shared(alignment, count * sizeof(T), dev, ctx));
}

/** @brief Performs a shared USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc_shared(size_t alignment, size_t size,
                                             const queue& q);

/** @brief Performs a shared USM allocation with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc_shared(size_t alignment, size_t count, const queue& q) {
  return static_cast<T*>(aligned_alloc_shared(alignment, count * sizeof(T), q));
}

/** @brief Performs a USM allocation.
 * @param size Number of bytes to allocate.
 * @param dev Device to allocate on, unused if @p allocKind is @ref usm::host.
 * @param ctx Context where the device is located.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc(size_t size, const device& dev,
                               const context& ctx, usm::alloc allocKind);

/** @brief Performs a USM allocation on the device.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param dev Device to allocate on, unused if @p allocKind is @ref usm::host.
 * @param ctx Context where the device is located.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc(size_t count, const device& dev, const context& ctx,
          usm::alloc allocKind) {
  return static_cast<T*>(malloc(count * sizeof(T), dev, ctx, allocKind));
}

/** @brief Performs a USM allocation on the device.
 * @param size Number of bytes to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* malloc(size_t size, const queue& q,
                               usm::alloc allocKind);

/** @brief Performs a USM allocation on the device.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* malloc(size_t count, const queue& q, usm::alloc allocKind) {
  return static_cast<T*>(malloc(count * sizeof(T), q, allocKind));
}

/** @brief Performs a USM allocation on the device with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param dev Device to allocate on, unused if @p allocKind is @ref usm::host.
 * @param ctx Context where the device is located.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc(size_t alignment, size_t size,
                                      const device& dev, const context& ctx,
                                      usm::alloc allocKind);

/** @brief Performs a USM allocation on the device with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param dev Device to allocate on, unused if @p allocKind is @ref usm::host.
 * @param ctx Context where the device is located.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc(size_t alignment, size_t count, const device& dev,
                 const context& ctx, usm::alloc allocKind) {
  return static_cast<T*>(
      aligned_alloc(alignment, count * sizeof(T), dev, ctx, allocKind));
}

/** @brief Performs a USM allocation on the device with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @param size Number of bytes to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
COMPUTECPP_EXPORT void* aligned_alloc(size_t alignment, size_t size,
                                      const queue& q, usm::alloc allocKind);

/** @brief Performs a USM allocation on the device with a specific alignment.
 * @param alignment Allocation alignment, in bytes.
 *        Must be a power of two.
 *        Zero indicates default alignment.
 * @tparam T Element type.
 * @param count Number of elements to allocate.
 * @param q Queue containing the context and device to allocate on.
 * @param allocKind Allocation kind.
 * @return A pointer to the memory block allocated by the function.
 *         nullptr if allocation fails for any reason.
 */
template <typename T>
T* aligned_alloc(size_t alignment, size_t count, const queue& q,
                 usm::alloc allocKind) {
  return static_cast<T*>(
      aligned_alloc(alignment, count * sizeof(T), q, allocKind));
}

/** @brief Frees USM allocated memory
 * @param ptr Pointer to USM allocated memory.
 *        Passing other kinds of pointers is not guaranteed to work.
 * @param ctx Context where the allocation was performed
 */
COMPUTECPP_EXPORT void free(void* ptr, const context& ctx);

/** @brief Frees USM allocated memory
 * @param ptr Pointer to USM allocated memory.
 *        Passing other kinds of pointers is not guaranteed to work.
 * @param q Queue containing the context where the allocation was performed
 */
COMPUTECPP_EXPORT void free(void* ptr, const queue& q);

/** @brief Unified Shared Memory pointer type query.
 * @param ptr The pointer to query.
 * @param ctx The SYCL context to which the USM allocation belongs.
 * @return The USM allocation type for @p ptr if @p ptr falls inside a valid USM
 * allocation. If @p ctx is a host context returns usm::alloc::host. Returns
 * usm::alloc::unknown if @p ptr is not a valid USM allocation.
 */
COMPUTECPP_EXPORT usm::alloc get_pointer_type(const void* ptr,
                                              const context& ctx);

/** @brief Unified Shared Memory device for pointer query.
 * @param ptr The pointer to query.
 * @param ctx The SYCL context to which the USM allocation belongs.
 * @return The device associated with the USM allocation. If @p ctx is a host
 * context, returns the host device in @p ctx. If @p ptr is an allocation of
 * type usm::alloc::host, returns the first device in @p ctx.
 * @throws cl::sycl::runtime_error if @p ptr is not a valid USM allocation.
 */
COMPUTECPP_EXPORT device get_pointer_device(const void* ptr,
                                            const context& ctx);

/** @brief USM allocator
 * @tparam T Underlying data type
 * @tparam AllocKind What kind of allocations to perform
 * @tparam Alignment Allocation alignment
 */
template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator : public detail::usm_allocator_base {
 private:
  using base_t = detail::usm_allocator_base;

 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

 private:
  // Hide some functions
  using base_t::allocate;
  using base_t::deallocate;

 public:
  /** @brief Allows allocator rebinding
   * @tparam U New underlying data type
   */
  template <typename U>
  struct rebind {
    using other = usm_allocator<U, AllocKind, Alignment>;
  };

  /** @brief Constructs an allocator using default selected device and context
   */
  usm_allocator() noexcept : base_t{AllocKind, Alignment} {}

  /** @brief Copy constructor
   */
  usm_allocator(const usm_allocator&) noexcept = default;

  /** @brief Copy constructor from an allocator of different type
   * @tparam U New underlying data type
   * @param copy Allocator to copy
   */
  template <class U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment>& copy) noexcept
      : base_t{copy, usm_allocator_base::rebind_copy_tag{}} {}

  /** @brief Constructs an allocator that will allocate memory
   *        on the provided device within the context.
   * @param ctx Allocation context
   * @param dev Device to allocate for. Must be associated with the context.
   */
  usm_allocator(const context& ctx, const device& dev) noexcept
      : base_t{AllocKind, Alignment, ctx, dev} {}

  /** @brief Constructs an allocator that will allocate memory
   *        on the device associated with the queue
   * @param q Queue containing the allocation context and device
   */
  usm_allocator(const queue& q) noexcept : base_t{AllocKind, Alignment, q} {}

  /** @brief Performs a USM allocation
   * @param count Number of elements to allocate
   * @return A pointer to the memory block allocated by the function
   * @throw std::bad_alloc if allocation fails
   */
  pointer allocate(size_t count) {
    return static_cast<pointer>(base_t::allocate(count * sizeof(T)));
  }

  /** @brief Frees USM allocated memory
   * @param ptr Pointer to USM allocated memory.
   *        Passing other kinds of pointers is not guaranteed to work.
   * @param count Number of allocated elements
   */
  void deallocate(pointer ptr, size_t count) {
    return base_t::deallocate(ptr, count * sizeof(T));
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind != usm::alloc::device))>
  void construct(pointer /*ptr*/, const_reference /*value*/) {
    this->throw_not_implemented("construct");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind == usm::alloc::device))>
  void construct(pointer, const_reference) {
    this->throw_not_supported_on_device("construct");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind != usm::alloc::device))>
  void destroy(pointer /*ptr*/) {
    this->throw_not_implemented("destroy");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind == usm::alloc::device))>
  void destroy(pointer) {
    this->throw_not_supported_on_device("destroy");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind != usm::alloc::device))>
  pointer address(reference /*value*/) const {
    this->throw_not_implemented("address");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind == usm::alloc::device))>
  pointer address(reference) const {
    this->throw_not_supported_on_device("address");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind != usm::alloc::device))>
  const_pointer address(const_reference /*value*/) const {
    this->throw_not_implemented("address");
  }

  template <COMPUTECPP_ENABLE_IF_VAL(AllocKind,
                                     (AllocKind == usm::alloc::device))>
  const_pointer address(const_reference) const {
    this->throw_not_supported_on_device("address");
  }
};

/** @brief Compares two USM allocators
 * @param lhs
 * @param rhs
 * @return True if allocators have the same template parameters
 *         and the same internal state
 */
template <class T1, usm::alloc AllocKindT1, size_t AlignmentT1, class T2,
          usm::alloc AllocKindT2, size_t AlignmentT2>
bool operator==(
    const usm_allocator<T1, AllocKindT1, AlignmentT1>& lhs,
    const usm_allocator<T2, AllocKindT2, AlignmentT2>& rhs) noexcept {
  return (AllocKindT1 == AllocKindT2) && (AlignmentT1 == AlignmentT2) &&
         lhs.core_equals(rhs);
}

/** @brief Compares two USM allocators
 * @param lhs
 * @param rhs
 * @return False if allocators have the same template parameters
 *         and the same internal state
 */
template <class T1, usm::alloc AllocKindT1, size_t AlignmentT1, class T2,
          usm::alloc AllocKindT2, size_t AlignmentT2>
bool operator!=(
    const usm_allocator<T1, AllocKindT1, AlignmentT1>& lhs,
    const usm_allocator<T2, AllocKindT2, AlignmentT2>& rhs) noexcept {
  return !(lhs == rhs);
}

}  // namespace experimental

// Expose USM functions in the sycl namespace

#if SYCL_LANGUAGE_VERSION < 202000

using cl::sycl::experimental::aligned_alloc;
using cl::sycl::experimental::aligned_alloc_device;
using cl::sycl::experimental::aligned_alloc_host;
using cl::sycl::experimental::aligned_alloc_shared;
using cl::sycl::experimental::malloc;
using cl::sycl::experimental::malloc_device;
using cl::sycl::experimental::malloc_host;
using cl::sycl::experimental::malloc_shared;

using cl::sycl::experimental::free;
using cl::sycl::experimental::usm_allocator;

using cl::sycl::experimental::get_pointer_device;
using cl::sycl::experimental::get_pointer_type;

#endif  // SYCL_LANGUAGE_VERSION

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_H_
