/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file allocator.h

  @brief Internal file, used for the definition of the standard allocators
*/
#ifndef RUNTIME_INCLUDE_SYCL_ALLOCATOR_H_
#define RUNTIME_INCLUDE_SYCL_ALLOCATOR_H_

#include "SYCL/common.h"
#include "SYCL/type_traits.h"

namespace cl {
namespace sycl {

/** @cond COMPUTECPP_DEV */
/** Implementation specific behaviour */
namespace detail {

/** @brief Forward calls from the runtime-side to the user-defined allocator.
 * This allows the internal runtime methods to call the user-defined typed
 * allocator without passing the template tag.
 * The isMapAllocator is used internally to determine if the allocator is
 * a map-based one, which will enable some optimizations if possible.
 */
class base_allocator {
 public:
  explicit base_allocator() = default;

  base_allocator(const base_allocator&) = default;
  base_allocator& operator=(const base_allocator&) = default;

  base_allocator(base_allocator&&) = default;
  base_allocator& operator=(base_allocator&&) = default;

  virtual ~base_allocator() = default;

  virtual void* allocate(size_t nElems) = 0;

  virtual void deallocate(void* p, size_t n) = 0;
};

/** @brief Typed derived class for a base allocator that holds an instance
 * to an user-defined allocator.
 */
template <typename AllocatorT>
class wrapped_allocator : public base_allocator {
 public:
  wrapped_allocator(AllocatorT allocator, size_t elemSizeMultiplier)
      : base_allocator(),
        m_a(allocator),
        m_elemSizeMultiplier(elemSizeMultiplier) {}

  void* allocate(size_t nElems) final {
    return m_a.allocate(nElems * m_elemSizeMultiplier);
  }

  void deallocate(void* p, size_t nElems) final {
    typename AllocatorT::pointer rp =
        reinterpret_cast<typename AllocatorT::pointer>(p);
    m_a.deallocate(rp, static_cast<typename AllocatorT::size_type>(
                           nElems * m_elemSizeMultiplier));
  }

  /** @brief Retrieves the original allocator
   * @return Allocator that was provided  on construction
   */
  AllocatorT get_allocator() const { return m_a; }

 private:
  AllocatorT m_a;

  size_t m_elemSizeMultiplier;
};

/** @brief Retrieves the original user-supplied allocator from the stored detail
 *        allocator object.
 *
 *        Relies on the fact that each allocator supplied by the user is wrapped
 *        into a wrapped_allocator
 * @tparam AllocatorT Type of the original allocator
 * @param base Pointer to the detail allocator object
 * @return The original allocator object
 */
template <class AllocatorT>
static AllocatorT cast_base_allocator(base_allocator* base) {
  return static_cast<wrapped_allocator<AllocatorT>*>(base)->get_allocator();
}

namespace aligned_mem {

/** @brief Retrieves the default alignment boundary
 * @return Alignment boundary in bytes
 */
COMPUTECPP_EXPORT size_t get_default_alignment();

/** @brief Retrieves the default required size multiplier
 * @return Required size multiplier in bytes
 */
COMPUTECPP_EXPORT size_t get_default_required_size_multiplier();

/** @brief Calculates the size that needs to be allocated for the memory to be
 *        properly aligned.
 * @param requestedSize Size in bytes the user requested. May allocate more
 *        according to the multiplier.
 * @param requiredSizeMultiplier The requirement for the allocation size to be
 *        a multiple of this (in bytes).
 * @return The size in bytes that needs to be allocated.
 */
COMPUTECPP_EXPORT size_t get_aligned_size(
    const size_t requestedSize, const size_t requiredSizeMultiplier =
                                    get_default_required_size_multiplier());

/** @brief Allocates aligned data.
 * @param requestedSize Size in bytes the user requested. May allocate more
 *        according to the multiplier.
 * @param alignment Memory alignment boundary in bytes.
 * @param requiredSizeMultiplier The requirement for the allocation size to be
 *        a multiple of this (in bytes).
 * @return Pointer to allocated aligned data
 */
COMPUTECPP_EXPORT void* allocate(const size_t requestedSize,
                                 const size_t alignment,
                                 const size_t requiredSizeMultiplier);

/** @brief Deallocates aligned data.
 * @param memptr Pointer to allocated aligned data
 * @param requestedSize Size in bytes the user requested. May deallocate more
 *        according to the multiplier.
 * @param requiredSizeMultiplier The requirement for the allocation size to be
 *        a multiple of this (in bytes).
 */
COMPUTECPP_EXPORT void deallocate(void* memptr, const size_t requestedSize,
                                  const size_t requiredSizeMultiplier);

/** @brief Checks whether the pointer points to aligned data
 * @param p Pointer to allocated data
 * @param totalSizeInBytes Total allocated size of the buffer in bytes
 * @param alignment Memory alignment boundary in bytes.
 * @param requiredSizeMultiplier The requirement for the allocation size to be
 *        a multiple of this (in bytes).
 */
COMPUTECPP_EXPORT bool is_aligned(void* p, const size_t totalSizeInBytes,
                                  const size_t alignment,
                                  const size_t requiredSizeMultiplier);

/** @brief An allocator that can allocate memory aligned to a certain alignment
 *        boundary with the final allocated size a multiple of some required
 *        size.
 */
class aligned_allocator : public base_allocator {
 protected:
  using allocator_t = std::allocator<byte>;

 public:
  using value_type = typename allocator_t::value_type;
  using pointer = typename allocator_t::pointer;
  using const_pointer = typename allocator_t::const_pointer;
  using reference = typename allocator_t::reference;
  using const_reference = typename allocator_t::const_reference;
  using size_type = typename allocator_t::size_type;
  using difference_type = typename allocator_t::difference_type;

 protected:
  size_t m_alignment;
  size_t m_requiredSizeMultiplier;

 public:
  /** @brief Constructs an allocator that can allocate memory aligned to a
   *        certain \ref{alignment} boundary with the final allocated size a
   *        multiple of \ref{requiredSizeMultiplier}.
   * @param alignment Memory alignment boundary in bytes.
   * @param requiredSizeMultiplier The requirement for the allocation size to
   *        be a multiple of this (in bytes).
   */
  aligned_allocator(
      size_t alignment = get_default_alignment(),
      size_t requiredSizeMultiplier = get_default_required_size_multiplier())
      : m_alignment(alignment),
        m_requiredSizeMultiplier(requiredSizeMultiplier) {}

  aligned_allocator(const aligned_allocator&) = default;
  aligned_allocator& operator=(const aligned_allocator&) = default;

  /**
    @brief Allocate \ref{numElems} elements of type T
    @param numElems How many elements to allocate
    @return T* pointer with new data
    */
  void* allocate(size_type sizeInBytes) final {
    return aligned_mem::allocate(sizeInBytes, m_alignment,
                                 m_requiredSizeMultiplier);
  }

  /**
    @brief Deallocates \ref{numElems} elements of type T from a pointer
    @param pointer T* pointer with previously allocated data
    @param numElems How many elements to deallocate
    */
  void deallocate(void* p, size_type sizeInBytes) final {
    aligned_mem::deallocate(p, sizeInBytes, m_requiredSizeMultiplier);
  }
};

}  // namespace aligned_mem

}  // namespace detail

/** @cond COMPUTECPP_DEV */

/** @brief Default SYCL allocator uses the aligned allocated, but also
 * removes the constness of the type.
 */
using default_allocator = detail::aligned_mem::aligned_allocator;

using buffer_allocator = default_allocator;

using image_allocator = default_allocator;

namespace detail {

/** @brief Creates a pointer to the internal base allocator structure from
 * an instance of an user-defined allocator class.
 * @param T Data type
 * @param AllocatorT Allocator type
 */
template <typename T, typename AllocatorT>
struct make_base_allocator {
 protected:
  /** @brief Creates a new base_allocator and wraps it into a unique pointer
   * @param allocator User-supplied allocator instance to be wrapped
   * @param elemSizeMultiplier When the allocation is made, the allocation size
   *        will be multiplied by this value
   * @return base_allocator wrapping the user supplied allocator
   */
  static unique_ptr_class<base_allocator> get(AllocatorT allocator,
                                              size_t elemSizeMultiplier) {
    return unique_ptr_class<wrapped_allocator<AllocatorT>>(
        new wrapped_allocator<AllocatorT>(allocator, elemSizeMultiplier));
  }

 public:
  /** @brief Creates a new base_allocator for use in a buffer object
   * @param allocator User-supplied allocator instance to be wrapped
   * @return base_allocator wrapping the user supplied allocator
   */
  static unique_ptr_class<base_allocator> get_buffer_allocator(
      AllocatorT allocator = AllocatorT()) {
    return get(allocator, get_elem_size_multiplier());
  }

  /** @brief Creates a new base_allocator for use in an image object
   * @param elemSizeMultiplier When the allocation is made, the allocation size
   *        will be multiplied by this value. Should represent the size of the
   *        image element calculated from the image channel type and order.
   * @param allocator User-supplied allocator instance to be wrapped
   * @return base_allocator wrapping the user supplied allocator
   */
  static unique_ptr_class<base_allocator> get_image_allocator(
      size_t elemSizeMultiplier, AllocatorT allocator = AllocatorT()) {
    return get(allocator, elemSizeMultiplier);
  }

  /** @brief Creates a new base_allocator for use in a local buffer object
   * @param elementSize When the allocation is made, the allocation size
   *        will be multiplied by this value. Should represent the size of the
   *        element the local buffer is storing.
   * @param allocator User-supplied allocator instance to be wrapped
   * @return base_allocator wrapping the user supplied allocator
   */
  static unique_ptr_class<base_allocator> get_local_allocator(
      size_t elementSize, AllocatorT allocator = AllocatorT()) {
    return get(allocator, elementSize * get_elem_size_multiplier());
  }

  /** @brief Calculates the required size multiplier
   *        depending on the type of the allocator.
   *
   *        This is required because the default allocator is not typed,
   *        so we need to multiply the allocation size
   *        with the size of the type.
   *        However, standard C++ allocators are typed, so the type size is
   *        already taken into account when allocation occurs.
   *
   * @return sizeof(T) for default allocator, 1 otherwise
   */
  static constexpr size_t get_elem_size_multiplier() {
    return (std::is_same<detail::decay_t<AllocatorT>,
                         detail::aligned_mem::aligned_allocator>::value)
               ? sizeof(T)
               : 1;
  }
};

/* clone_data.
 * @brief Copies a range of iterators into a temporary memory address
 * allocated using an user-defined allocator.
 * A deleter is set to call the user-define deallocate function.
 * @tparam T Type of the underlying data
 * @tparam AllocatorT Type of user supplied allocator
 * @param begin Start of the range
 * @param end End of the range
 * @param alloc Allocator object to be used
 * @return type-erased shared pointer
 */
template <typename T, typename AllocatorT, typename Iterator>
shared_ptr_class<void> clone_data(Iterator begin, Iterator end,
                                  AllocatorT alloc = AllocatorT()) {
  size_t size = std::distance(begin, end);
  auto typedAlloc = wrapped_allocator<AllocatorT>(
      alloc, make_base_allocator<T, AllocatorT>::get_elem_size_multiplier());
  shared_ptr_class<void> p(typedAlloc.allocate(size),
                           [typedAlloc, size](void* f) mutable {
                             typedAlloc.deallocate(static_cast<T*>(f), size);
                           });
  std::copy(begin, end, static_cast<T*>(p.get()));
  return p;
}

/* clone_data.
 * @brief Copies size elements from a const pointer in the user side into
 * a temporary memory address allocated using an user-defined allocator.
 * A deleter is set to call the user-define deallocate function.
 * @tparam T Type of the underlying data
 * @tparam AllocatorT Type of user supplied allocator
 * @param hostPointer Data to be cloned
 * @param size How much data to allocate
 * @param alloc Allocator object to be used
 * @return type-erased shared pointer
 */
template <typename T, typename AllocatorT>
shared_ptr_class<void> clone_data(const T* hostPointer, size_t size,
                                  AllocatorT alloc = AllocatorT()) {
  return clone_data<T>(hostPointer, hostPointer + size, alloc);
}

}  // namespace detail

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_ALLOCATOR_H_

////////////////////////////////////////////////////////////////////////////////
