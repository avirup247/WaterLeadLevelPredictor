/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file buffer_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_BUFFER_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_BUFFER_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/accessor/accessor_ops.h"
#include "SYCL/atomic.h"
#include "SYCL/compat_2020.h"

namespace cl {
namespace sycl {

namespace detail {

/** Retrieves the access range based on the number of accessor dimensions
 *
 * For 0-dim accessors, the access range is a single point.
 * For other dimensions, use the provided range
 *
 * @tparam kDims Number of accessor dimensions
 * @tparam interface_dims Number of range dimensions
 * @param providedRange Range to use for non-zero-dim accessors
 * @return Access range for the accessor
 */
template <int kDims, int interface_dims>
detail::access_range get_access_range(
    const cl::sycl::range<interface_dims>& providedRange) {
  return {detail::index_array{0, 0, 0},
          (kDims == 0) ? range<interface_dims>{detail::index_array{1, 1, 1}}
                       : providedRange};
}

////////////////////////////////////////////////////////////////////////////////
// accessor_buffer_interface

template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor_buffer_interface
    : public detail::accessor_common<elemT, kDims, kMode, kTarget,
                                     isPlaceholder> {
 private:
  using base_t =
      detail::accessor_common<elemT, kDims, kMode, kTarget, isPlaceholder>;

 protected:
  using base_t::interface_dims;
  using base_t::is_read_only;

 public:
  /// Alias for the type returned by single subscript operators
  using single_subscript_op_t =
      detail::subscript_op<0, elemT, kDims, kMode, kTarget, isPlaceholder>;

#if SYCL_LANGUAGE_VERSION >= 202001
  using reference = typename base_t::reference;
  using iterator = std::conditional_t<is_read_only, const elemT*, elemT*>;
  using const_iterator = const elemT*;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = size_t;

 protected:
  using return_t = reference;
#else
 protected:
  using return_t = typename std::conditional<
      is_read_only, typename base_t::const_reference,
      typename device_arg<elemT, kDims, kMode, kTarget,
                          isPlaceholder>::raw_ref_type>::type;
#endif  // SYCL_LANGUAGE_VERSION

 protected:
#ifdef __SYCL_DEVICE_ONLY__
  using base_t::m_deviceArgs;
#endif

  static constexpr bool is_atomic_ctr = (kMode == access::mode::atomic);
  static constexpr bool is_n_dim = (0 < kDims) && (kDims < 4);

  /** The atomic address space is only valid when combined with an atomic
   *        accessor, but it needs to be defined for all other targets as well.
   */
  static constexpr auto atomic_address_space =
      detail::get_atomic_address_space<kTarget>::value;

  // Inherit constructors
  using base_t::base_t;

 public:
  /** Multiple-subscript syntax subscript operator
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param index The first index to specify which element to access.
   * @return An subscript_op object of dimensions 1 less than the accessor.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, !is_atomic_ctr && kDims > 1)>
  detail::subscript_op<(kDims - 1), elemT, kDims, kMode, kTarget, isPlaceholder>
  operator[](size_t index) const {
    detail::index_array idx(index, 0, 0);
    return {*this, idx};
  }

 private:
  /** Checks whether data is being accessed out-of-bounds
   * @param index Index being accessed
   * @return size_t Actual index.
   *         If the input index is valid, just return that,
   *         otherwise return 0.
   */
  inline size_t check_bounds(size_t index) const {
#ifdef COMPUTECPP_CHECK_OUT_OF_BOUNDS
    if (index >= detail::size(*this)) {
#ifdef COMPUTECPP_CHECK_VERBOSE
      printf("Out of bounds access with index %zu\n", index);
#endif  // COMPUTECPP_CHECK_VERBOSE
      auto ptr = reinterpret_cast<int*>(this->get_device_ptr());
      ptr[detail::error_code_begin(this->get_size())] = 0x5ca1ab1e;
      // Return a valid index instead
      return 0;
    }
#endif  // COMPUTECPP_CHECK_OUT_OF_BOUNDS
    return index;
  }

  /** Computes the linear index for this accessor based on the internal range
   * @tparam indexDims Number of index dimensions
   * @param index The index we're accessing
   * @return Single linearized index for the accessor
   */
  template <int indexDims>
  inline size_t get_linear_index(const id<indexDims>& index) const {
#ifndef __SYCL_DEVICE_ONLY__
    auto rng = range<interface_dims>(this->get_store_range());
#else
    auto rng = range<indexDims>{detail::index_array{m_deviceArgs.m_fullRange}};
#endif  // __SYCL_DEVICE_ONLY__

    return detail::construct_linear_row_major_index(index, rng);
  }

  /** Linearizes the accessor offset
   * @return Number of elements the accessor is offset
   *         from the beginning of the buffer
   */
  inline size_t linear_offset_from_buffer() const {
    const id<kDims> offset = this->get_offset();
    range<interface_dims> rng = this->get_store_range();
    return detail::construct_linear_row_major_index(offset, rng);
  }

 public:
  /** Multiple-subscript syntax subscript operator
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param index The index to specify which element to access.
   * @return A reference to the element specified by the index parameter.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1 && !is_atomic_ctr))>
  return_t operator[](size_t index) const {
#if SYCL_LANGUAGE_VERSION >= 202002
    const auto linearOffsetAdd = this->linear_offset_from_buffer();
#else
    constexpr size_t linearOffsetAdd = 0;
#endif  // SYCL_LANGUAGE_VERSION >= 202002
    index += linearOffsetAdd;
    index = this->check_bounds(index);
    return this->get_device_ptr()[index];
  }

  /** Conversion operator for 0-dimensional accessor
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @return A reference to the element specified by the index parameter.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 0 && !is_atomic_ctr))>
  operator return_t() const {
    return *this->get_device_ptr();
  }

  /** Multiple-subscript syntax subscript operator for 1-dimensional accessor
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param index The index to specify which element to accesss.
   * @return A reference to the element specified by the index parameter.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1 && is_atomic_ctr))>
  atomic<elemT, atomic_address_space> operator[](size_t index) const {
#if SYCL_LANGUAGE_VERSION >= 202002
    const auto linearOffsetAdd = this->linear_offset_from_buffer();
#else
    constexpr size_t linearOffsetAdd = 0;
#endif  // SYCL_LANGUAGE_VERSION >= 202002
    index += linearOffsetAdd;
    index = this->check_bounds(index);
    return cl::sycl::atomic<elemT, atomic_address_space>::make_from_device_ptr(
        this->get_device_ptr() + index);
  }

  /** Implicit conversion from a zero dimensional atomic accessor
   *  to an atomic type
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for atomic 0-dim buffer accessors
   * @return Atomic object encapsulating the accessed data
   */
  template <COMPUTECPP_ENABLE_IF(elemT, ((kDims == 0) && is_atomic_ctr))>
  operator atomic<elemT>() const {
    return atomic<elemT>::make_from_device_ptr(this->get_device_ptr());
  }

  /** Subscript operator using an id for non-atomic accessors
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param index The index to specify which element to access.
   * @return A reference to the element specified by the index parameter.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, !is_atomic_ctr && is_n_dim)>
  return_t operator[](id<kDims> index) const {
    size_t idx = this->get_linear_index(index);
#if SYCL_LANGUAGE_VERSION >= 202002
    const auto linearOffsetAdd = this->linear_offset_from_buffer();
#else
    constexpr size_t linearOffsetAdd = 0;
#endif  // SYCL_LANGUAGE_VERSION >= 202002
    idx += linearOffsetAdd;
    idx = this->check_bounds(idx);
    return this->get_device_ptr()[idx];
  }

  /** Subscript operator using an id for atomic accessors
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param index The index to specify which element to access.
   * @return A reference to the element specified by the index parameter.
   */
  template <COMPUTECPP_ENABLE_IF(elemT, is_atomic_ctr&& is_n_dim)>
  atomic<elemT, atomic_address_space> operator[](id<kDims> index) const {
    size_t idx = this->get_linear_index(index);
#if SYCL_LANGUAGE_VERSION >= 202002
    const auto linearOffsetAdd = this->linear_offset_from_buffer();
#else
    constexpr size_t linearOffsetAdd = 0;
#endif  // SYCL_LANGUAGE_VERSION >= 202002
    idx += linearOffsetAdd;
    idx = this->check_bounds(idx);
    return atomic<elemT, atomic_address_space>::make_from_device_ptr(
        (this->get_pointer() + idx));
  }

  /** Returns the device argument, which can be either a
   * pointer with an address space or an OpenCL image type, this is deduced by
   * the device_arg struct.
   * @return The device argument deduced by the device_arg struct.
   */
  typename device_arg<elemT, kDims, kMode, kTarget,
                      isPlaceholder>::ptr_class_type
  get_pointer() const {
    return typename device_arg<elemT, kDims, kMode, kTarget, isPlaceholder>::
        ptr_class_type(this->get_device_ptr());
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  using base_t::byte_size;
  using base_t::size;

  /** @brief Returns true iff size() == 0
   */
  bool empty() const noexcept { return size() == 0; }

  /** @brief Returns the maximum number of elements any accessor of this type
   * would be able to access
   */
  size_type max_size() const noexcept {
    return std::numeric_limits<difference_type>::max();
  }

  /** @brief Returns a pointer to the memory this accessor is accessing
   */
  iterator data() const noexcept {
    auto idx = this->linear_offset_from_buffer();
    idx = this->check_bounds(idx);
    return this->get_device_ptr() + idx;
  }

  /** @brief Returns an iterator to the first element of the memory within the
   * access range
   */
  iterator begin() const noexcept { return data(); }

  /** @brief Returns an iterator to the first element past the last element of
   * the memory within the access range
   */
  iterator end() const noexcept { return data() + size(); }

  /** @brief Returns a const iterator to the first element of the memory within
   * the access range
   */
  const_iterator cbegin() const noexcept { return begin(); }

  /** @brief Returns a const iterator that points past the last element of the
   * memory within the access range
   */
  const_iterator cend() const noexcept { return end(); }

#endif  // SYCL_LANGUAGE_VERSION

  /** Returns the plane ID
   * @return 0 for the host, the corresponding planeId for the device
   */
#if defined(__SYCL_DEVICE_ONLY__)
  std::int8_t get_device_plane_id() const { return m_deviceArgs.m_planeId; }
#else
  std::int8_t get_device_plane_id() const { return 0; }
#endif  // __SYCL_DEVICE_ONLY__
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// accessor

/** Non-specialized accessor template that covers only
 *  global and constant buffers.
 *
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions
 * @tparam kMode Access mode
 * @tparam kTarget Access target
 * @tparam isPlaceholder Whether the accessor is a placeholder
 */
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT COMPUTECPP_VALID_KERNEL_ARG_IF(
    kTarget != access::target::host_buffer,
    "Cannot pass host accessor to SYCL kernel") accessor
    : public detail::accessor_buffer_interface<elemT, kDims, kMode, kTarget,
                                               access::placeholder::false_t> {
 private:
  static_assert((kTarget == access::target::global_buffer) ||
                    (kTarget == access::target::constant_buffer),
                "Default case covers only global and constant buffers");

  using base_t =
      detail::accessor_buffer_interface<elemT, kDims, kMode, kTarget,
                                        access::placeholder::false_t>;

 protected:
  // Trying to inherit these members breaks GCC 7
  // interface_dims cannot be inherited at all
  // because it's protected status breaks deduction guides
  static constexpr bool is_atomic_ctr = (kMode == access::mode::atomic);
  static constexpr bool is_n_dim = (0 < kDims) && (kDims < 4);

  static constexpr bool is_global_buffer_ctr =
      (kTarget == access::target::global_buffer);
  static constexpr bool is_const_buffer_ctr =
      (kTarget == access::target::constant_buffer);
  static constexpr bool is_global_or_const_atom_ctr =
      (is_global_buffer_ctr || (is_const_buffer_ctr && !is_atomic_ctr));

 public:
  /** Constructs a buffer accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for global and constant
   *         buffer access, atomic mode not allowed
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <typename AllocatorT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr)>
  accessor(buffer<elemT, detail::acc_interface_dims<kDims>::value, AllocatorT>&
               bufferRef,
           handler& commandHandler, const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /**
   * Constructs an accessor of access target
   * access::target::global_buffer or access::target::constant_buffer by taking
   * a buffer object, a handler, an offset and a range and initialises the
   * base_accessor with the buffer, the handler, the access target, the access
   * mode and the element size. This constructor is for constructing a sub
   * accessor.
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @tparam COMPUTECPP_ENABLE_IF Only allows the constructor if the accessor is
   * global, or
   * is constant and not atomic.
   * @param bufferRef Reference to the buffer object the accessor is to access
   * data from.
   * @param commandHandler Reference to the handler object for the command group
   * @param accessOffset The offset that the sub accessor should have access
   * from.
   * @param accessRange The range that the sub accessor should have access to.
   * scope that the accessor is being constructed within.
   * @deprecated Need to reverse the order of the access offset and range,
   *           see 4.7.6.6 Buffer accessor interface in SYCL 1.2.1
   */
  template <typename AllocatorT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           id<kDims> accessOffset, range<kDims> accessRange)
      : base_t{bufferRef, commandHandler,
               detail::access_range(accessOffset, accessRange)} {}

  /** Constructs a ranged buffer accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only allowed for global and constant buffer
   *         accessors which are not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  template <typename AllocatorT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, const property_list& propList = {})
      : accessor{bufferRef, commandHandler, accessRange, id<kDims>{},
                 propList} {}

  /** Constructs a ranged buffer accessor with an offset
   * @tparam AllocatorT Type of the buffer objects allocator
   * @tparam COMPUTECPP_ENABLE_IF Only allows the constructor if the accessor is
   *         global, or is constant and not atomic, and is not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param propList Additional properties
   */
  template <typename AllocatorT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, id<kDims> accessOffset,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::access_range(accessOffset, accessRange)} {
    (void)propList;
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Constructs a buffer accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @tparam TagT The type of the CTAD tag parameter
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for global and constant
   *         buffer access, atomic mode not allowed
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param tag CTAD tag parameter, used in deduction guides
   * @param propList Additional properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr)>
  accessor(buffer<elemT, detail::acc_interface_dims<kDims>::value, AllocatorT>&
               bufferRef,
           handler& commandHandler, TagT /*tag*/,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a ranged buffer accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT The type of the CTAD tag parameter
   * @tparam COMPUTECPP_ENABLE_IF Only allowed for global and constant buffer
   *         accessors which are not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param tag CTAD tag parameter, used in deduction guides
   * @param propList Additional properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, TagT /*tag*/,
           const property_list& propList = {})
      : accessor{bufferRef, commandHandler, accessRange, id<kDims> {}} {
    (void)propList;
  }

  /** Constructs a ranged buffer accessor with an offset
   * @tparam AllocatorT Type of the buffer objects allocator
   * @tparam TagT The type of the CTAD tag parameter
   * @tparam COMPUTECPP_ENABLE_IF Only allows the constructor if the accessor is
   *         global, or is constant and not atomic, and is not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param tag CTAD tag parameter, used in deduction guides
   * @param propList Additional properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, id<kDims> accessOffset, TagT /*tag*/,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::access_range(accessOffset, accessRange)} {
    (void)propList;
  }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

 protected:
  friend class accessor<elemT, kDims, kMode, kTarget,
                        access::placeholder::true_t>;

  /** Constructs an accessor from a \ref{storage_mem} object.
   *  Used to create normal accessors from placeholder ones.
   * @internal
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for non-local accessors
   *         which are not 0-dimensional
   * @param store Storage object representing the buffer/image
   * @param commandHandler Command group handler
   * @param accessRange Data access range
   * @param propList Additional properties
   */
  template <COMPUTECPP_ENABLE_IF(elemT, is_global_or_const_atom_ctr&& is_n_dim)>
  accessor(storage_mem&& store, handler& commandHandler,
           detail::access_range accessRange, const property_list& propList = {})
      : base_t{store, commandHandler, accessRange} {
    (void)propList;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

////////////////////////////////////////////////////////////////////////////////
// placeholder accessor

/**
 * @brief A public facing accessor that can be constructed outside of a command
 *        group.
 *
 *        Even though it can be constructed, it cannot be accessed outside of a
 *        command group. Before it can be accessed, it has to be registered in a
 *        command group handler.
 *
 *        This is an extension of the SYCL specification.
 *
 *        The reason this is a specialization instead of using enable_if in one
 *        class is that COMPUTECPP_CONVERT_ATTR has to be placed on the regular
 *        accessor class and COMPUTECPP_CONVERT_ATTR_PLACEHOLDER on the
 *        placeholder one.
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions
 * @tparam kMode Access mode
 * @tparam kTarget Access target
 */
template <typename elemT, int kDims, access::mode kMode, access::target kTarget>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT COMPUTECPP_VALID_KERNEL_ARG_IF(
    kTarget != access::target::host_buffer,
    "Cannot pass host accessor to SYCL kernel")
    accessor<elemT, kDims, kMode, kTarget, access::placeholder::true_t>
    : public detail::accessor_buffer_interface<elemT, kDims, kMode, kTarget,
                                               access::placeholder::true_t> {
 private:
  static_assert((kTarget == access::target::global_buffer) ||
                    (kTarget == access::target::constant_buffer),
                "Default case covers only global and constant buffers");

  using base_t = detail::accessor_buffer_interface<elemT, kDims, kMode, kTarget,
                                                   access::placeholder::true_t>;

 protected:
  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @param bufferRef Reference to the buffer object the accessor is to access
   *        data from.
   * @param accessRange the offset and range that this accessor can access.
   */
  accessor(storage_mem& store, detail::access_range accessRange)
      : base_t{store, accessRange} {}
  /**
   * @brief returns an accessor similar to *this, with the offset changed to
   *        this->get_offset()[0] + addedOffset. A call to this function will
   * fail with ACCESSOR_ARGUMENTS_ERROR if the created accessor has an offset
   *        less than 0 or if the new offset plus the current access range
   * exceed this->get_range()[0].
   */
  accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t>
  get_accessor_with_added_offset(int addedOffset) const {
    detail::access_range range = this->get_access_range();
    // ensure that negative values move the offset backwards
    intptr_t result = static_cast<intptr_t>(addedOffset) +
                      static_cast<intptr_t>(range.offset[0]);
    // check underflow
    if (result < 0) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          CL_SUCCESS, detail::cpp_error_code::ACCESSOR_ARGUMENTS_ERROR, nullptr,
          "Attempted arithmetic operation out of accessor bounds. Underflow");
    }
    // check overflow
    size_t resultSizeT = static_cast<size_t>(result);
    if (resultSizeT + range.range[0] > this->get_store_range()[0]) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          CL_SUCCESS, detail::cpp_error_code::ACCESSOR_ARGUMENTS_ERROR, nullptr,
          "Attempted arithmetic operation out of accessor bounds. Overflow");
    }
    // set the result
    range.offset[0] = resultSizeT;

    cl::sycl::storage_mem storage(this->get_store());
    return accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t>(
        storage, range);
  }

 public:
  /** Constructs a default placeholder accessor without associated storage
   * @param propList Additional properties
   */
  accessor(const property_list& propList = {}) : base_t{} { (void)propList; }

  /** Constructs a placeholder accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @param bufferRef Reference to the buffer object the accessor is to access
   *        data from.
   * @param propList Additional properties
   */
  template <typename AllocatorT = default_allocator>
  explicit accessor(buffer<elemT, detail::acc_interface_dims<kDims>::value,
                           AllocatorT>& bufferRef,
                    const property_list& propList = {})
      : base_t{bufferRef,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @param bufferRef Buffer object where access is being requested
   * @param accessOffset Point where data access starts
   * @param accessRange Range of data to access
   * @deprecated Need to reverse the order of the access offset and range,
   *             see 4.7.6.6 Buffer accessor interface in SYCL 1.2.1
   */
  template <typename AllocatorT>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, id<kDims> accessOffset,
           range<kDims> accessRange)
      : base_t{bufferRef, detail::access_range(accessOffset, accessRange)} {}

  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
           range<kDims> accessRange, const property_list& propList = {})
      : accessor{bufferRef, accessRange, id<kDims>{}, propList} {}

  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param accessOffset Point where data access starts
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
           range<kDims> accessRange, id<kDims> accessOffset,
           const property_list& propList = {})
      : base_t{bufferRef, detail::access_range(accessOffset, accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged placeholder accessor
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  accessor(buffer<elemT, kDims>& bufferRef, range<kDims> accessRange,
           const property_list& propList = {})
      : accessor{bufferRef, accessRange, id<kDims>{}, propList} {}

  /** Constructs a ranged placeholder accessor
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param accessOffset Point where data access starts
   * @param propList Additional properties
   */
  accessor(buffer<elemT, kDims>& bufferRef, range<kDims> accessRange,
           id<kDims> accessOffset, const property_list& propList = {})
      : base_t{bufferRef, detail::access_range(accessOffset, accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, const property_list& propList = {})
      : accessor{bufferRef, commandHandler, accessRange, id<kDims>{},
                 propList} {}

  /** Constructs a ranged placeholder accessor
   * @tparam AllocatorT Specifies the type of the buffer objects allocator.
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Reference to the handler object for the command group
   * scope that the accessor is being constructed within.
   * @param accessRange Range of data to access
   * @param accessOffset Point where data access starts
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, id<kDims> accessOffset,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a ranged placeholder accessor
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  accessor(buffer<elemT, kDims>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, const property_list& propList = {})
      : accessor{bufferRef, commandHandler, accessRange, id<kDims>{},
                 propList} {}

  /** Constructs a ranged placeholder accessor
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Reference to the handler object for the command group
   * scope that the accessor is being constructed within.
   * @param accessRange Range of data to access
   * @param accessOffset Point where data access starts
   * @param propList Additional properties
   */
  accessor(buffer<elemT, kDims>& bufferRef, handler& commandHandler,
           range<kDims> accessRange, id<kDims> accessOffset,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler,
               detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a placeholder buffer accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& commandHandler,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler} {
    (void)propList;
  }

  /** Constructs a placeholder buffer accessor
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  accessor(buffer<elemT, kDims>& bufferRef, handler& commandHandler,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler} {
    (void)propList;
  }

  /** Constructs a placeholder 0-dimensional buffer accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for 0 dimensions
   * @param bufferRef Buffer object where access is being requested
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF(elemT, (kDims == 0))>
  accessor(buffer<elemT, 1, AllocatorT>& bufferRef, handler& commandHandler,
           const property_list& propList = {})
      : base_t{bufferRef, commandHandler} {
    (void)propList;
  }

  /** @brief Obtains a normal accessor from the placeholder accessor
   * @param commandHandler Command group handler where the accessor will be used
   * @return Normal accessor that does not need to be registered
   */
  accessor<elemT, kDims, kMode, kTarget, access::placeholder::false_t>
  get_access(handler& commandHandler) const {
    return accessor<elemT, kDims, kMode, kTarget, access::placeholder::false_t>(
        cl::sycl::storage_mem(this->get_store()), commandHandler,
        this->get_access_range());
  }

  /** @brief Creates and returns a new accessor with its offset changed by rhs
   * @param rhs the offset to add to the current accessor's offset in the new
   * accessor
   * @return A new accessor with its get_offset()[0] equal to
   * this->get_offset()[0] + rhs
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1))>
  accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t> operator+(
      int rhs) const {
    return get_accessor_with_added_offset(rhs);
  }

  /** @brief Creates and returns a new accessor with its offset changed by rhs
   * @param rhs the offset to subtract to the current accessor's offset in the
   * new accessor
   * @return A new accessor with its get_offset()[0] equal to
   * this->get_offset()[0] - rhs
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1))>
  accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t> operator-(
      int rhs) const {
    return get_accessor_with_added_offset(-rhs);
  }

  /** @brief Changes this->offset by rhs as this->get_offset()[0] + rhs
   * @param rhs the offset to add to the current accessor's offset
   * @return This accessor
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1))>
  accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t>& operator+=(
      int rhs) {
    *this = get_accessor_with_added_offset(rhs);
    return *this;
  }

  /** @brief Changes this->offset by rhs as this->get_offset()[0] - rhs
   * @param rhs the offset to subtract to the current accessor's offset
   * @return This accessor
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 1))>
  accessor<elemT, 1, kMode, kTarget, access::placeholder::true_t>& operator-=(
      int rhs) {
    *this = get_accessor_with_added_offset(-rhs);
    return *this;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR_PLACEHOLDER;

#if SYCL_LANGUAGE_VERSION >= 202001
#ifndef COMPUTECPP_DISABLE_ACC_DEDUCTION_GUIDES

// Buffer accessor deduction guides

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_only))
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(write_only))
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_write))
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, const property_list&)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_only),
         const property_list&)
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(write_only),
         const property_list&)
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_write),
         const property_list&)
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(read_only))
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(write_only))
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(read_write))
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         const property_list&)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(read_only), const property_list&)
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(write_only), const property_list&)
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
         decltype(read_write), const property_list&)
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(read_only))
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(write_only))
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(read_write))
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         const property_list&)
    ->accessor<elemT, kDims, detail::default_access_mode_v<elemT>,
               access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(read_only), const property_list&)
    ->accessor<elemT, kDims, access_mode::read, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(write_only), const property_list&)
    ->accessor<elemT, kDims, access_mode::write, access::target::global_buffer>;

template <typename elemT, int kDims, typename AllocatorT>
accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>, id<kDims>,
         decltype(read_write), const property_list&)
    ->accessor<elemT, kDims, access_mode::read_write,
               access::target::global_buffer>;

#endif  // COMPUTECPP_DISABLE_ACC_DEDUCTION_GUIDES
#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_BUFFER_ACCESSOR_H_
