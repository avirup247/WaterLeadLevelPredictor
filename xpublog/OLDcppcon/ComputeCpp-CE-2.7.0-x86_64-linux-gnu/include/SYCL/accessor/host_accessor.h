/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file host_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/accessor/buffer_accessor.h"
#include "SYCL/common.h"
#include "SYCL/property.h"

namespace cl {
namespace sycl {

template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT
    accessor<elemT, kDims, kMode, access::target::host_buffer,
             access::placeholder::false_t>
    : public detail::accessor_buffer_interface<elemT, kDims, kMode,
                                               access::target::host_buffer,
                                               access::placeholder::false_t> {
 private:
  using base_t =
      detail::accessor_buffer_interface<elemT, kDims, kMode,
                                        access::target::host_buffer,
                                        access::placeholder::false_t>;

 protected:
  using base_t::interface_dims;
  using base_t::is_n_dim;

 public:
  /** Constructs a host accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for host buffer accessors
   * @param bufferRef Buffer object where access is being requested
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(buffer<elemT, interface_dims, AllocatorT>& bufferRef,
           const property_list& propList = {})
      : base_t{bufferRef,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a ranged host accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for host buffer accessors
   *         which are not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param propList Additional properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF(elemT, is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
           range<kDims> accessRange, const property_list& propList = {})
      : accessor{bufferRef, accessRange, id<kDims>{}, propList} {}

  /** Constructs a ranged host accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for host buffer accessors
   *         which are not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param accessRange Range of data to access
   * @param accessOffset Point where data access starts
   * @param propList Additional properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF(elemT, is_n_dim)>
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
           range<kDims> accessRange, id<kDims> accessOffset,
           const property_list& propList = {})
      : base_t{bufferRef, detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a ranged host accessor
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for host buffer accessors
   *         which are not 0-dimensional
   * @param bufferRef Buffer object where access is being requested
   * @param accessOffset Point where data access starts
   * @param accessRange Range of data to access
   * @deprecated Need to reverse the order of the access offset and range,
   *             see 4.7.6.6 Buffer accessor interface in SYCL 1.2.1
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF(elemT, is_n_dim)>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, id<kDims> accessOffset,
           range<kDims> accessRange)
      : base_t{bufferRef, detail::access_range(accessOffset, accessRange)} {}

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

#if SYCL_LANGUAGE_VERSION >= 202001

/** Accessor class that provides host access, either immediate or delayed
 * @tparam elemT Underlying data type
 * @tparam kDims Number of dimensions
 * @tparam kMode Access mode
 */
template <typename elemT, int kDims, access_mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT host_accessor
    : public detail::accessor_buffer_interface<elemT, kDims, kMode,
                                               access::target::host_buffer,
                                               access::placeholder::false_t> {
  using base_t =
      detail::accessor_buffer_interface<elemT, kDims, kMode,
                                        access::target::host_buffer,
                                        access::placeholder::false_t>;

 public:
  /** Constructs a 0-dim host accessor to a buffer, immediate access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims == 0))>
  host_accessor(buffer<elemT, 1, AllocatorT>& bufferRef,
                const property_list& propList = {})
      : base_t{bufferRef,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a 0-dim host accessor to a buffer, delayed access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims == 0))>
  host_accessor(buffer<elemT, 1, AllocatorT>& bufferRef, handler& cgh,
                const property_list& propList = {})
      : base_t{bufferRef, cgh,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a host accessor to a buffer, immediate access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                const property_list& propList = {})
      : base_t{bufferRef,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a host accessor to a buffer, immediate access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  explicit host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                         TagT /*tag*/, const property_list& propList = {})
      : base_t{bufferRef,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a host accessor to a buffer, delayed access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                const property_list& propList = {})
      : base_t{bufferRef, cgh,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a host accessor to a buffer, delayed access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                TagT /*tag*/, const property_list& propList = {})
      : base_t{bufferRef, cgh,
               detail::get_access_range<kDims>(bufferRef.get_range())} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, immediate access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param accessRange Range to access
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                range<kDims> accessRange, const property_list& propList = {})
      : base_t{bufferRef, detail::get_access_range<kDims>(accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, immediate access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param accessRange Range to access
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                range<kDims> accessRange, TagT /*tag*/,
                const property_list& propList = {})
      : base_t{bufferRef, detail::get_access_range<kDims>(accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, immediate access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param accessRange Range to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                range<kDims> accessRange, id<kDims> accessOffset,
                const property_list& propList = {})
      : base_t{bufferRef, detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, immediate access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param accessRange Range to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef,
                range<kDims> accessRange, id<kDims> accessOffset, TagT /*tag*/,
                const property_list& propList = {})
      : base_t{bufferRef, detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, delayed access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param accessRange Range to access
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                range<kDims> accessRange, const property_list& propList = {})
      : base_t{bufferRef, cgh, detail::get_access_range(accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, delayed access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param accessRange Range to access
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                range<kDims> accessRange, TagT /*tag*/,
                const property_list& propList = {})
      : base_t{bufferRef, cgh, detail::get_access_range(accessRange)} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, delayed access
   * @tparam AllocatorT Type of the buffer allocator
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param accessRange Range to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param propList List of properties
   */
  template <typename AllocatorT, COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                range<kDims> accessRange, id<kDims> accessOffset,
                const property_list& propList = {})
      : base_t{bufferRef, cgh,
               detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }

  /** Constructs a ranged host accessor to a buffer, delayed access.
   *  Only valid for CTAD.
   * @tparam AllocatorT Type of the buffer allocator
   * @tparam TagT Type of the deduction tag
   * @param bufferRef Buffer to access
   * @param cgh Command group handler
   * @param accessRange Range to access
   * @param accessOffset Offset from the beginning of the buffer
   * @param tag The tag used to deduce the accessor type
   * @param propList List of properties
   */
  template <typename AllocatorT, typename TagT,
            COMPUTECPP_ENABLE_IF_VAL(kDims, (kDims > 0))>
  host_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef, handler& cgh,
                range<kDims> accessRange, id<kDims> accessOffset, TagT /*tag*/,
                const property_list& propList = {})
      : base_t{bufferRef, cgh,
               detail::access_range{accessOffset, accessRange}} {
    (void)propList;
  }
};

// Deduction guides for the host_accessor
template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>,
              decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>,
              decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>,
              decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>, id<kDims>)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>, id<kDims>,
              decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>, id<kDims>,
              decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, range<kDims>, id<kDims>,
              decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              id<kDims>)
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              id<kDims>, decltype(read_only))
    ->host_accessor<elemT, kDims, access_mode::read>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              id<kDims>, decltype(read_write))
    ->host_accessor<elemT, kDims, access_mode::read_write>;

template <typename elemT, int kDims, typename AllocatorT>
host_accessor(buffer<elemT, kDims, AllocatorT>&, handler&, range<kDims>,
              id<kDims>, decltype(write_only))
    ->host_accessor<elemT, kDims, access_mode::write>;

#endif  //  SYCL_LANGUAGE_VERSION

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_ACCESSOR_H_
