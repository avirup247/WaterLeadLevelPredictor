/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file local_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_LOCAL_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_LOCAL_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/accessor/buffer_accessor.h"

namespace cl {
namespace sycl {

/** Specialization for a local accessor
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions (> 0)
 * @tparam kMode Access mode
 */
template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT accessor<
    elemT, kDims, kMode, access::target::local, access::placeholder::false_t>
    : public detail::accessor_buffer_interface<elemT, kDims, kMode,
                                               access::target::local,
                                               access::placeholder::false_t> {
 private:
  static_assert((kMode == access::mode::read_write ||
                 kMode == access::mode::atomic),
                "access::target::local is only compatible with "
                "access::mode::read_write.");

  using base_t = detail::accessor_buffer_interface<
      elemT, kDims, kMode, access::target::local, access::placeholder::false_t>;

 protected:
  using base_t::interface_dims;

 public:
  /** Constructs a local buffer accessor
   * @param numElements Data range
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  accessor(range<interface_dims> numElements, handler& commandHandler,
           const property_list& propList = {})
      : base_t{kDims, numElements, commandHandler} {
    (void)propList;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

/** Specialization for a 0-dimensional local accessor
 * @tparam elemT Underlying data type
 * @tparam kMode Access mode
 */
template <typename elemT, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT accessor<
    elemT, 0, kMode, access::target::local, access::placeholder::false_t>
    : public detail::accessor_buffer_interface<elemT, 0, kMode,
                                               access::target::local,
                                               access::placeholder::false_t> {
 private:
  static_assert((kMode == access::mode::read_write ||
                 kMode == access::mode::atomic),
                "access::target::local is only compatible with "
                "access::mode::read_write.");

  static constexpr const int kDims = 0;

  using base_t = detail::accessor_buffer_interface<
      elemT, kDims, kMode, access::target::local, access::placeholder::false_t>;

 public:
  /** Constructs a 0-dimensional local buffer accessor
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  accessor(handler& commandHandler, const property_list& propList = {})
      : base_t{kDims, range<1>{1}, commandHandler} {
    (void)propList;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

/** Specialization for a subgroup_local accessor
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions
 * @tparam kMode Access mode
 */
template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT
    accessor<elemT, kDims, kMode, access::target::subgroup_local,
             access::placeholder::false_t>
    : public detail::accessor_buffer_interface<elemT, kDims, kMode,
                                               access::target::subgroup_local,
                                               access::placeholder::false_t> {
 private:
  static_assert((kMode == access::mode::read_write ||
                 kMode == access::mode::atomic),
                "access::target::subgroup_local is only compatible with "
                "access::mode::read_write.");

  using base_t =
      detail::accessor_buffer_interface<elemT, kDims, kMode,
                                        access::target::subgroup_local,
                                        access::placeholder::false_t>;

 protected:
  using base_t::interface_dims;
  using base_t::is_n_dim;

 public:
  /** Constructs a subgroup_local buffer accessor
   * @tparam COMPUTECPP_ENABLE_IF Only enabled when (kDims > 0)
   * @param numElements Data range
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <COMPUTECPP_ENABLE_IF(elemT, is_n_dim)>
  accessor(range<interface_dims> numElements, handler& commandHandler,
           const property_list& propList = {})
      : base_t{kDims, numElements, commandHandler} {
    (void)propList;
  }

  /** Constructs a 0-dimensional subgroup_local buffer accessor
   * @tparam COMPUTECPP_ENABLE_IF Only enabled when (kDims == 0)
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <COMPUTECPP_ENABLE_IF(elemT, (kDims == 0))>
  accessor(handler& commandHandler, const property_list& propList = {})
      : base_t{kDims, range<1>{1}, commandHandler} {
    (void)propList;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

#if SYCL_LANGUAGE_VERSION >= 202002

/** Local accessor type
 * @tparam dataT Underlying data type
 * @tparam dimensions Number of accessor dimensions (> 0)
 */
template <typename dataT, int dimensions = 1>
class local_accessor
    : public accessor<dataT, dimensions, access::mode::read_write,
                      access::target::local, access::placeholder::false_t> {
 private:
  using base_t = accessor<dataT, dimensions, access::mode::read_write,
                          access::target::local, access::placeholder::false_t>;

 public:
  using base_t::base_t;
};

namespace detail {

template <typename dataT, int dimensions>
struct opencl_backend_traits<sycl::local_accessor<dataT, dimensions>>
    : opencl_backend_traits<
          sycl::accessor<dataT, dimensions, access::mode::read_write,
                         access::target::local, access::placeholder::false_t>> {
};

}  // namespace detail

#endif  // SYCL_LANGUAGE_VERSION >= 202002

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_LOCAL_ACCESSOR_H_
