/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file host_image_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_IMAGE_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_IMAGE_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/accessor/image_accessor.h"

namespace cl {
namespace sycl {

/** Specialization for a host image accessor
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions (> 0)
 * @tparam kMode Access mode
 */
template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT
    accessor<elemT, kDims, kMode, access::target::host_image,
             access::placeholder::false_t>
    : public detail::accessor_image_interface<elemT, kDims, kMode,
                                              access::target::host_image> {
 private:
  using base_t = detail::accessor_image_interface<elemT, kDims, kMode,
                                                  access::target::host_image>;

 public:
  /** Constructs a host image accessor
   * @tparam AllocatorT Type of the image allocator
   * @param imageRef Image object where access is being requested
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  explicit accessor(image<kDims, AllocatorT>& imageRef,
                    const property_list& propList = {})
      : base_t{imageRef} {
    (void)propList;
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_HOST_IMAGE_ACCESSOR_H_
