/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_BUILTINS_EXTENDED_H_
#define RUNTIME_INCLUDE_SYCL_BUILTINS_EXTENDED_H_

#include "SYCL/builtins/math_symbols.h"
#include "SYCL/device_event.h"
#include "SYCL/type_traits.h"

#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DETAIL_NAMESPACE abacus::detail::common
#else
#include <cstring>
#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {
namespace detail {

/** Performs async_work_group_copy with a stride of 1
 * @tparam DestT Destination type, some sort of multi_ptr
 * @tparam SrcT Source type, some sort of multi_ptr
 * @param dest Destination pointer
 * @param src Source pointer
 * @param numElements Number of elements to copy
 * @param isZeroId Whether the current work-item ID is all zeros
 * @return Device event associated with the asynchronous copy
 */
template <typename DestT, typename SrcT>
device_event async_work_group_copy_non_strided(DestT dest, SrcT src,
                                               size_t numElements,
                                               bool isZeroId) {
#ifdef __SYCL_DEVICE_ONLY__
  (void)isZeroId;
  __sycl_event_t previousEvent = 0;
  return COMPUTECPP_BUILTIN_INVOKE_IMPL(
      async_work_group_copy, device_event, COMPUTECPP_CPP_TO_CL(dest),
      COMPUTECPP_CPP_TO_CL(src), numElements, previousEvent);
#else   // __SYCL_DEVICE_ONLY__
  if (isZeroId) {
    const auto copySize =
        sizeof(detail::remove_pointer_t<typename SrcT::ptr_t>) * numElements;
    std::memcpy(dest.get(), src.get(), copySize);
  }
  return {};
#endif  // __SYCL_DEVICE_ONLY__
}

/** Performs async_work_group_copy with a specified stride on the source pointer
 * @tparam DestT Destination type, some sort of multi_ptr
 * @tparam SrcT Source type, some sort of multi_ptr
 * @param dest Destination pointer
 * @param src Source pointer
 * @param numElements Number of elements to copy
 * @param srcStride Stride used for the source pointer
 * @param isZeroId Whether the current work-item ID is all zeros
 * @return Device event associated with the asynchronous copy
 */
template <typename DestT, typename SrcT>
device_event async_work_group_copy_src_strided(DestT dest, SrcT src,
                                               size_t numElements,
                                               size_t srcStride,
                                               bool isZeroId) {
#ifdef __SYCL_DEVICE_ONLY__
  (void)isZeroId;
  __sycl_event_t previousEvent = 0;
  return COMPUTECPP_BUILTIN_INVOKE_IMPL(
      async_work_group_strided_copy, device_event, COMPUTECPP_CPP_TO_CL(dest),
      COMPUTECPP_CPP_TO_CL(src), numElements, srcStride, previousEvent);
#else   // __SYCL_DEVICE_ONLY__
  if (isZeroId) {
    size_t srcIter = 0;
    for (size_t i = 0; i < numElements; ++i) {
      *(dest + i) = *(src + srcIter);
      srcIter += srcStride;
    }
  }
  return {};
#endif  // __SYCL_DEVICE_ONLY__
}

/** Performs async_work_group_copy with a specified stride
 *  on the destination pointer
 * @tparam DestT Destination type, some sort of multi_ptr
 * @tparam SrcT Source type, some sort of multi_ptr
 * @param dest Destination pointer
 * @param src Source pointer
 * @param numElements Number of elements to copy
 * @param destStride Stride used for the destination pointer
 * @param isZeroId Whether the current work-item ID is all zeros
 * @return Device event associated with the asynchronous copy
 */
template <typename DestT, typename SrcT>
device_event async_work_group_copy_dest_strided(DestT dest, SrcT src,
                                                size_t numElements,
                                                size_t destStride,
                                                bool isZeroId) {
#ifdef __SYCL_DEVICE_ONLY__
  (void)isZeroId;
  __sycl_event_t previousEvent = 0;
  return COMPUTECPP_BUILTIN_INVOKE_IMPL(
      async_work_group_strided_copy, device_event, COMPUTECPP_CPP_TO_CL(dest),
      COMPUTECPP_CPP_TO_CL(src), numElements, destStride, previousEvent);
#else   // __SYCL_DEVICE_ONLY__
  if (isZeroId) {
    size_t dstIter = 0;
    for (size_t i = 0; i < numElements; ++i) {
      *(dest + dstIter) = *(src + i);
      dstIter += destStride;
    }
  }
  return {};
#endif  // __SYCL_DEVICE_ONLY__
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#ifndef __SYCL_DEVICE_ONLY__
#undef COMPUTECPP_DETAIL_NAMESPACE
#endif  // __SYCL_DEVICE_ONLY__

#endif  // RUNTIME_INCLUDE_SYCL_BUILTINS_EXTENDED_H_
