////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2002-2019 Codeplay Software Limited
//  All Rights Reserved.
//
//  ComputeCpp : SYCL 1.2.1 Implementation
//
//  File: accessor_util.h
//
////////////////////////////////////////////////////////////////////////////////

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_ACCESSOR_UTIL_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_ACCESSOR_UTIL_H_

#include "SYCL/common.h"

namespace cl {
namespace sycl {

template <typename, int, typename>
class buffer;

namespace codeplay {

template <access::mode kMode, access::target kTarget, typename elemT, int kDims,
          typename AllocatorT>
accessor<elemT, kDims, kMode, kTarget, access::placeholder::true_t>
make_placeholder_accessor(buffer<elemT, kDims, AllocatorT>& bufferRef) {
  return accessor<elemT, kDims, kMode, kTarget, access::placeholder::true_t>(
      bufferRef);
}

//  make_placeholder_accessor

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_ACCESSOR_UTIL_H_
