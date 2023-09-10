////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2002-2021 Codeplay Software Limited
//  All Rights Reserved.
//
//  ComputeCpp : SYCL 1.2.1 Implementation
//
////////////////////////////////////////////////////////////////////////////////

/** @file compat_2020.h
 *
 * @brief This file contains functions to help transition from
 * sycl 1.2.1 to sycl 2020
 */

#ifndef RUNTIME_INCLUDE_SYCL_COMPAT_2020_H_
#define RUNTIME_INCLUDE_SYCL_COMPAT_2020_H_

#include "SYCL/predefines.h"

namespace cl {
namespace sycl {
namespace detail {
/**
 * When the active SYCL version is 2020 it will call the byte_size() member
 * function, otherwise get_size() will be called.
 */
template <typename T>
constexpr size_t byte_size(const T& x) {
#if SYCL_LANGUAGE_VERSION >= 202001
  return x.byte_size();
#else
  return x.get_size();
#endif  // SYCL_LANGUAGE_VERSION >= 202001
}

/**
 * When the active SYCL version is 2020, it will call the size() member
 * function, otherwise get_count() will be called.
 */
template <typename T>
constexpr size_t size(const T& x) {
#if SYCL_LANGUAGE_VERSION >= 202001
  return x.size();
#else
  return x.get_count();
#endif  // SYCL_LANGUAGE_VERSION >= 202001
}
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_COMPAT_2020_H_
