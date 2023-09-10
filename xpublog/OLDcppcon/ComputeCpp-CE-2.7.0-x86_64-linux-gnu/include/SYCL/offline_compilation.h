////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2002-2018 Codeplay Software Limited
//  All Rights Reserved.
//
//  ComputeCpp : SYCL 1.2.1 Implementation
//
////////////////////////////////////////////////////////////////////////////////

#ifndef RUNTIME_INCLUDE_SYCL_DETAIL_OFFLINE_COMPILATION_H_
#define RUNTIME_INCLUDE_SYCL_DETAIL_OFFLINE_COMPILATION_H_
#endif

#include "predefines.h"

namespace cl {
namespace sycl {
namespace detail {

/** @brief This enum specifies the backend for offline compilation
 */
enum class offline_backend {
  no_offline,
  aorta_x86_64,
  custom,
  aorta_aarch64,
  aorta_rcar_cve,
};

/** @brief  This class is a function object wrapper(functor) and is used
           to query for offline compilation
 */
class offline_compilation_query {
 public:
  COMPUTECPP_TEST_VIRTUAL ~offline_compilation_query() = default;

  /** @brief Returns the offline compilation backend
   */
  inline offline_backend get_offline_backend() const {
    return get_offline_compilation_backend();
  }

 private:
  COMPUTECPP_TEST_VIRTUAL inline offline_backend
  get_offline_compilation_backend() const {
#ifdef COMPUTECPP_OFFLINE_COMPILATION
#ifdef COMPUTECPP_OFFLINE_TARGET_AORTA_X86_64
    return offline_backend::aorta_x86_64;
#elif COMPUTECPP_OFFLINE_TARGET_CUSTOM
    return offline_backend::custom;
#elif COMPUTECPP_OFFLINE_TARGET_AORTA_AARCH64
    return offline_backend::aorta_aarch64;
#elif COMPUTECPP_OFFLINE_TARGET_AORTA_RCAR_CVE
    return offline_backend::aorta_rcar_cve;
#else
// We need to throw when offline compilation is set without AORTA as target
#error Offline compilation is set without AORTA as a target
    return offline_backend::no_offline;
#endif
#else
    return offline_backend::no_offline;
#endif
  }
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl
