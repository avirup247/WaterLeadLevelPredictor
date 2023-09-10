/**************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

****************************************************************/

/**
 * @file sycl_language_version.h
 * @brief This file contains the sycl_language_version class which allows the
 * runtime to query the SYCL language version specified by an application.
 */

#ifndef RUNTIME_INCLUDE_SYCL_LANGUAGE_VERSION_H_
#define RUNTIME_INCLUDE_SYCL_LANGUAGE_VERSION_H_

#include "computecpp_export.h"

#include "SYCL/predefines.h"

namespace cl {
namespace sycl {
namespace detail {

/** @brief Container of a variable which stores the SYCL language version so
 * that it can be specified by an application including SYCL.hpp to the runtime.
 */
class sycl_language_version {
 public:
  COMPUTECPP_EXPORT static int value;
};

/** @brief Class which simply initializes @ref sycl_language_version to
 * SYCL_LANGUAGE_VERSION on construction.
 */
class init_sycl_language_version {
 public:
  init_sycl_language_version() {
    sycl_language_version::value = SYCL_LANGUAGE_VERSION;
  }
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif /* RUNTIME_INCLUDE_SYCL_LANGUAGE_VERSION_H_ */
