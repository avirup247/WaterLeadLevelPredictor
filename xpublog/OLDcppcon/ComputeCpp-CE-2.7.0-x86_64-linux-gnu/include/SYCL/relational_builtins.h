/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

    * relational_builtins.h *

*******************************************************************************/
/**
  @file
  @brief This file defines some internal relational routines for vectors.
 */

#include "SYCL/common.h"

namespace cl {
namespace sycl {

/** @cond COMPUTECPP_DEV */

namespace detail {

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability scalar types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability scalar types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(T x) {
  return (x < 0) ? 1 : 0;
}

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(vec<T, 2> x) {
  int res = 0;
  res = (x.x() < 0) ? 1 : res;
  res = (x.y() < 0) ? 1 : res;
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(vec<T, 3> x) {
  int res = 0;
  res = (x.x() < 0) ? 1 : res;
  res = (x.y() < 0) ? 1 : res;
  res = (x.z() < 0) ? 1 : res;
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(vec<T, 4> x) {
  int res = 0;
  res = (x.x() < 0) ? 1 : res;
  res = (x.y() < 0) ? 1 : res;
  res = (x.z() < 0) ? 1 : res;
  res = (x.w() < 0) ? 1 : res;
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(vec<T, 8> x) {
  int res = 0;
  char4 xlow(x.s0(), x.s1(), x.s2(), x.s3());
  char4 xhigh(x.s4(), x.s5(), x.s6(), x.s7());
  res = any(xlow);
  res = any(xhigh) || res;
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if the MSB is enabled return 1 otherwise
/// return 0.
///
/// Implementation of SYCL 1.2 any().
template <typename T>
COMPUTECPP_EXPORT int any(vec<T, 16> x) {
  int res = 0;
  char8 xlow(x.s0(), x.s1(), x.s2(), x.s3(), x.s4(), x.s5(), x.s6(), x.s7());
  char8 xhigh(x.s8(), x.s9(), x.sA(), x.sB(), x.sC(), x.sD(), x.sE(), x.sF());
  res = any(xlow);
  res = any(xhigh) || res;
  return res;
}

/// @deprecated
/// @brief Test whether the MSB is set
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability scalar types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability scalar types
///
/// If the MSB of x is set it returns 1, otherwise it is 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(T x) {
  return (x < 0) ? 1 : 0;
}

/// @deprecated
/// @brief Test whether MSB is set for all components of the vector
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if MSB is set for all components of return 1
/// otherwise return 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(vec<T, 2> x) {
  int res = 0;
  res = (x.x() < 0);
  res = res && (x.y() < 0);
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set for all components of the vector
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if MSB is set for all components of return 1
/// otherwise return 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(vec<T, 3> x) {
  int res = 0;
  res = (x.x() < 0);
  res = res && (x.y() < 0);
  res = res && (x.z() < 0);
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set for all components of the vector
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if MSB is set for all components of return 1
/// otherwise return 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(vec<T, 4> x) {
  int res = 0;
  res = (x.x() < 0);
  res = res && (x.y() < 0);
  res = res && (x.z() < 0);
  res = res && (x.w() < 0);
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set for all components of the vector
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if MSB is set for all components of return 1
/// otherwise return 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(vec<T, 8> x) {
  int res = 0;
  char4 xlow(x.s0(), x.s1(), x.s2(), x.s3());
  char4 xhigh(x.s4(), x.s5(), x.s6(), x.s7());
  res = all(xlow);
  res = all(xhigh) && res;
  return res;
}

/// @deprecated
/// @brief Test whether MSB is set for all components of the vector
/// @param[in] x An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
/// @return      An integral type belonging to the OpenCL/SYCL
/// interoperability vector types
///
/// For each component of x, if MSB is set for all components of return 1
/// otherwise return 0.
///
/// Implementation of SYCL 1.2 all().
template <typename T>
COMPUTECPP_EXPORT int all(vec<T, 16> x) {
  int res = 0;
  char8 xlow(x.s0(), x.s1(), x.s2(), x.s3(), x.s4(), x.s5(), x.s6(), x.s7());
  char8 xhigh(x.s8(), x.s9(), x.sA(), x.sB(), x.sC(), x.sD(), x.sE(), x.sF());
  res = all(xlow);
  res = all(xhigh) && res;
  return res;
}

}  // namespace detail

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl
