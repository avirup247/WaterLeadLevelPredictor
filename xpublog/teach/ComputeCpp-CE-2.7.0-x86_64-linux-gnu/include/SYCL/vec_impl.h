/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_impl.h

 @brief This file contains the implementation of the @ref cl::sycl::vec class
 definition as defined by the SYCL 1.2.1 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_IMPL_H_
#define RUNTIME_INCLUDE_SYCL_VEC_IMPL_H_

#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/common.h"
#include "SYCL/deduce.h"
#include "SYCL/host_compiler_macros.h"
#include "SYCL/vec_common.h"
#include "SYCL/vec_macros.h"
#include "SYCL/vec_mem_container_storage_impl.h"
#include "SYCL/vec_swizzles.h"

#include <cstring>

////////////////////////////////////////////////////////////////////////////////

#ifndef __SYCL_DEVICE_ONLY__

COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wold-style-cast")
COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wreserved-id-macro")
#include "abacus/abacus_config"
COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)

#endif  // __SYCL_DEVICE_ONLY__

namespace cl {
namespace sycl {

/* forward declaration of pointer classes */
template <typename dataT, access::address_space addressSpace>
class multi_ptr;
/* forward declaration of the accessor class */
template <typename dataT, int kDims, access::mode accessMode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;

namespace detail {

/** @brief Struct template which inherits from std::false_type if T is not a
 * specialisation of the cl::sycl::vec class template and inherits from
 * std::true_type if T is a specialisation of the cl::sycl::vec class template,
 * through a partial template specialisation.
 * @tparam T Type which is being queried for being a specialisation of the
 * cl::sycl::vec class template.
 */
template <typename T>
struct is_vec : std::false_type {};

/** @brief Partial template specialisation of @ref cl::sycl::detail::is_vec for
 * vecT and kDims where T is cl::sycl::vec<vecT, kDims>.
 * @tparam vecT Element type of the specialisation of the cl::sycl::vec class
 * template.
 * @tparam kElems Number of elements of the specialisation of the cl::sycl::vec
 * class template.
 */
template <typename vecT, int kElems>
struct is_vec<vec<vecT, kElems>> : std::true_type {};

template <typename srcVectT, typename destVecT>
struct is_valid_vec_as_conversion {
  static constexpr bool value =
      (is_vec<srcVectT>::value && is_vec<destVecT>::value &&
       sizeof(srcVectT) == sizeof(destVecT));
};

template <typename srcVectT, typename destVecT>
struct is_valid_vec_convert_conversion {
  static constexpr bool value =
      (is_vec<srcVectT>::value && is_vec<destVecT>::value &&
       srcVectT::width == destVecT::width);
};

/** Base class for mem_container classes, inherits from mem_container_storage.
 * @tparam dataT The data type.
 * @tparam kElems The number of elements.
 */
template <typename dataT, int kElems>
class mem_container_base : public mem_container_storage<dataT, kElems> {};

/** Recursive class used to define the functionality for different sized vec
  objects. The vec class will inherit from this class recursively adding
  functionality the different sizes of vec object.
* @tparam dataT The data type for the vector.
* @tparam kElems The number of elements for the vector.
* @tparam RecursiveSize The size at the current recursion.
*/
template <typename dataT, int kElems, int RecursiveSize>
class mem_container;

/** Specialization of mem_container for recursion size 1. Inherits from
  mem_container_base.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 1>
    : public mem_container_base<dataT, kElems> {
 public:
  /** Returns element 0 of the vector in the 16 element format.
   * @return Returns element 0 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s0)

  swizzled_vec<dataT, kElems, detail::x>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x>*>(this);
    return *swizzledVec;
  }
};

/** Specialization of mem_container for recursion size 2. Inherits from
  mem_container for recursion size 1.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 2> : public mem_container<dataT, kElems, 1> {
 public:
  /** Returns element 1 of the vector in the 16 element format.
   * @return Returns element 1 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s1)

  swizzled_vec<dataT, kElems, detail::y>& y() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y>& y() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES

  swizzled_vec<dataT, kElems, detail::x, detail::x>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y>& xy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y>& xy() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x>& yx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x>& yx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y>& yy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y>& yy() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

/** Specialization of mem_container for recursion size 3. Inherits from
  mem_container for recursion size 2.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 3> : public mem_container<dataT, kElems, 2> {
 public:
  /** Returns element 2 of the vector in the 16 element format.
   * @return Returns element 2 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s2)

  swizzled_vec<dataT, kElems, detail::z>& z() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z>& z() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z>*>(this);
    return *swizzledVec;
  }
#ifdef SYCL_SIMPLE_SWIZZLES

  swizzled_vec<dataT, kElems, detail::x, detail::z>& xz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z>& xz() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z>& yz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z>& yz() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x>& zx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x>& zx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y>& zy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y>& zy() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z>& zz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z>& zz() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y>& xxy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y>& xxy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z>& xxz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z>& xxz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x>& xyx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x>& xyx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y>& xyy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y>& xyy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z>& xyz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z>& xyz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x>& xzx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x>& xzx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y>& xzy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y>& xzy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z>& xzz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z>& xzz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x>& yxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x>& yxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y>& yxy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y>& yxy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z>& yxz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z>& yxz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x>& yyx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x>& yyx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y>& yyy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y>& yyy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z>& yyz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z>& yyz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x>& yzx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x>& yzx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y>& yzy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y>& yzy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z>& yzz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z>& yzz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x>& zxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x>& zxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y>& zxy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y>& zxy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z>& zxz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z>& zxz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x>& zyx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x>& zyx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y>& zyy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y>& zyy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z>& zyz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z>& zyz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x>& zzx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x>& zzx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y>& zzy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y>& zzy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z>& zzz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z>& zzz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z>*>(
        this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

/** Specialization of mem_container for recursion size 4. Inherits from
  mem_container for recursion size 3.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 4> : public mem_container<dataT, kElems, 3> {
 public:
  /** Returns element 0 of the vector in rgba format.
   * @return Returns element 0 of the vector.
   */
  dataT r() { return this->x(); }

  /** Returns element 0 of the vector in rgba format.
   * @return Returns element 0 of the vector.
   */
  const dataT r() const { return this->x(); }

  /** Returns element 1 of the vector in rgba format.
   * @return Returns element 1 of the vector.
   */
  dataT g() { return this->y(); }

  /** Returns element 0 of the vector in rgba format.
   * @return Returns element 0 of the vector.
   */
  const dataT g() const { return this->y(); }

  /** Returns element 2 of the vector in rgba format.
   * @return Returns element 2 of the vector.
   */
  dataT b() { return this->z(); }

  /** Returns element 0 of the vector in rgba format.
   * @return Returns element 0 of the vector.
   */
  const dataT b() const { return this->z(); }

  /** Returns element 3 of the vector in rgba format.
   * @return Returns element 3 of the vector.
   */
  dataT a() { return this->w(); }

  /** Returns element 0 of the vector in rgba format.
   * @return Returns element 0 of the vector.
   */
  const dataT a() const { return this->w(); }

  /** Returns element 3 of the vector in the 16 element format.
   * @return Returns element 3 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s3)

  swizzled_vec<dataT, kElems, detail::w>& w() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w>& w() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w>*>(this);
    return *swizzledVec;
  }
#ifdef SYCL_SIMPLE_SWIZZLES

  swizzled_vec<dataT, kElems, detail::x, detail::w>& xw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w>& xw() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w>& yw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w>& yw() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w>& zw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w>& zw() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x>& wx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x>& wx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y>& wy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y>& wy() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z>& wz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z>& wz() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w>& ww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w>& ww() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w>& xxw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w>& xxw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w>& xyw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w>& xyw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w>& xzw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w>& xzw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x>& xwx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x>& xwx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y>& xwy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y>& xwy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z>& xwz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z>& xwz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w>& xww() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w>& xww()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w>& yxw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w>& yxw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w>& yyw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w>& yyw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w>& yzw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w>& yzw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x>& ywx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x>& ywx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y>& ywy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y>& ywy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z>& ywz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z>& ywz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w>& yww() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w>& yww()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w>& zxw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w>& zxw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w>& zyw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w>& zyw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w>& zzw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w>& zzw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x>& zwx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x>& zwx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y>& zwy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y>& zwy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z>& zwz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z>& zwz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w>& zww() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w>& zww()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x>& wxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x>& wxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y>& wxy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y>& wxy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z>& wxz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z>& wxz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w>& wxw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w>& wxw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x>& wyx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x>& wyx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y>& wyy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y>& wyy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z>& wyz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z>& wyz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w>& wyw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w>& wyw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x>& wzx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x>& wzx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y>& wzy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y>& wzy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z>& wzz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z>& wzz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w>& wzw() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w>& wzw()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x>& wwx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x>& wwx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y>& wwy() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y>& wwy()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z>& wwz() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z>& wwz()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z>*>(
        this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w>& www() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w>& www()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::x>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::x>&
  xxxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::y>&
  xxxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::y>&
  xxxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::z>&
  xxxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::z>&
  xxxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::w>&
  xxxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::x, detail::w>&
  xxxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::x>&
  xxyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::x>&
  xxyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::y>&
  xxyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::y>&
  xxyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::z>&
  xxyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::z>&
  xxyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::w>&
  xxyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::y, detail::w>&
  xxyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::x>&
  xxzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::x>&
  xxzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::y>&
  xxzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::y>&
  xxzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::z>&
  xxzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::z>&
  xxzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::w>&
  xxzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::z, detail::w>&
  xxzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::x>&
  xxwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::x>&
  xxwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::y>&
  xxwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::y>&
  xxwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::z>&
  xxwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::z>&
  xxwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::w>&
  xxww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::x,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::x, detail::w, detail::w>&
  xxww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::x,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::x>&
  xyxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::x>&
  xyxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::y>&
  xyxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::y>&
  xyxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::z>&
  xyxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::z>&
  xyxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::w>&
  xyxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::x, detail::w>&
  xyxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::x>&
  xyyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::x>&
  xyyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::y>&
  xyyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::y>&
  xyyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::z>&
  xyyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::z>&
  xyyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::w>&
  xyyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::y, detail::w>&
  xyyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::x>&
  xyzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::x>&
  xyzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::y>&
  xyzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::y>&
  xyzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::z>&
  xyzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::z>&
  xyzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::w>&
  xyzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::z, detail::w>&
  xyzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::x>&
  xywx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::x>&
  xywx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::y>&
  xywy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::y>&
  xywy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::z>&
  xywz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::z>&
  xywz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::w>&
  xyww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::y,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::y, detail::w, detail::w>&
  xyww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::y,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::x>&
  xzxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::x>&
  xzxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::y>&
  xzxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::y>&
  xzxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::z>&
  xzxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::z>&
  xzxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::w>&
  xzxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::x, detail::w>&
  xzxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::x>&
  xzyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::x>&
  xzyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::y>&
  xzyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::y>&
  xzyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::z>&
  xzyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::z>&
  xzyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::w>&
  xzyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::y, detail::w>&
  xzyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::x>&
  xzzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::x>&
  xzzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::y>&
  xzzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::y>&
  xzzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::z>&
  xzzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::z>&
  xzzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::w>&
  xzzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::z, detail::w>&
  xzzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::x>&
  xzwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::x>&
  xzwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::y>&
  xzwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::y>&
  xzwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::z>&
  xzwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::z>&
  xzwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::w>&
  xzww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::z,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::z, detail::w, detail::w>&
  xzww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::z,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::x>&
  xwxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::x>&
  xwxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::y>&
  xwxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::y>&
  xwxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::z>&
  xwxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::z>&
  xwxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::w>&
  xwxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::x, detail::w>&
  xwxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::x>&
  xwyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::x>&
  xwyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::y>&
  xwyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::y>&
  xwyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::z>&
  xwyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::z>&
  xwyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::w>&
  xwyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::y, detail::w>&
  xwyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::x>&
  xwzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::x>&
  xwzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::y>&
  xwzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::y>&
  xwzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::z>&
  xwzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::z>&
  xwzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::w>&
  xwzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::z, detail::w>&
  xwzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::x>&
  xwwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::x>&
  xwwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::y>&
  xwwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::y>&
  xwwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::z>&
  xwwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::z>&
  xwwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::w>&
  xwww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::x, detail::w,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::x, detail::w, detail::w, detail::w>&
  xwww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::x, detail::w,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::x>&
  yxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::x>&
  yxxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::y>&
  yxxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::y>&
  yxxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::z>&
  yxxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::z>&
  yxxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::w>&
  yxxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::x, detail::w>&
  yxxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::x>&
  yxyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::x>&
  yxyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::y>&
  yxyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::y>&
  yxyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::z>&
  yxyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::z>&
  yxyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::w>&
  yxyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::y, detail::w>&
  yxyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::x>&
  yxzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::x>&
  yxzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::y>&
  yxzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::y>&
  yxzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::z>&
  yxzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::z>&
  yxzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::w>&
  yxzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::z, detail::w>&
  yxzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::x>&
  yxwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::x>&
  yxwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::y>&
  yxwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::y>&
  yxwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::z>&
  yxwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::z>&
  yxwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::w>&
  yxww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::x,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::x, detail::w, detail::w>&
  yxww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::x,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::x>&
  yyxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::x>&
  yyxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::y>&
  yyxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::y>&
  yyxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::z>&
  yyxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::z>&
  yyxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::w>&
  yyxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::x, detail::w>&
  yyxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::x>&
  yyyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::x>&
  yyyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::y>&
  yyyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::y>&
  yyyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::z>&
  yyyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::z>&
  yyyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::w>&
  yyyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::y, detail::w>&
  yyyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::x>&
  yyzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::x>&
  yyzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::y>&
  yyzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::y>&
  yyzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::z>&
  yyzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::z>&
  yyzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::w>&
  yyzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::z, detail::w>&
  yyzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::x>&
  yywx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::x>&
  yywx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::y>&
  yywy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::y>&
  yywy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::z>&
  yywz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::z>&
  yywz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::w>&
  yyww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::y,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::y, detail::w, detail::w>&
  yyww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::y,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::x>&
  yzxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::x>&
  yzxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::y>&
  yzxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::y>&
  yzxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::z>&
  yzxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::z>&
  yzxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::w>&
  yzxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::x, detail::w>&
  yzxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::x>&
  yzyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::x>&
  yzyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::y>&
  yzyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::y>&
  yzyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::z>&
  yzyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::z>&
  yzyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::w>&
  yzyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::y, detail::w>&
  yzyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::x>&
  yzzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::x>&
  yzzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::y>&
  yzzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::y>&
  yzzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::z>&
  yzzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::z>&
  yzzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::w>&
  yzzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::z, detail::w>&
  yzzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::x>&
  yzwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::x>&
  yzwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::y>&
  yzwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::y>&
  yzwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::z>&
  yzwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::z>&
  yzwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::w>&
  yzww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::z,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::z, detail::w, detail::w>&
  yzww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::z,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::x>&
  ywxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::x>&
  ywxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::y>&
  ywxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::y>&
  ywxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::z>&
  ywxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::z>&
  ywxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::w>&
  ywxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::x, detail::w>&
  ywxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::x>&
  ywyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::x>&
  ywyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::y>&
  ywyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::y>&
  ywyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::z>&
  ywyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::z>&
  ywyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::w>&
  ywyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::y, detail::w>&
  ywyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::x>&
  ywzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::x>&
  ywzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::y>&
  ywzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::y>&
  ywzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::z>&
  ywzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::z>&
  ywzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::w>&
  ywzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::z, detail::w>&
  ywzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::x>&
  ywwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::x>&
  ywwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::y>&
  ywwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::y>&
  ywwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::z>&
  ywwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::z>&
  ywwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::w>&
  ywww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::y, detail::w,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::y, detail::w, detail::w, detail::w>&
  ywww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::y, detail::w,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::x>&
  zxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::x>&
  zxxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::y>&
  zxxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::y>&
  zxxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::z>&
  zxxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::z>&
  zxxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::w>&
  zxxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::x, detail::w>&
  zxxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::x>&
  zxyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::x>&
  zxyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::y>&
  zxyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::y>&
  zxyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::z>&
  zxyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::z>&
  zxyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::w>&
  zxyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::y, detail::w>&
  zxyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::x>&
  zxzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::x>&
  zxzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::y>&
  zxzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::y>&
  zxzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::z>&
  zxzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::z>&
  zxzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::w>&
  zxzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::z, detail::w>&
  zxzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::x>&
  zxwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::x>&
  zxwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::y>&
  zxwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::y>&
  zxwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::z>&
  zxwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::z>&
  zxwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::w>&
  zxww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::x,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::x, detail::w, detail::w>&
  zxww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::x,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::x>&
  zyxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::x>&
  zyxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::y>&
  zyxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::y>&
  zyxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::z>&
  zyxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::z>&
  zyxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::w>&
  zyxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::x, detail::w>&
  zyxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::x>&
  zyyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::x>&
  zyyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::y>&
  zyyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::y>&
  zyyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::z>&
  zyyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::z>&
  zyyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::w>&
  zyyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::y, detail::w>&
  zyyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::x>&
  zyzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::x>&
  zyzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::y>&
  zyzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::y>&
  zyzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::z>&
  zyzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::z>&
  zyzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::w>&
  zyzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::z, detail::w>&
  zyzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::x>&
  zywx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::x>&
  zywx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::y>&
  zywy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::y>&
  zywy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::z>&
  zywz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::z>&
  zywz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::w>&
  zyww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::y,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::y, detail::w, detail::w>&
  zyww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::y,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::x>&
  zzxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::x>&
  zzxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::y>&
  zzxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::y>&
  zzxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::z>&
  zzxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::z>&
  zzxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::w>&
  zzxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::x, detail::w>&
  zzxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::x>&
  zzyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::x>&
  zzyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::y>&
  zzyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::y>&
  zzyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::z>&
  zzyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::z>&
  zzyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::w>&
  zzyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::y, detail::w>&
  zzyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::x>&
  zzzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::x>&
  zzzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::y>&
  zzzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::y>&
  zzzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::z>&
  zzzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::z>&
  zzzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::w>&
  zzzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::z, detail::w>&
  zzzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::x>&
  zzwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::x>&
  zzwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::y>&
  zzwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::y>&
  zzwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::z>&
  zzwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::z>&
  zzwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::w>&
  zzww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::z,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::z, detail::w, detail::w>&
  zzww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::z,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::x>&
  zwxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::x>&
  zwxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::y>&
  zwxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::y>&
  zwxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::z>&
  zwxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::z>&
  zwxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::w>&
  zwxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::x, detail::w>&
  zwxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::x>&
  zwyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::x>&
  zwyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::y>&
  zwyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::y>&
  zwyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::z>&
  zwyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::z>&
  zwyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::w>&
  zwyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::y, detail::w>&
  zwyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::x>&
  zwzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::x>&
  zwzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::y>&
  zwzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::y>&
  zwzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::z>&
  zwzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::z>&
  zwzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::w>&
  zwzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::z, detail::w>&
  zwzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::x>&
  zwwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::x>&
  zwwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::y>&
  zwwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::y>&
  zwwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::z>&
  zwwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::z>&
  zwwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::w>&
  zwww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::z, detail::w,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::z, detail::w, detail::w, detail::w>&
  zwww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::z, detail::w,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::x>&
  wxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::x>&
  wxxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::y>&
  wxxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::y>&
  wxxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::z>&
  wxxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::z>&
  wxxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::w>&
  wxxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::x, detail::w>&
  wxxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::x>&
  wxyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::x>&
  wxyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::y>&
  wxyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::y>&
  wxyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::z>&
  wxyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::z>&
  wxyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::w>&
  wxyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::y, detail::w>&
  wxyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::x>&
  wxzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::x>&
  wxzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::y>&
  wxzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::y>&
  wxzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::z>&
  wxzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::z>&
  wxzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::w>&
  wxzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::z, detail::w>&
  wxzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::x>&
  wxwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::x>&
  wxwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::y>&
  wxwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::y>&
  wxwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::z>&
  wxwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::z>&
  wxwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::w>&
  wxww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::x,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::x, detail::w, detail::w>&
  wxww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::x,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::x>&
  wyxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::x>&
  wyxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::y>&
  wyxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::y>&
  wyxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::z>&
  wyxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::z>&
  wyxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::w>&
  wyxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::x, detail::w>&
  wyxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::x>&
  wyyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::x>&
  wyyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::y>&
  wyyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::y>&
  wyyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::z>&
  wyyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::z>&
  wyyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::w>&
  wyyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::y, detail::w>&
  wyyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::x>&
  wyzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::x>&
  wyzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::y>&
  wyzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::y>&
  wyzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::z>&
  wyzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::z>&
  wyzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::w>&
  wyzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::z, detail::w>&
  wyzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::x>&
  wywx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::x>&
  wywx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::y>&
  wywy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::y>&
  wywy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::z>&
  wywz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::z>&
  wywz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::w>&
  wyww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::y,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::y, detail::w, detail::w>&
  wyww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::y,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::x>&
  wzxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::x>&
  wzxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::y>&
  wzxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::y>&
  wzxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::z>&
  wzxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::z>&
  wzxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::w>&
  wzxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::x, detail::w>&
  wzxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::x>&
  wzyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::x>&
  wzyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::y>&
  wzyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::y>&
  wzyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::z>&
  wzyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::z>&
  wzyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::w>&
  wzyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::y, detail::w>&
  wzyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::x>&
  wzzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::x>&
  wzzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::y>&
  wzzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::y>&
  wzzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::z>&
  wzzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::z>&
  wzzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::w>&
  wzzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::z, detail::w>&
  wzzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::x>&
  wzwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::x>&
  wzwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::y>&
  wzwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::y>&
  wzwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::z>&
  wzwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::z>&
  wzwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::w>&
  wzww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::z,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::z, detail::w, detail::w>&
  wzww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::z,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::x>&
  wwxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::x, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::x>&
  wwxx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::x, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::y>&
  wwxy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::x, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::y>&
  wwxy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::x, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::z>&
  wwxz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::x, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::z>&
  wwxz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::x, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::w>&
  wwxw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::x, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::x, detail::w>&
  wwxw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::x, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::x>&
  wwyx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::y, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::x>&
  wwyx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::y, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::y>&
  wwyy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::y, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::y>&
  wwyy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::y, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::z>&
  wwyz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::y, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::z>&
  wwyz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::y, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::w>&
  wwyw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::y, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::y, detail::w>&
  wwyw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::y, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::x>&
  wwzx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::z, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::x>&
  wwzx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::z, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::y>&
  wwzy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::z, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::y>&
  wwzy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::z, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::z>&
  wwzz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::z, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::z>&
  wwzz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::z, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::w>&
  wwzw() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::z, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::z, detail::w>&
  wwzw() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::z, detail::w>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::x>&
  wwwx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::w, detail::x>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::x>&
  wwwx() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::w, detail::x>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::y>&
  wwwy() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::w, detail::y>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::y>&
  wwwy() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::w, detail::y>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::z>&
  wwwz() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::w, detail::z>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::z>&
  wwwz() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::w, detail::z>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::w>&
  wwww() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::w, detail::w,
                                      detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::w, detail::w, detail::w, detail::w>&
  wwww() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::w, detail::w,
                                            detail::w, detail::w>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::r>&
  rrrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::r>&
  rrrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::g>&
  rrrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::g>&
  rrrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::b>&
  rrrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::b>&
  rrrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::a>&
  rrra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::r, detail::a>&
  rrra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::r>&
  rrgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::r>&
  rrgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::g>&
  rrgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::g>&
  rrgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::b>&
  rrgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::b>&
  rrgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::a>&
  rrga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::g, detail::a>&
  rrga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::r>&
  rrbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::r>&
  rrbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::g>&
  rrbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::g>&
  rrbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::b>&
  rrbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::b>&
  rrbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::a>&
  rrba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::b, detail::a>&
  rrba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::r>&
  rrar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::r>&
  rrar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::g>&
  rrag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::g>&
  rrag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::b>&
  rrab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::b>&
  rrab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::a>&
  rraa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::r,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::r, detail::a, detail::a>&
  rraa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::r,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::r>&
  rgrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::r>&
  rgrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::g>&
  rgrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::g>&
  rgrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::b>&
  rgrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::b>&
  rgrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::a>&
  rgra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::r, detail::a>&
  rgra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::r>&
  rggr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::r>&
  rggr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::g>&
  rggg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::g>&
  rggg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::b>&
  rggb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::b>&
  rggb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::a>&
  rgga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::g, detail::a>&
  rgga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::r>&
  rgbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::r>&
  rgbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::g>&
  rgbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::g>&
  rgbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::b>&
  rgbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::b>&
  rgbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::a>&
  rgba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::b, detail::a>&
  rgba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::r>&
  rgar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::r>&
  rgar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::g>&
  rgag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::g>&
  rgag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::b>&
  rgab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::b>&
  rgab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::a>&
  rgaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::g,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::g, detail::a, detail::a>&
  rgaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::g,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::r>&
  rbrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::r>&
  rbrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::g>&
  rbrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::g>&
  rbrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::b>&
  rbrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::b>&
  rbrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::a>&
  rbra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::r, detail::a>&
  rbra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::r>&
  rbgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::r>&
  rbgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::g>&
  rbgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::g>&
  rbgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::b>&
  rbgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::b>&
  rbgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::a>&
  rbga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::g, detail::a>&
  rbga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::r>&
  rbbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::r>&
  rbbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::g>&
  rbbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::g>&
  rbbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::b>&
  rbbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::b>&
  rbbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::a>&
  rbba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::b, detail::a>&
  rbba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::r>&
  rbar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::r>&
  rbar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::g>&
  rbag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::g>&
  rbag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::b>&
  rbab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::b>&
  rbab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::a>&
  rbaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::b,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::b, detail::a, detail::a>&
  rbaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::b,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::r>&
  rarr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::r>&
  rarr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::g>&
  rarg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::g>&
  rarg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::b>&
  rarb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::b>&
  rarb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::a>&
  rara() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::r, detail::a>&
  rara() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::r>&
  ragr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::r>&
  ragr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::g>&
  ragg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::g>&
  ragg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::b>&
  ragb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::b>&
  ragb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::a>&
  raga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::g, detail::a>&
  raga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::r>&
  rabr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::r>&
  rabr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::g>&
  rabg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::g>&
  rabg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::b>&
  rabb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::b>&
  rabb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::a>&
  raba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::b, detail::a>&
  raba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::r>&
  raar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::r>&
  raar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::g>&
  raag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::g>&
  raag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::b>&
  raab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::b>&
  raab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::a>&
  raaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::r, detail::a,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::r, detail::a, detail::a, detail::a>&
  raaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::r, detail::a,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::r>&
  grrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::r>&
  grrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::g>&
  grrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::g>&
  grrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::b>&
  grrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::b>&
  grrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::a>&
  grra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::r, detail::a>&
  grra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::r>&
  grgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::r>&
  grgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::g>&
  grgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::g>&
  grgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::b>&
  grgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::b>&
  grgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::a>&
  grga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::g, detail::a>&
  grga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::r>&
  grbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::r>&
  grbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::g>&
  grbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::g>&
  grbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::b>&
  grbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::b>&
  grbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::a>&
  grba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::b, detail::a>&
  grba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::r>&
  grar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::r>&
  grar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::g>&
  grag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::g>&
  grag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::b>&
  grab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::b>&
  grab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::a>&
  graa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::r,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::r, detail::a, detail::a>&
  graa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::r,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::r>&
  ggrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::r>&
  ggrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::g>&
  ggrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::g>&
  ggrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::b>&
  ggrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::b>&
  ggrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::a>&
  ggra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::r, detail::a>&
  ggra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::r>&
  gggr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::r>&
  gggr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::g>&
  gggg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::g>&
  gggg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::b>&
  gggb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::b>&
  gggb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::a>&
  ggga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::g, detail::a>&
  ggga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::r>&
  ggbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::r>&
  ggbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::g>&
  ggbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::g>&
  ggbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::b>&
  ggbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::b>&
  ggbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::a>&
  ggba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::b, detail::a>&
  ggba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::r>&
  ggar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::r>&
  ggar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::g>&
  ggag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::g>&
  ggag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::b>&
  ggab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::b>&
  ggab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::a>&
  ggaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::g,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::g, detail::a, detail::a>&
  ggaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::g,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::r>&
  gbrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::r>&
  gbrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::g>&
  gbrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::g>&
  gbrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::b>&
  gbrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::b>&
  gbrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::a>&
  gbra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::r, detail::a>&
  gbra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::r>&
  gbgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::r>&
  gbgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::g>&
  gbgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::g>&
  gbgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::b>&
  gbgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::b>&
  gbgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::a>&
  gbga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::g, detail::a>&
  gbga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::r>&
  gbbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::r>&
  gbbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::g>&
  gbbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::g>&
  gbbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::b>&
  gbbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::b>&
  gbbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::a>&
  gbba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::b, detail::a>&
  gbba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::r>&
  gbar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::r>&
  gbar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::g>&
  gbag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::g>&
  gbag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::b>&
  gbab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::b>&
  gbab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::a>&
  gbaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::b,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::b, detail::a, detail::a>&
  gbaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::b,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::r>&
  garr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::r>&
  garr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::g>&
  garg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::g>&
  garg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::b>&
  garb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::b>&
  garb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::a>&
  gara() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::r, detail::a>&
  gara() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::r>&
  gagr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::r>&
  gagr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::g>&
  gagg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::g>&
  gagg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::b>&
  gagb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::b>&
  gagb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::a>&
  gaga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::g, detail::a>&
  gaga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::r>&
  gabr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::r>&
  gabr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::g>&
  gabg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::g>&
  gabg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::b>&
  gabb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::b>&
  gabb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::a>&
  gaba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::b, detail::a>&
  gaba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::r>&
  gaar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::r>&
  gaar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::g>&
  gaag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::g>&
  gaag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::b>&
  gaab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::b>&
  gaab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::a>&
  gaaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::g, detail::a,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::g, detail::a, detail::a, detail::a>&
  gaaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::g, detail::a,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::r>&
  brrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::r>&
  brrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::g>&
  brrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::g>&
  brrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::b>&
  brrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::b>&
  brrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::a>&
  brra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::r, detail::a>&
  brra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::r>&
  brgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::r>&
  brgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::g>&
  brgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::g>&
  brgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::b>&
  brgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::b>&
  brgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::a>&
  brga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::g, detail::a>&
  brga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::r>&
  brbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::r>&
  brbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::g>&
  brbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::g>&
  brbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::b>&
  brbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::b>&
  brbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::a>&
  brba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::b, detail::a>&
  brba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::r>&
  brar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::r>&
  brar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::g>&
  brag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::g>&
  brag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::b>&
  brab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::b>&
  brab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::a>&
  braa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::r,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::r, detail::a, detail::a>&
  braa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::r,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::r>&
  bgrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::r>&
  bgrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::g>&
  bgrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::g>&
  bgrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::b>&
  bgrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::b>&
  bgrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::a>&
  bgra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::r, detail::a>&
  bgra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::r>&
  bggr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::r>&
  bggr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::g>&
  bggg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::g>&
  bggg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::b>&
  bggb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::b>&
  bggb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::a>&
  bgga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::g, detail::a>&
  bgga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::r>&
  bgbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::r>&
  bgbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::g>&
  bgbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::g>&
  bgbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::b>&
  bgbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::b>&
  bgbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::a>&
  bgba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::b, detail::a>&
  bgba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::r>&
  bgar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::r>&
  bgar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::g>&
  bgag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::g>&
  bgag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::b>&
  bgab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::b>&
  bgab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::a>&
  bgaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::g,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::g, detail::a, detail::a>&
  bgaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::g,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::r>&
  bbrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::r>&
  bbrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::g>&
  bbrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::g>&
  bbrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::b>&
  bbrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::b>&
  bbrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::a>&
  bbra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::r, detail::a>&
  bbra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::r>&
  bbgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::r>&
  bbgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::g>&
  bbgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::g>&
  bbgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::b>&
  bbgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::b>&
  bbgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::a>&
  bbga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::g, detail::a>&
  bbga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::r>&
  bbbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::r>&
  bbbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::g>&
  bbbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::g>&
  bbbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::b>&
  bbbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::b>&
  bbbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::a>&
  bbba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::b, detail::a>&
  bbba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::r>&
  bbar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::r>&
  bbar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::g>&
  bbag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::g>&
  bbag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::b>&
  bbab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::b>&
  bbab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::a>&
  bbaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::b,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::b, detail::a, detail::a>&
  bbaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::b,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::r>&
  barr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::r>&
  barr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::g>&
  barg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::g>&
  barg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::b>&
  barb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::b>&
  barb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::a>&
  bara() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::r, detail::a>&
  bara() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::r>&
  bagr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::r>&
  bagr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::g>&
  bagg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::g>&
  bagg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::b>&
  bagb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::b>&
  bagb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::a>&
  baga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::g, detail::a>&
  baga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::r>&
  babr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::r>&
  babr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::g>&
  babg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::g>&
  babg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::b>&
  babb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::b>&
  babb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::a>&
  baba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::b, detail::a>&
  baba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::r>&
  baar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::r>&
  baar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::g>&
  baag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::g>&
  baag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::b>&
  baab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::b>&
  baab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::a>&
  baaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::b, detail::a,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::b, detail::a, detail::a, detail::a>&
  baaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::b, detail::a,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::r>&
  arrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::r>&
  arrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::g>&
  arrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::g>&
  arrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::b>&
  arrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::b>&
  arrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::a>&
  arra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::r, detail::a>&
  arra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::r>&
  argr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::r>&
  argr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::g>&
  argg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::g>&
  argg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::b>&
  argb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::b>&
  argb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::a>&
  arga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::g, detail::a>&
  arga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::r>&
  arbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::r>&
  arbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::g>&
  arbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::g>&
  arbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::b>&
  arbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::b>&
  arbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::a>&
  arba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::b, detail::a>&
  arba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::r>&
  arar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::r>&
  arar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::g>&
  arag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::g>&
  arag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::b>&
  arab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::b>&
  arab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::a>&
  araa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::r,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::r, detail::a, detail::a>&
  araa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::r,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::r>&
  agrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::r>&
  agrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::g>&
  agrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::g>&
  agrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::b>&
  agrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::b>&
  agrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::a>&
  agra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::r, detail::a>&
  agra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::r>&
  aggr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::r>&
  aggr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::g>&
  aggg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::g>&
  aggg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::b>&
  aggb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::b>&
  aggb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::a>&
  agga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::g, detail::a>&
  agga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::r>&
  agbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::r>&
  agbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::g>&
  agbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::g>&
  agbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::b>&
  agbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::b>&
  agbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::a>&
  agba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::b, detail::a>&
  agba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::r>&
  agar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::r>&
  agar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::g>&
  agag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::g>&
  agag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::b>&
  agab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::b>&
  agab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::a>&
  agaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::g,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::g, detail::a, detail::a>&
  agaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::g,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::r>&
  abrr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::r>&
  abrr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::g>&
  abrg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::g>&
  abrg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::b>&
  abrb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::b>&
  abrb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::a>&
  abra() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::r, detail::a>&
  abra() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::r>&
  abgr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::r>&
  abgr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::g>&
  abgg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::g>&
  abgg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::b>&
  abgb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::b>&
  abgb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::a>&
  abga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::g, detail::a>&
  abga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::r>&
  abbr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::r>&
  abbr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::g>&
  abbg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::g>&
  abbg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::b>&
  abbb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::b>&
  abbb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::a>&
  abba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::b, detail::a>&
  abba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::r>&
  abar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::r>&
  abar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::g>&
  abag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::g>&
  abag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::b>&
  abab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::b>&
  abab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::a>&
  abaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::b,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::b, detail::a, detail::a>&
  abaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::b,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::r>&
  aarr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::r, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::r>&
  aarr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::r, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::g>&
  aarg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::r, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::g>&
  aarg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::r, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::b>&
  aarb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::r, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::b>&
  aarb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::r, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::a>&
  aara() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::r, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::r, detail::a>&
  aara() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::r, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::r>&
  aagr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::g, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::r>&
  aagr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::g, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::g>&
  aagg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::g, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::g>&
  aagg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::g, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::b>&
  aagb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::g, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::b>&
  aagb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::g, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::a>&
  aaga() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::g, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::g, detail::a>&
  aaga() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::g, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::r>&
  aabr() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::b, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::r>&
  aabr() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::b, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::g>&
  aabg() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::b, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::g>&
  aabg() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::b, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::b>&
  aabb() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::b, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::b>&
  aabb() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::b, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::a>&
  aaba() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::b, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::b, detail::a>&
  aaba() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::b, detail::a>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::r>&
  aaar() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::a, detail::r>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::r>&
  aaar() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::a, detail::r>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::g>&
  aaag() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::a, detail::g>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::g>&
  aaag() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::a, detail::g>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::b>&
  aaab() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::a, detail::b>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::b>&
  aaab() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::a, detail::b>*>(this);
    return *swizzledVec;
  }
  swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::a>&
  aaaa() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::a, detail::a,
                                      detail::a, detail::a>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::a, detail::a, detail::a, detail::a>&
  aaaa() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::a, detail::a,
                                            detail::a, detail::a>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

/** Specialization of mem_container for recursion size 8. Inherits from
  mem_container_base.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 8>
    : public mem_container_base<dataT, kElems> {
 public:
  /** Returns element 0 of the vector in the 16 element format.
   * @return Returns element 0 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s0)

  /** Returns element 1 of the vector in the 16 element format.
   * @return Returns element 1 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s1)

  /** Returns element 2 of the vector in the 16 element format.
   * @return Returns element 2 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s2)

  /** Returns element 3 of the vector in the 16 element format.
   * @return Returns element 3 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s3)

  /** Returns element 4 of the vector in the 16 element format.
   * @return Returns element 4 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s4)

  /** Returns element 5 of the vector in the 16 element format.
   * @return Returns element 5 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s5)

  /** Returns element 6 of the vector in the 16 element format.
   * @return Returns element 6 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s6)

  /** Returns element 7 of the vector in the 16 element format.
   * @return Returns element 7 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s7)
};

/** Specialization of mem_container for recursion size 16. Inherits from
  mem_container for recursion size 8.
*/
template <typename dataT, int kElems>
class mem_container<dataT, kElems, 16>
    : public mem_container<dataT, kElems, 8> {
 public:
  /** Returns element 8 of the vector in the 16 element format.
   * @return Returns element 8 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s8)

  /** Returns element 9 of the vector in the 16 element format.
   * @return Returns element 9 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s9)

  /** Returns element 10 of the vector in the 16 element format.
   * @return Returns element 10 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sA)

  /** Returns element 11 of the vector in the 16 element format.
   * @return Returns element 11 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sB)

  /** Returns element 12 of the vector in the 16 element format.
   * @return Returns element 12 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sC)

  /** Returns element 13 of the vector in the 16 element format.
   * @return Returns element 13 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sD)

  /** Returns element 14 of the vector in the 16 element format.
   * @return Returns element 14 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sE)

  /** Returns element 15 of the vector in the 16 element format.
   * @return Returns element 15 of the vector.
   */
  COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(sF)
};

}  // namespace detail

namespace detail {
namespace vec_ops {

template <size_t size>
struct logical_return {};
template <>
struct logical_return<1> {
  using type = cl::sycl::cl_char;
};
template <>
struct logical_return<2> {
  using type = cl::sycl::cl_short;
};
template <>
struct logical_return<4> {
  using type = cl::sycl::cl_int;
};
template <>
struct logical_return<8> {
  using type = cl::sycl::cl_long;
};

#ifdef __SYCL_DEVICE_ONLY__

/** @brief Performs the assignment for a logical op for vectors.
 * @tparam kElems The number of elements of the vector.
 */
template <int kElems>
struct logical_op {
  /** @brief Performs an assignment of op_res to m_data.
   * Used for 2+ element vectors.
   * Reinterprets op_res to the required type for m_data.
   * @tparam return_t The return type of m_data.
   * @tparam T the type of op_res.
   * @param m_data The data to be assigned to.
   * @param op_res The value to be assigned.
   */
  template <typename return_t, typename T>
  static void assign(return_t& m_data, T&& op_res) {
    m_data = reinterpret_cast<return_t>(op_res);
  }
};

/** @brief Performs the assignment for a logical op for vectors.
 */
template <>
struct logical_op<1> {
  /** @brief Performs an assignment of op_res to m_data.
   * Used for one element vectors.
   * Assigns the negated value of op_res as logical_ops return 1 for scalars
   * and -1 for vectors.
   * @tparam return_t The return type of m_data.
   * @tparam T the type of op_res.
   * @param m_data The data to be assigned to.
   * @param op_res The value to be assigned.
   */
  template <typename return_t, typename T>
  static void assign(return_t& m_data, T&& op_res) {
    m_data = -op_res;
  }
};

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace vec_ops
}  // namespace detail

/** @brief Vector class for host and device.
 * @tparam dataT The data type for the vector.
 * @tparam kElems The number of elements for the vector.
 */
template <typename dataT, int kElems>
class vec : public detail::mem_container<dataT, kElems, kElems> {
 public:
  using element_type = dataT;
  static const int width = kElems;

  /** Alias to the underlying vector type. Device only.
   */
#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = detail::__sycl_vector<dataT, kElems>;
#endif

  /** Initializes all vector values to 0.
   */
  vec();

  /** Initializes all vector values to the scalar.
   */
  explicit vec(dataT arg) {
    for (int i = 0; i < kElems; i++) {
      this->set_value(i, arg);
    }
  }

  /** Performs a rhs swizzle operation and assigns the result to a new vec
   * object. Enabled only if the number of swizzled indexes
   * (sizeof...(kIndexRhsN)) match the width of the rhs vector.
   * @tparam kElemsRhs The number of elements in rhs swizzled_vec object.
   * @tparam kIndexRhsN The variadic argument pack for swizzle indexes of
   * swizzled_vec parameter.
   * @param rhs The rhs swizzled_vec reference.
   */
  template <int kElemsRhs, int... kIndexRhsN,
            typename E = typename std::enable_if<
                sizeof...(kIndexRhsN) == kElems && (kElems > 1), dataT>::type>
  vec(const swizzled_vec<dataT, kElemsRhs, kIndexRhsN...>& rhs);

  /** Constructor that takes any combination of scalars and vectors.
   * @tparam recurseT0 The first argument.
   * @tparam recurseT1 The second argument.
   * @tparam recurseTN The variadic argument pack for rest of arguments.
   * @param arg0 The first parameter.
   * @param arg1 The second parameter.
   * @param args The rest of parameters.
   */
  template <typename recurseT0, typename recurseT1, typename... recurseTN>
  vec(recurseT0 arg0, recurseT1 arg1, recurseTN... args);

#ifdef __SYCL_DEVICE_ONLY__
  /** Initializes the vector with the __sycl_vector parameter. Device only.
   */
  template <COMPUTECPP_ENABLE_IF(dataT, (kElems > 1))>
  explicit vec(vector_t rhs) {
    this->m_data = rhs;
  }

  /**
   * Implicitly converts a vector to the underlying vector type. Device only.
   */
  operator vector_t();

  /**
   * Implicitly converts a vector to the underlying vector type. Device only.
   */
  operator vector_t() const;

#endif  // __SYCL_DEVICE_ONLY__

#ifndef __SYCL_DEVICE_ONLY__

  /**! constructor (abacus converter)
   * Initializes a vector from the corresponding abacus_* vector type. Host
   * only. This constructor will copy the data from the abacus_vector.
   */
  template <typename abacusT>
  vec(const abacus_vector<abacusT, kElems>& rhs);

  /**! assignment conversion operator (abacus converter)
   * This conversion operator will copy the data from the abacus_vector.
   * Note: A reinterpret_cast between the two vectors is tested to work and can
   * avoid unnecessary copying.
   */
  template <typename abacusT>
  // NOLINTNEXTLINE(misc-unconventional-assign-operator)
  vec<abacusT, kElems>& operator=(const abacus_vector<abacusT, kElems>& rhs);
#endif

#ifdef __SYCL_DEVICE_ONLY__
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 1)>
  operator dataT() {
    return this->m_data;
  }
#else
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 1)>
  operator dataT() {
    return this->m_data[0];
  }
#endif  // __SYCL_DEVICE_ONLY__

  /** Assigns a scalar object all vector values of this object.
   * @param rhs The rhs scalar reference.
   * @return this object.
   */
  vec<dataT, kElems>& operator=(dataT rhs);

  /** @brief Converts this SYCL vec type to a SYCL vec of a different element
   *
   *        Converts this SYCL vec type to a SYCL vec of a different element
   *        type specified by \ref{converT} with the same number of elements
   *        using the rounding mode specified by \ref{roundingMode}
   * @tparam converT Underlying type of the converted vector
   * @tparam roundingMode Rounding mode of the converted vector
   * @return The converted vector
   */
  template <typename convertT, rounding_mode roundingMode,
            typename =
                typename std::enable_if<detail::is_valid_vec_convert_conversion<
                    vec<convertT, kElems>, vec<dataT, kElems>>::value>::type>
  vec<convertT, kElems> convert() const;

  /** Re-interpret the vec to the new type, requires asT to be a specialisation
   * of the vec class and asT to have the same size as dataT.
   * @tparam asT The type of the vec to be returned.
   * @return The re-interpreted vector
   */
  template <typename asT, typename = typename std::enable_if<
                              detail::is_valid_vec_as_conversion<
                                  asT, vec<dataT, kElems>>::value>::type>
  asT as() const;

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies + to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec operator+(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data + rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies + to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec operator+(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] + rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies - to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec operator-(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data - rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies - to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec operator-(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] - rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies * to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec operator*(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data * rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies * to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec operator*(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] * rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies / to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec operator/(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data / rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies / to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec operator/(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] / rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies % to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec operator%(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data % rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies % to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec operator%(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] % rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies & to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec operator&(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data & rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies & to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec operator&(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] & rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies | to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec operator|(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data | rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies | to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec operator|(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] | rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies ^ to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec operator^(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data ^ rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies ^ to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec operator^(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] ^ rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies << to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec operator<<(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data << rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies << to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec operator<<(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] << rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >> to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec operator>>(const vec& rhs) const {
    vec newVec;
    newVec.m_data = this->m_data >> rhs.m_data;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >> to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec operator>>(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] >> rhs.get_value(i));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies + to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec operator+(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data + rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies + to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec operator+(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] + rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies - to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec operator-(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data - rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies - to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec operator-(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] - rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies * to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec operator*(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data * rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies * to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec operator*(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] * rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies / to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec operator/(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data / rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies / to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec operator/(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] / rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies % to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec operator%(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data % rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies % to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec operator%(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] % rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies & to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec operator&(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data & rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies & to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec operator&(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] & rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies | to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec operator|(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data | rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies | to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec operator|(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] | rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies ^ to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec operator^(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data ^ rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies ^ to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec operator^(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] ^ rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies << to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec operator<<(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data << rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies << to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec operator<<(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] << rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >> to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec operator>>(dataT rhs) const {
    vec newVec;
    newVec.m_data = this->m_data >> rhs;
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >> to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec operator>>(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, this->m_data[i] >> rhs);
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies += to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator+=(const vec& rhs) {
    this->m_data += rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies += to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator+=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] += rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies -= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator-=(const vec& rhs) {
    this->m_data -= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies -= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator-=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] -= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies *= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator*=(const vec& rhs) {
    this->m_data *= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies *= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator*=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] *= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies /= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator/=(const vec& rhs) {
    this->m_data /= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies /= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator/=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] /= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies %= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator%=(const vec& rhs) {
    this->m_data %= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies %= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator%=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] %= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies &= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator&=(const vec& rhs) {
    this->m_data &= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies &= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator&=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] &= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies |= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator|=(const vec& rhs) {
    this->m_data |= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies |= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator|=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] |= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies ^= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator^=(const vec& rhs) {
    this->m_data ^= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies ^= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator^=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] ^= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies <<= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator<<=(const vec& rhs) {
    this->m_data <<= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies <<= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator<<=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] <<= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >>= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator>>=(const vec& rhs) {
    this->m_data >>= rhs.m_data;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >>= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  vec& operator>>=(const vec& rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] >>= rhs.m_data[i];
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies += to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator+=(dataT rhs) {
    this->m_data += rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies += to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator+=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] += rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies -= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator-=(dataT rhs) {
    this->m_data -= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies -= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator-=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] -= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies *= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator*=(dataT rhs) {
    this->m_data *= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies *= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator*=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] *= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies /= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator/=(dataT rhs) {
    this->m_data /= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies /= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator/=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] /= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies %= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator%=(dataT rhs) {
    this->m_data %= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies %= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator%=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] %= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies &= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator&=(dataT rhs) {
    this->m_data &= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies &= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator&=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] &= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies |= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator|=(dataT rhs) {
    this->m_data |= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies |= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator|=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] |= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies ^= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator^=(dataT rhs) {
    this->m_data ^= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies ^= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator^=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] ^= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies <<= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator<<=(dataT rhs) {
    this->m_data <<= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies <<= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator<<=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] <<= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >>= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator>>=(dataT rhs) {
    this->m_data >>= rhs;
    return *this;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >>= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  vec& operator>>=(dataT rhs) {
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] >>= rhs;
    }
    return *this;
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies && to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator&&(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data && rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies && to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator&&(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] && rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies || to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator||(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data || rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies || to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator||(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] || rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies == to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator==(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data == rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies == to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator==(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] == rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies != to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!=(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data != rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies != to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!=(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] != rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies < to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data < rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies < to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] < rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies > to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data > rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies > to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] > rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies <= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<=(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data <= rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies <= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<=(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] <= rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>=(const vec& rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data >= rhs.m_data);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >= to this vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>=(const vec& rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] >= rhs.get_value(i)));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies && to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator&&(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data && rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies && to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator&&(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] && rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies || to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator||(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data || rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies || to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator||(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] || rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies == to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator==(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data == rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies == to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator==(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] == rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies != to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!=(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data != rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies != to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!=(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] != rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies < to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data < rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies < to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] < rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies > to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data > rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies > to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] > rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies <= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<=(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data <= rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies <= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator<=(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] <= rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Applies >= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>=(dataT rhs) const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data,
                                                this->m_data >= rhs);
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Applies >= to this vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator>=(dataT rhs) const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    for (int i = 0; i < kElems; i++) {
      newVec.set_value(i, -(this->m_data[i] >= rhs));
    }
    return newVec;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

  /** @brief Creates a new vec from the application of ! to each vector element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!() const {
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        newVec;
    detail::vec_ops::logical_op<kElems>::assign(newVec.m_data, !(this->m_data));
    return newVec;
  }

#else  // __SYCL_DEVICE_ONLY__

  /** @brief Creates a new vec from the application of ! to each vector element.
   * @return A new vector of correct ype with the result of the ~ operation.
   */
  template <typename U = dataT>
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
  operator!() const {
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
    COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
    COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
    vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
        result;
    for (int i = 0; i < kElems; i++) {
      result.set_value(i, -(!(this->m_data[i])));
    }
    return result;
    COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  }

#endif  // __SYCL_DEVICE_ONLY__

  COMPUTECPP_CLANG_FORMAT_BARRIER

  /** @brief Pre-increment operator.
   * @return this object.
   */
  vec<dataT, kElems>& operator++();

  /** @brief Post-increment operator.
   * @return The old value before increment.
   */
  vec<dataT, kElems> operator++(int);

  /** @brief Pre-increment operator.
   * @return this object.
   */
  vec<dataT, kElems>& operator--();

  /** @brief Post-increment operator.
   * @return The old value before increment.
   */
  vec<dataT, kElems> operator--(int);

  /** @brief Creates a new vec objects and negates the sign for each vector
   * value.
   * @return Returns an int vector representing the result of the operation.
   */
  vec<dataT, kElems> operator-() const;

  /** @brief Creates a new vec objects and perform the 1-complement for each
   * vector value.
   * @return Returns an int vector representing the result of the operation.
   */
  vec<dataT, kElems> operator~() const;

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s1> hi() const {
    return this->swizzle<elem::s1>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s2, elem::s3> hi() const {
    return this->swizzle<elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s2, elem::s3> hi() const {
    return this->swizzle<elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s4, elem::s5, elem::s6, elem::s7> hi() const {
    return this->swizzle<elem::s4, elem::s5, elem::s6, elem::s7>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 16)>
  swizzled_vec<dataT, 16, elem::s8, elem::s9, elem::sA, elem::sB, elem::sC,
               elem::sD, elem::sE, elem::sF>
  hi() const {
    return this->swizzle<elem::s8, elem::s9, elem::sA, elem::sB, elem::sC,
                         elem::sD, elem::sE, elem::sF>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s0> lo() const {
    return this->swizzle<elem::s0>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s0, elem::s1> lo() const {
    return this->swizzle<elem::s0, elem::s1>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s0, elem::s1> lo() const {
    return this->swizzle<elem::s0, elem::s1>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s0, elem::s1, elem::s2, elem::s3> lo() const {
    return this->swizzle<elem::s0, elem::s1, elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 16)>
  swizzled_vec<dataT, 16, elem::s0, elem::s1, elem::s2, elem::s3, elem::s4,
               elem::s5, elem::s6, elem::s7>
  lo() const {
    return this->swizzle<elem::s0, elem::s1, elem::s2, elem::s3, elem::s4,
                         elem::s5, elem::s6, elem::s7>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s1> odd() const {
    return this->swizzle<elem::s1>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s1, elem::s3> odd() const {
    return this->swizzle<elem::s1, elem::s3>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s1, elem::s3> odd() const {
    return this->swizzle<elem::s1, elem::s3>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s1, elem::s3, elem::s5, elem::s7> odd() const {
    return this->swizzle<elem::s1, elem::s3, elem::s5, elem::s7>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 16)>
  swizzled_vec<dataT, 16, elem::s1, elem::s3, elem::s5, elem::s7, elem::s9,
               elem::sB, elem::sD, elem::sF>
  odd() const {
    return this->swizzle<elem::s1, elem::s3, elem::s5, elem::s7, elem::s9,
                         elem::sB, elem::sD, elem::sF>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s0> even() const {
    return this->swizzle<elem::s0>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s0, elem::s2> even() const {
    return this->swizzle<elem::s0, elem::s2>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s0, elem::s2> even() const {
    return this->swizzle<elem::s0, elem::s2>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s0, elem::s2, elem::s4, elem::s6> even() const {
    return this->swizzle<elem::s0, elem::s2, elem::s4, elem::s6>();
  }

  /** @brief Returns a swizzle of this vec with the elements of the vectors
   * even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 16)>
  swizzled_vec<dataT, 16, elem::s0, elem::s2, elem::s4, elem::s6, elem::s8,
               elem::sA, elem::sC, elem::sE>
  even() const {
    return this->swizzle<elem::s0, elem::s2, elem::s4, elem::s6, elem::s8,
                         elem::sA, elem::sC, elem::sE>();
  }

  COMPUTECPP_CLANG_FORMAT_BARRIER

  /**
  @brief Load elements of a multi_ptr instance into this vec instance. This is
  performed as a sequential loop on the host device and using the OpenCL vloadn
  builtin functions on an OpenCL device.
  Responsible for non-const multi_ptr instances.
  @tparam addressSpace The address space of the multi_ptr to be loaded from.
  @param offset The offset in elements of this vec.
  @param ptr The multi_ptr to be loaded from.
  */
  template <access::address_space addressSpace>
  void load(
      size_t offset,
      multi_ptr<typename std::remove_const<dataT>::type, addressSpace> ptr) {
    this->load(offset, static_cast<multi_ptr<const dataT, addressSpace>>(ptr));
  }

  /**
  @brief Load elements of a raw pointer into this vec instance. This is
  performed as a sequential loop on the host device and using the OpenCL vloadn
  builtin functions on an OpenCL device.
  @tparam addressSpace The address space of the raw pointer to be loaded from.
  @tparam load_data_t Type of the raw pointer data, to be deduced.
  @param offset The offset in elements of this vec.
  @param ptr The multi_ptr to be loaded from.
  */
  template <access::address_space addressSpace, typename load_data_t>
  void load(size_t offset, const load_data_t* ptr) {
    this->load(offset, multi_ptr<const dataT, addressSpace>(ptr));
  }

  /**
  @brief Load elements of a multi_ptr instance into this vec instance. This is
  performed as a sequential loop on the host device and using the OpenCL vloadn
  builtin functions on an OpenCL device. The definition of this member function
  template is in vec_load_store.h.
  @tparam addressSpace The address space of the multi_ptr to be loaded from.
  @param offset The offset in elements of this vec.
  @param ptr The multi_ptr to be loaded from.
  */
  template <access::address_space addressSpace>
  void load(size_t offset, multi_ptr<const dataT, addressSpace> ptr);

  /**
  @brief Load elements of a the multi_ptr retrieved from an accessor instance
  into this vec instance. This is performed as a sequential loop on the host
  device and using the OpenCL vloadn builtin functions on an OpenCL device. The
  definition of this member function template is in vec_load_store.h.
  @tparam kDims The dimensionality of the accessor to be loaded from.
  @tparam accessMode The access mode of the accessor to be loaded from.
  @tparam accessTarget The access target of the accessor to be loaded from.
  @param offset The offset in elements of this vec.
  @param acc The accessor to be loaded from.
  */
  template <int kDims, access::mode accessMode, access::target accessTarget>
  void load(size_t offset,
            accessor<dataT, kDims, accessMode, accessTarget> acc);

  /**
  @brief Store elements of this vec instance into a multi_ptr instance. This is
  performed as a sequential loop on the host device and using the OpenCL vstoren
  builtin functions on an OpenCL device. The definition of this member function
  template is in vec_load_store.h.
  @tparam addressSpace The address space of the multi_ptr to be stored to.
  @param offset The offset in elements of this vec.
  @param ptr The multi_ptr to be stored to.
  */
  template <access::address_space addressSpace>
  void store(size_t offset, multi_ptr<dataT, addressSpace> ptr) const;

  /**
  @brief Store elements of this vec instance into the multi_ptr from an accessor
  instance. This is performed as a sequential loop on the host device and using
  the OpenCL vstoren builtin functions on an OpenCL device. The definition of
  this member function template is in vec_load_store.h.
  @tparam kDims The dimensionality of the accessor to be stored to.
  @tparam accessMode The access mode of the accessor to be stored to.
  @tparam accessTarget The access target of the accessor to be stored to.
  @param offset The offset in elements of this vec.
  @param ptr The accessor to be stored to.
  */
  template <int kDims, access::mode accessMode, access::target accessTarget>
  void store(size_t offset,
             accessor<dataT, kDims, accessMode, accessTarget> acc) const;

  /** Creates a swizzled_vec object by performing a reinterperet_cast on this
   * pointer using the swizzle indexes.
   * @tparam kIndexesN The variadic argument pack for swizzle indexes.
   * @return Returns the result of the reinterpret_cast.
   */
  template <int... kIndexesN>
  const swizzled_vec<dataT, kElems, kIndexesN...>& swizzle() const;

  /** Creates a swizzled_vec object by performing a reinterperet_cast on this
    pointer using the swizzle indexes.
  * @tparam kIndexesN The variadic argument pack for swizzle indexes.
  * @return Returns the result of the reinterpret_cast.
  */
  template <int... kIndexesN>
  swizzled_vec<dataT, kElems, kIndexesN...>& swizzle();

  /** Returns the number of elements of the vector
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const;

  /** Returns the size of the vector
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const;

#if SYCL_LANGUAGE_VERSION >= 202001
  /** Returns the number of elements of the vector
   */
  size_t size() const noexcept;

  /** Returns the size of the vector
   */
  size_t byte_size() const noexcept;
#endif  // SYCL_LANGUAGE_VERSION >= 202001

 protected:
  /** Recursive method for adding a combination of vec object arguments and
   * scalar arguments to the vec constructor. Overload taking a vec object.
   * Assigns each element of the vec object's m_data member. Performs a
   * static_assert to ensure the correct number of arguments are given. This is
   * the main recursive function for vec objects.
   * @tparam kIndex The current index being assigned to.
   * @tparam kArgDim The size of next argument to be added to the vector.
   * @tparam kIndexesN The variadic argument pack for rest of arguments.
   * param arg The vec object to be assigned to vec object.
   * @param args Rest of arguments.
   */
  template <int kIndex, int kArgDim, typename... kIndexesN>
  void add_arg(vec<dataT, kArgDim> arg, kIndexesN... args);

  /** Recursive method for adding a combination of vec object arguments and
   * scalar arguments to the vec constructor. Overload taking a vec object.
   * Assigns each element of the vec object's m_data member. Performs a
   * static_assert to ensure the correct number of arguments are given. This is
   * the main recursive function for scalars.
   * @tparam kIndex The current index being assigned to.
   * @tparam kIndexesN The variadic argument pack for rest of arguments.
   * param arg The scalar value to be assigned to vec object.
   * @param args Rest of arguments.
   */
  template <int kIndex, typename... kIndexesN>
  void add_arg(dataT arg, kIndexesN... args);

  /** Recursive method for adding a combination of vec object arguments and
   * scalar arguments to the vec constructor. Overload taking a vec object.
   * Assigns each element of the vec object's m_data member. Performs a
   * static_assert to ensure the correct number of arguments are given. This is
   * the final recursive which simply has a static_asset.
   * @tparam kIndex The current index being assigned to.
   */
  template <int kIndex>
  void add_arg();
};  // class vec

#if SYCL_LANGUAGE_VERSION >= 202001

/** Deduction guide for vec class template.
 */
template <typename dataT, typename... argsN>
vec(dataT, argsN...)->vec<dataT, sizeof...(argsN) + 1>;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

/** @brief Creates a new vector from performing rhs + scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the + operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator+(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs + scalar;
}

/** @brief Creates a new vector from performing rhs * scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the * operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator*(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs * scalar;
}

/** @brief Creates a new vector from performing rhs & scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the & operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator&(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs & scalar;
}

/** @brief Creates a new vector from performing rhs | scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the | operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator|(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs | scalar;
}

/** @brief Creates a new vector from performing rhs ^ scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the ^ operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator^(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs ^ scalar;
}

/** @brief Creates a new vector from performing rhs - scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the - operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator-(dataT scalar, const vec<dataT, kElems>& rhs) {
  vec<dataT, kElems> t(scalar);
  return t - rhs;
}

/** @brief Creates a new vector from performing rhs / scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the / operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator/(dataT scalar, const vec<dataT, kElems>& rhs) {
  vec<dataT, kElems> t(scalar);
  return t / rhs;
}

/** @brief Creates a new vector from performing rhs % scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the % operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator%(dataT scalar, const vec<dataT, kElems>& rhs) {
  vec<dataT, kElems> t(scalar);
  return t % rhs;
}

/** @brief Creates a new vector from performing rhs << scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the << operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator<<(dataT scalar, const vec<dataT, kElems>& rhs) {
  vec<dataT, kElems> t(scalar);
  return t << rhs;
}

/** @brief Creates a new vector from performing rhs >> scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The result of the >> operation
 */
template <typename dataT, int kElems>
vec<dataT, kElems> operator>>(dataT scalar, const vec<dataT, kElems>& rhs) {
  vec<dataT, kElems> t(scalar);
  return t >> rhs;
}

/** @brief Creates a new vector from performing rhs && scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the && operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator&&(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs && scalar;
}

/** @brief Creates a new vector from performing rhs || scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the || operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator||(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs || scalar;
}

/** @brief Creates a new vector from performing rhs == scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the == operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator==(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs == scalar;
}

/** @brief Creates a new vector from performing rhs != scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the != operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator!=(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs != scalar;
}

#ifdef __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs > scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the > operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator<(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs > scalar;
}

#else  // __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs > scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the > operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator<(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs > scalar;
}

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs < scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the < operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator>(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs < scalar;
}

#else  // __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs < scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the < operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator>(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs < scalar;
}

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs >= scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the >= operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator<=(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs >= scalar;
}

#else  // __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs >= scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the >= operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator<=(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs >= scalar;
}

#endif  // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs <= scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the <= operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator>=(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs <= scalar;
}

#else  // __SYCL_DEVICE_ONLY__

/** @brief Creates a new vector from performing rhs <= scalar
 * @param scalar The scalar value.
 * @param rhs The rhs vec reference.
 * @return The return of the <= operation
 */
template <typename dataT, int kElems>
vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, kElems>
operator>=(dataT scalar, const vec<dataT, kElems>& rhs) {
  return rhs <= scalar;
}

#endif  // __SYCL_DEVICE_ONLY__

COMPUTECPP_CLANG_FORMAT_BARRIER

namespace detail {

/** @brief Overload for deducing vec<T, N>.
 * @see deduce_type_impl_f.
 */
template <typename T, int N>
vec<deduce_type_t<T>, N> deduce_type_impl_f(vec<T, N>);

/** @brief Overload for deducing vec<T, 1>.
 * @see deduce_type_impl_f.
 */
template <typename T>
deduce_type_t<T> deduce_type_impl_f(vec<T, 1>);

}  // namespace detail

}  // namespace sycl
}  // namespace cl

#undef COMPUTECPP_VEC_OP_BODY

#undef COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_IMPL_H_

////////////////////////////////////////////////////////////////////////////////
