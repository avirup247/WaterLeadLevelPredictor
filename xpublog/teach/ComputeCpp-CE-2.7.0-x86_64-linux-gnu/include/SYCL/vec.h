/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec.h

 @brief This file contains the @ref cl::sycl::vec class definition as defined by
 the SYCL 1.2.1 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_H_
#define RUNTIME_INCLUDE_SYCL_VEC_H_

#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/common.h"

#include "SYCL/vec_common.h"    // IWYU pragma: export
#include "SYCL/vec_impl.h"      // IWYU pragma: export
#include "SYCL/vec_macros.h"    // IWYU pragma: export
#include "SYCL/vec_swizzles.h"  // IWYU pragma: export

#include <cstring>

////////////////////////////////////////////////////////////////////////////////

#ifndef __SYCL_DEVICE_ONLY__

#include "abacus/abacus_config"

#endif  // __SYCL_DEVICE_ONLY__

/* Vec
 ***
 */

/* Overview
   ========
   swizzles are implemented across the following files:
   vec_impl.h contains the vec class definition and implementation of many of
   the sycl::vec methods.
   vec_macros.h contains all macros that are used for the simple swizzle methods
   and for the swizzle apply functions.
   vec_common.h contains common definitions for al vec classes and swizzles.
   vec_swizzles.h contains the swizzle apply functions for host and device.
   vec_swizzles_impl.h contains the swizzle class definition and operators.
   vec.h contains the vec and swizzled_vec classes and their base classes.
*/

/* Vec
   ===
   The vec and swizzled_vec class share the common base class
   mem_container_storage, this is the only class in the inheritance that
   contains members. On the host side the data for the vector is defined by a
   standard C++ array, on the device side the vector is defined by the
   ext_vector_type clang attribute which represents an OpenCL compatible
   vector.
*/

/* Swizzles
   ========
   Swizzles are implemented using variadic template arguments to represent the
   swizzle indexes. The way this works is that the there is a vec class and a
   swizzled_vec class, they both share the same base classes which define the
   data storage, but have different functionality for constructors, conversions
   and operators. However the swizzled_vec class does not represent the new
   object, it simply represents the indexing that will be used when a swizzle
   is performed. The reason for this is that the result of a swizzle method can
   be used in many different contexts such as constructors, parameters,
   assignments and other operators, some some cases it isn't used at all, and
   this effects how the swizzle indexes are used. For example if you have a
   swizzled_vec that is assigned to a vec object, the operation is different
   from it you were assigning it to another swizzled_vec object. Additionally
   if you choose not to anything at all with the swizzled_vec, there should be
   no change to the original vec object. The actual swizzle operations are
   applied different for host and for device; for host they are applied using
   recursive variadic functions and for device they are applied using a
   template specialization which calls the underlying OpenCL built-in
   operations. Swizzles can invoked in two ways, either by calling the
   template function swizzle which takes a variadic pack of arguments specifying
   the swizzle indexes, this can allow up to 16 element swizzles. They can also
   be invoked by calling the simple swizzle methods, which implement OpenCL
   built-in swizzle operations underneath, however they only support four
   element swizzles. There are some places where the implementation for host
   and device vary, for these respective macros are used to separate the
   definitions.

   Example of OpenCL C swizzles:

   float4 foo = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
   float4 bar = foo.wzyx;

   Example of SYCL swizzles:

   float4 foo(1.0f, 2.0f, 3.0f, 4.0f);
   float4 bar = foo.wzyx();

   For more details about swizzles see section 6.1.7 of the OpenCL
   specification.
*/

/* Testing
   ========
   Current regression tests for vec and swizzles are:
   * vec_vector_add
   * vec_operators
   * vec_swizzles
   The vec class and swizzle methods can be tested further using the test
   suite.
*/

/* Current Status
   ==============
   Features that are currently not implemented include:
   .odd() and .even() need implemeneted for host and device
   swizzles need full implementation
*/

namespace cl {
namespace sycl {

namespace detail {

#ifndef __SYCL_DEVICE_ONLY__

template <typename dataT, int kElems>
inline const dataT* mem_container_storage<dataT, kElems>::get_data() const {
  return m_data;
}

template <typename dataT, int kElems>
inline dataT* mem_container_storage<dataT, kElems>::get_data() {
  return m_data;
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_data(
    const vec<dataT, kElems>& rhs) {
  for (int i = 0; i < kElems; i++) {
    m_data[i] = rhs.get_value(i);
  }
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(int index) const {
  return this->get_value(index, std::true_type{});
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(
    int index, std::true_type) const {
  return m_data[index];
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(
    int /*index*/, std::false_type) const {
  return static_cast<dataT>(0);
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(
    int index, const dataT& value) {
  this->set_value(index, value, std::true_type{});
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(int index,
                                                            const dataT& value,
                                                            std::true_type) {
  m_data[index] = value;
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(
    int /*index*/, const dataT& /*value*/, std::false_type) {}

#else

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_data(
    const vec<dataT, kElems>& rhs) {
  m_data = rhs.m_data;
}

template <typename dataT, int kElems>
inline detail::__sycl_vector<dataT, kElems>
mem_container_storage<dataT, kElems>::get_data() const {
  return m_data;
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_data(
    detail::__sycl_vector<dataT, kElems> rhs) {
  m_data = rhs;
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(int index) const {
  return this->get_value(index, std::true_type{});
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(
    int index, std::true_type) const {
  return m_data[index];
}

template <typename dataT, int kElems>
inline dataT mem_container_storage<dataT, kElems>::get_value(
    int /*index*/, std::false_type) const {
  return static_cast<dataT>(0);
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(
    int index, const dataT& value) {
  this->set_value(index, value, std::true_type{});
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(int index,
                                                            const dataT& value,
                                                            std::true_type) {
  m_data[index] = value;
}

template <typename dataT, int kElems>
inline void mem_container_storage<dataT, kElems>::set_value(
    int /*index*/, const dataT& /*value*/, std::false_type) {}

/** @brief sets the internal vector value to the interal value of rhs
 * @param rhs The vector to set a value from
 */
template <typename dataT>
inline void mem_container_storage<dataT, 1>::set_data(
    const vec<dataT, 1>& rhs) {
  m_data.x = rhs.m_data.x;
}

/** @brief returns the interal vector value of this vector
 * @return the value stored by this vector
 */
template <typename dataT>
inline dataT mem_container_storage<dataT, 1>::get_data() const {
  return m_data.x;
}

/** @brief sets the internal vector value to the value of rhs
 * @param rhs The value to set this vector to
 */
template <typename dataT>
inline void mem_container_storage<dataT, 1>::set_data(dataT rhs) {
  m_data.x = rhs;
}

/** @brief returns the vector value at specified index
 * @param index The index of the value to retrieve
 * @return the value stored by this vector at the specified index
 */
template <typename dataT>
inline dataT mem_container_storage<dataT, 1>::get_value(int index) const {
  return this->get_value(index, std::true_type{});
}

template <typename dataT>
inline dataT mem_container_storage<dataT, 1>::get_value(int index,
                                                        std::true_type) const {
  (void)index;
  return m_data.x;
}

template <typename dataT>
inline dataT mem_container_storage<dataT, 1>::get_value(int index,
                                                        std::false_type) const {
  (void)index;
  return static_cast<dataT>(0);
}

/** @brief sets the value of the vector at the specified index to the input
 * value
 * @param index The index of the value to set
 * @param value The value to set
 */
template <typename dataT>
inline void mem_container_storage<dataT, 1>::set_value(int index,
                                                       const dataT& value) {
  this->set_value(index, value, std::true_type{});
}

template <typename dataT>
inline void mem_container_storage<dataT, 1>::set_value(int index,
                                                       const dataT& value,
                                                       std::true_type) {
  (void)index;
  m_data.x = value;
}

template <typename dataT>
inline void mem_container_storage<dataT, 1>::set_value(int index,
                                                       const dataT& value,
                                                       std::false_type) {
  (void)index;
  (void)value;
}

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail

template <typename dataT, int kElems>
vec<dataT, kElems>::vec() {
  for (int i = 0; i < kElems; i++) {
    this->set_value(i, static_cast<dataT>(0));
  }
}

template <typename dataT, int kElems>
template <int kElemsRhs, int... kIndexRhsN, typename E>
vec<dataT, kElems>::vec(
    const swizzled_vec<dataT, kElemsRhs, kIndexRhsN...>& rhs) {
  vec<dataT, kElems> newVec =
      detail::swizzle_rhs<dataT, kElems, kElemsRhs, kIndexRhsN...>::apply(rhs);
  this->set_data(newVec);
}

template <typename dataT, int kElems>
template <typename recurseT0, typename recurseT1, typename... recurseTN>
vec<dataT, kElems>::vec(recurseT0 arg0, recurseT1 arg1, recurseTN... args) {
  add_arg<0>(arg0, arg1, args...);
}

#ifdef __SYCL_DEVICE_ONLY__
template <typename dataT, int kElems>
vec<dataT, kElems>::operator detail::__sycl_vector<dataT, kElems>() {
  return this->get_data();
}

template <typename dataT, int kElems>
vec<dataT, kElems>::operator detail::__sycl_vector<dataT, kElems>() const {
  return this->get_data();
}

#endif  // __SYCL_DEVICE_ONLY__

#ifndef __SYCL_DEVICE_ONLY__

template <typename dataT, int kElems>
template <typename abacusT>
vec<dataT, kElems>::vec(const abacus_vector<abacusT, kElems>& rhs) {
  for (int i = 0; i < kElems; i++) {
    this->set_value(i, rhs[i]);
  }
}

// NOLINTNEXTLINE(misc-unconventional-assign-operator)
template <typename dataT, int kElems>
template <typename abacusT>
vec<abacusT, kElems>& vec<dataT, kElems>::operator=(
    const abacus_vector<abacusT, kElems>& rhs) {
  for (int i = 0; i < kElems; i++) {
    this->set_value(i, rhs[i]);
  }
  return *this;
}

#endif  // __SYCL_DEVICE_ONLY__

template <typename dataT, int kElems>
vec<dataT, kElems>& vec<dataT, kElems>::operator=(dataT rhs) {
#ifdef __SYCL_DEVICE_ONLY__
  this->set_data(rhs);
#else
  for (int i = 0; i < kElems; i++) {
    this->set_value(i, rhs);
  }
#endif  // __SYCL_DEVICE_ONLY__
  return *this;
}

template <typename dataT, int kElems>
template <typename convertT, rounding_mode roundingMode, class>
vec<convertT, kElems> vec<dataT, kElems>::convert() const {
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
  vec<convertT, kElems> newVec;
  for (int i = 0; i < kElems; ++i) {
    newVec.set_value(i, static_cast<convertT>(this->get_value(i)));
  }
  return newVec;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
}

template <typename dataT, int kElems>
template <typename asT, class>
asT vec<dataT, kElems>::as() const {
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
  asT newVec;
  std::memcpy(&newVec, this, sizeof(asT));
  return newVec;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
}

template <typename dataT, int kElems>
vec<dataT, kElems>& vec<dataT, kElems>::operator++() {
  (*this) += 1;
  return *this;
}

template <typename dataT, int kElems>
vec<dataT, kElems> vec<dataT, kElems>::operator++(int) {
  vec<dataT, kElems> save = *this;
  (*this) += 1;
  return save;
}

template <typename dataT, int kElems>
vec<dataT, kElems>& vec<dataT, kElems>::operator--() {
  (*this) -= 1;
  return *this;
}

template <typename dataT, int kElems>
vec<dataT, kElems> vec<dataT, kElems>::operator--(int) {
  vec<dataT, kElems> save = *this;
  (*this) -= 1;
  return save;
}

template <typename dataT, int kElems>
vec<dataT, kElems> vec<dataT, kElems>::operator-() const {
#ifdef __SYCL_DEVICE_ONLY__
  return vec<dataT, kElems>(-(this->get_data()));
#else
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
  COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
  vec<dataT, kElems> result;
  for (int i = 0; i < kElems; i++) {
    result.set_value(i, -(this->get_value(i)));
  }
  return result;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
#endif  // __SYCL_DEVICE_ONLY__
}

template <typename dataT, int kElems>
vec<dataT, kElems> vec<dataT, kElems>::operator~() const {
#ifdef __SYCL_DEVICE_ONLY__
  return vec<dataT, kElems>(~(this->m_data));
#else
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_GNU_CXX_DIAGNOSTIC(ignored "-Wuninitialized")
  COMPUTECPP_GCC_CXX_DIAGNOSTIC(ignored "-Wmaybe-uninitialized")
  vec<dataT, kElems> result;
  for (int i = 0; i < kElems; i++) {
    result.set_value(i, ~(this->get_value(i)));
  }
  return result;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
#endif  // __SYCL_DEVICE_ONLY__
}

template <typename dataT, int kElems>
template <int... kIndexesN>
const swizzled_vec<dataT, kElems, kIndexesN...>& vec<dataT, kElems>::swizzle()
    const {
  return (*(
      reinterpret_cast<const swizzled_vec<dataT, kElems, kIndexesN...>* const>(
          this)));
}

template <typename dataT, int kElems>
template <int... kIndexesN>
swizzled_vec<dataT, kElems, kIndexesN...>& vec<dataT, kElems>::swizzle() {
  return (
      *(reinterpret_cast<swizzled_vec<dataT, kElems, kIndexesN...>*>(this)));
}

template <typename dataT, int kElems>
size_t vec<dataT, kElems>::get_count() const {
  return kElems;
}

template <typename dataT, int kElems>
size_t vec<dataT, kElems>::get_size() const {
  return sizeof(vec<dataT, kElems>);
}

template <typename dataT, int kElems>
template <int kIndex, int kArgDim, typename... kIndexesN>
void vec<dataT, kElems>::add_arg(vec<dataT, kArgDim> arg, kIndexesN... args) {
  static_assert(kIndex + (kArgDim - 1) < kElems,
                "Error: Invalid number of constructor arguments.");
  for (int i = 0; i < kArgDim; i++) {
    this->set_value(kIndex + i, arg.get_value(i));
  }
  add_arg<kIndex + kArgDim>(args...);
}

template <typename dataT, int kElems>
template <int kIndex, typename... kIndexesN>
void vec<dataT, kElems>::add_arg(dataT arg, kIndexesN... args) {
  static_assert(kIndex < kElems,
                "Error: Invalid number of constructor arguments.");
  this->set_value(kIndex, arg);
  add_arg<kIndex + 1>(args...);
}

template <typename dataT, int kElems>
template <int kIndex>
void vec<dataT, kElems>::add_arg() {
  static_assert(kIndex == kElems,
                "Error: Invalid number of constructor arguments.");
}

}  // namespace sycl
}  // namespace cl

#undef COMPUTECPP_VEC_OP_BODY

#undef COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_H_

////////////////////////////////////////////////////////////////////////////////
