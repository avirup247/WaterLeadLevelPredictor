/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_swizzles_impl.h

 @brief This file contains internal tool to implement vector swizzle operations.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_IMPL_H_
#define RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_IMPL_H_

#include "SYCL/common.h"
#include "SYCL/deduce.h"
#include "SYCL/vec.h"

////////////////////////////////////////////////////////////////////////////////

namespace cl {

namespace sycl {

/** @cond COMPUTECPP_DEV */

/* Overview
   ========
   This file implements the swizzle class and associated operators.
   See vec.h for a full explanation on the vector implementation.
*/

namespace detail {

/** Type trait to select the return type of a swizzle operation
 * It can be either vec<dataT, N> if N != 1 or dataT if N == 1
 */
template <typename dataT, unsigned width>
struct swizzle_return_ty {
  using Type = vec<dataT, width>;
  swizzle_return_ty(const Type& v) : m_v(v) {}
  swizzle_return_ty(const dataT& v) : m_v(v) {}
  operator Type() { return m_v; }

  Type m_v;
};

template <typename dataT>
struct swizzle_return_ty<dataT, 1> {
  using Type = dataT;
  swizzle_return_ty(const Type& v) : m_v(v) {}
  // swizzle_return_ty(vec<dataT, 1> v) : m_v(v.x()) {}
  operator Type() { return m_v; }

  Type m_v;
};

/** @brief Intermediate class for specializing operators that need to be
 * implemented differently for swizzle methods and single element access
 * methods. The original class is used for 2, 3 and 4 dimensional swizzled_vec
 * objects.
 * @tparam dataT The data type for the vector.
 * @tparam kElems The number of elements for the vector.
 * @tparam kIndexesN Variadic argument pack for swizzle indexes.
 */
template <typename dataT, int kElems, int... kIndexesN>
class swizzled_vec_intermediate : public mem_container_base<dataT, kElems> {
 public:
};

/** @brief Intermediate class for specializing operators that need to be
 * implemented differently for swizzle methods and single element access
 * methods. This specialized class is used for 1 dimensional swizzled_vec
 * objects.
 * @tparam dataT The data type for the vector.
 * @tparam kElems The number of elements for the vector.
 * @tparam kIndexesN Variadic argument pack for swizzle indexes.
 */
#ifndef __SYCL_DEVICE_ONLY__
template <typename dataT, int kElems, int kIndexesN>
class swizzled_vec_intermediate<dataT, kElems, kIndexesN>
    : public mem_container_base<dataT, kElems> {
 public:
  /** Returns the value of the m_data field at the first (and only) swizzle
   * index.
   * @tparam kElemsRes The number of elements for the resultant vec object.
   * @return The scalar value from the swizzle index.
   */
  operator dataT() { return this->m_data[kIndexesN]; }

  operator dataT() const { return this->m_data[kIndexesN]; }
};

#endif
}  // namespace detail

/** COMPUTECPP_DEV @endcond */

/** @brief Vector class representing a swizzled vector for host and device.
  Has additional variadic template argument pack to represent the swizzle
  indexes.
* @tparam dataT The data type for the vector.
* @tparam kElems The number of elements for the vector.
* @tparam kIndexesN The variadic argument pack for the swizzle indexes.
*/
template <typename dataT, int kElems, int... kIndexesN>
class swizzled_vec
    : public detail::swizzled_vec_intermediate<dataT, kElems, kIndexesN...> {
 public:
  /** @brief Alias for the variadic swizzle index pack
   */
  using swizzle_pack_t = detail::swizzle_pack<kIndexesN...>;

  /** @brief Retrieves the requested index from the variadic swizzle index pack
   * @param indexPos Position of the index in the swizzle index pack
   * @return Index located at the requested position in the swizzle index pack
   */
  static constexpr int get_index(int indexPos) {
    return swizzle_pack_t::get(indexPos);
  }

  /** @brief Returns the number of elements in the vector resulting from the
   * swizzle operation.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return sizeof...(kIndexesN); }

  /**
   * @brief Returns the size of the vector resulting from the swizzle operation
   * in bytes.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(vec<dataT, kElems>); }

#if SYCL_LANGUAGE_VERSION >= 202001
  /**
   * @brief Returns the number of elements in the vector resulting from the
   * swizzle operation.
   */
  size_t size() const noexcept { return sizeof...(kIndexesN); }

  /**
   * @brief Returns the size of the vector resulting from the swizzle operation
   * in bytes
   */
  size_t byte_size() const noexcept { return sizeof(vec<dataT, kElems>); }
#endif  // SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, sizeof...(kIndexesN)> convert() const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec.template as<asT>();
  }

  /** Initialises each vector value with the respective vector value of the rhs
   * swizzled_vec.
   * @param rhs The reference to the swizzled_vec to be assigned.
   */
  swizzled_vec(const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    this->m_data = rhs.m_data;
#else
    for (int i = 0; i < kElems; i++) {
      this->m_data[i] = rhs.m_data[i];
    }
#endif
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s1> hi() const {
    return this->swizzle<elem::s1>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s2, elem::s3> hi() const {
    return this->swizzle<elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s2, elem::s3> hi() const {
    return this->swizzle<elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors hi operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s4, elem::s5, elem::s6, elem::s7> hi() const {
    return this->swizzle<elem::s4, elem::s5, elem::s6, elem::s7>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors hi operation
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

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s0> lo() const {
    return this->swizzle<elem::s0>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s0, elem::s1> lo() const {
    return this->swizzle<elem::s0, elem::s1>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s0, elem::s1> lo() const {
    return this->swizzle<elem::s0, elem::s1>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors lo operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s0, elem::s1, elem::s2, elem::s3> lo() const {
    return this->swizzle<elem::s0, elem::s1, elem::s2, elem::s3>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors lo operation
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

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s1> odd() const {
    return this->swizzle<elem::s1>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s1, elem::s3> odd() const {
    return this->swizzle<elem::s1, elem::s3>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s1, elem::s3> odd() const {
    return this->swizzle<elem::s1, elem::s3>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors odd operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s1, elem::s3, elem::s5, elem::s7> odd() const {
    return this->swizzle<elem::s1, elem::s3, elem::s5, elem::s7>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors odd operation
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

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 2)>
  swizzled_vec<dataT, 2, elem::s0> even() const {
    return this->swizzle<elem::s0>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 3)>
  swizzled_vec<dataT, 3, elem::s0, elem::s2> even() const {
    return this->swizzle<elem::s0, elem::s2>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 4)>
  swizzled_vec<dataT, 4, elem::s0, elem::s2> even() const {
    return this->swizzle<elem::s0, elem::s2>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors even operation
   * @tparam COMPUTECPP_ENABLE_IF Only enabled for ${size} element vectors
   * condition.
   * @return The returned swizzle
   */
  template <COMPUTECPP_ENABLE_IF(dataT, kElems == 8)>
  swizzled_vec<dataT, 8, elem::s0, elem::s2, elem::s4, elem::s6> even() const {
    return this->swizzle<elem::s0, elem::s2, elem::s4, elem::s6>();
  }

  /** @brief Returns a swizzle of this swizzled_vec with the elements of the
   * vectors even operation
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

#ifdef SYCL_SIMPLE_SWIZZLES

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0)>>
  return_t& x() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0)>>
  const return_t& x() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1)>>
  return_t& y() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1)>>
  const return_t& y() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2)>>
  return_t& z() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2)>>
  const return_t& z() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3)>>
  return_t& w() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3)>>
  const return_t& w() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0)>>
  return_t& xx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0)>>
  const return_t& xx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1)>>
  return_t& xy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1)>>
  const return_t& xy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2)>>
  return_t& xz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2)>>
  const return_t& xz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3)>>
  return_t& xw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3)>>
  const return_t& xw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0)>>
  return_t& yx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0)>>
  const return_t& yx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1)>>
  return_t& yy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1)>>
  const return_t& yy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2)>>
  return_t& yz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2)>>
  const return_t& yz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3)>>
  return_t& yw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3)>>
  const return_t& yw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0)>>
  return_t& zx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0)>>
  const return_t& zx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1)>>
  return_t& zy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1)>>
  const return_t& zy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2)>>
  return_t& zz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2)>>
  const return_t& zz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3)>>
  return_t& zw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3)>>
  const return_t& zw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0)>>
  return_t& wx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0)>>
  const return_t& wx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1)>>
  return_t& wy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1)>>
  const return_t& wy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2)>>
  return_t& wz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2)>>
  const return_t& wz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3)>>
  return_t& ww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3)>>
  const return_t& ww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(0)>>
  return_t& xxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(0)>>
  const return_t& xxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(1)>>
  return_t& xxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(1)>>
  const return_t& xxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(2)>>
  return_t& xxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(2)>>
  const return_t& xxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(3)>>
  return_t& xxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(0), get_index(3)>>
  const return_t& xxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(0)>>
  return_t& xyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(0)>>
  const return_t& xyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(1)>>
  return_t& xyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(1)>>
  const return_t& xyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(2)>>
  return_t& xyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(2)>>
  const return_t& xyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(3)>>
  return_t& xyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(1), get_index(3)>>
  const return_t& xyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(0)>>
  return_t& xzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(0)>>
  const return_t& xzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(1)>>
  return_t& xzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(1)>>
  const return_t& xzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(2)>>
  return_t& xzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(2)>>
  const return_t& xzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(3)>>
  return_t& xzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(2), get_index(3)>>
  const return_t& xzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(0)>>
  return_t& xwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(0)>>
  const return_t& xwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(1)>>
  return_t& xwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(1)>>
  const return_t& xwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(2)>>
  return_t& xwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(2)>>
  const return_t& xwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(3)>>
  return_t& xww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0),
                                          get_index(3), get_index(3)>>
  const return_t& xww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(0)>>
  return_t& yxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(0)>>
  const return_t& yxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(1)>>
  return_t& yxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(1)>>
  const return_t& yxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(2)>>
  return_t& yxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(2)>>
  const return_t& yxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(3)>>
  return_t& yxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(0), get_index(3)>>
  const return_t& yxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(0)>>
  return_t& yyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(0)>>
  const return_t& yyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(1)>>
  return_t& yyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(1)>>
  const return_t& yyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(2)>>
  return_t& yyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(2)>>
  const return_t& yyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(3)>>
  return_t& yyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(1), get_index(3)>>
  const return_t& yyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(0)>>
  return_t& yzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(0)>>
  const return_t& yzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(1)>>
  return_t& yzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(1)>>
  const return_t& yzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(2)>>
  return_t& yzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(2)>>
  const return_t& yzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(3)>>
  return_t& yzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(2), get_index(3)>>
  const return_t& yzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(0)>>
  return_t& ywx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(0)>>
  const return_t& ywx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(1)>>
  return_t& ywy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(1)>>
  const return_t& ywy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(2)>>
  return_t& ywz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(2)>>
  const return_t& ywz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(3)>>
  return_t& yww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1),
                                          get_index(3), get_index(3)>>
  const return_t& yww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(0)>>
  return_t& zxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(0)>>
  const return_t& zxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(1)>>
  return_t& zxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(1)>>
  const return_t& zxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(2)>>
  return_t& zxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(2)>>
  const return_t& zxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(3)>>
  return_t& zxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(0), get_index(3)>>
  const return_t& zxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(0)>>
  return_t& zyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(0)>>
  const return_t& zyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(1)>>
  return_t& zyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(1)>>
  const return_t& zyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(2)>>
  return_t& zyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(2)>>
  const return_t& zyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(3)>>
  return_t& zyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(1), get_index(3)>>
  const return_t& zyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(0)>>
  return_t& zzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(0)>>
  const return_t& zzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(1)>>
  return_t& zzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(1)>>
  const return_t& zzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(2)>>
  return_t& zzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(2)>>
  const return_t& zzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(3)>>
  return_t& zzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(2), get_index(3)>>
  const return_t& zzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(0)>>
  return_t& zwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(0)>>
  const return_t& zwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(1)>>
  return_t& zwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(1)>>
  const return_t& zwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(2)>>
  return_t& zwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(2)>>
  const return_t& zwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(3)>>
  return_t& zww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2),
                                          get_index(3), get_index(3)>>
  const return_t& zww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(0)>>
  return_t& wxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(0)>>
  const return_t& wxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(1)>>
  return_t& wxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(1)>>
  const return_t& wxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(2)>>
  return_t& wxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(2)>>
  const return_t& wxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(3)>>
  return_t& wxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(0), get_index(3)>>
  const return_t& wxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(0)>>
  return_t& wyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(0)>>
  const return_t& wyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(1)>>
  return_t& wyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(1)>>
  const return_t& wyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(2)>>
  return_t& wyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(2)>>
  const return_t& wyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(3)>>
  return_t& wyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(1), get_index(3)>>
  const return_t& wyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(0)>>
  return_t& wzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(0)>>
  const return_t& wzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(1)>>
  return_t& wzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(1)>>
  const return_t& wzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(2)>>
  return_t& wzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(2)>>
  const return_t& wzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(3)>>
  return_t& wzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(2), get_index(3)>>
  const return_t& wzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(0)>>
  return_t& wwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(0)>>
  const return_t& wwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(1)>>
  return_t& wwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(1)>>
  const return_t& wwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(2)>>
  return_t& wwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(2)>>
  const return_t& wwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(3)>>
  return_t& www() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3),
                                          get_index(3), get_index(3)>>
  const return_t& www() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& xxxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& xxxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& xxxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& xxxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& xxxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& xxxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& xxxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& xxxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& xxyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& xxyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& xxyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& xxyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& xxyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& xxyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& xxyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& xxyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& xxzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& xxzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& xxzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& xxzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& xxzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& xxzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& xxzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& xxzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& xxwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& xxwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& xxwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& xxwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& xxwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& xxwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& xxww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& xxww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& xyxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& xyxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& xyxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& xyxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& xyxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& xyxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& xyxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& xyxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& xyyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& xyyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& xyyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& xyyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& xyyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& xyyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& xyyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& xyyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& xyzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& xyzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& xyzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& xyzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& xyzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& xyzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& xyzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& xyzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& xywx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& xywx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& xywy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& xywy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& xywz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& xywz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& xyww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& xyww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& xzxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& xzxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& xzxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& xzxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& xzxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& xzxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& xzxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& xzxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& xzyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& xzyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& xzyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& xzyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& xzyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& xzyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& xzyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& xzyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& xzzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& xzzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& xzzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& xzzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& xzzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& xzzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& xzzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& xzzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& xzwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& xzwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& xzwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& xzwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& xzwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& xzwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& xzww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& xzww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& xwxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& xwxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& xwxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& xwxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& xwxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& xwxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& xwxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& xwxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& xwyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& xwyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& xwyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& xwyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& xwyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& xwyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& xwyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& xwyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& xwzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& xwzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& xwzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& xwzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& xwzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& xwzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& xwzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& xwzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& xwwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& xwwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& xwwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& xwwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& xwwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& xwwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& xwww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& xwww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& yxxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& yxxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& yxxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& yxxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& yxxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& yxxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& yxxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& yxxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& yxyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& yxyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& yxyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& yxyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& yxyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& yxyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& yxyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& yxyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& yxzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& yxzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& yxzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& yxzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& yxzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& yxzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& yxzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& yxzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& yxwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& yxwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& yxwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& yxwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& yxwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& yxwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& yxww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& yxww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& yyxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& yyxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& yyxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& yyxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& yyxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& yyxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& yyxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& yyxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& yyyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& yyyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& yyyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& yyyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& yyyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& yyyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& yyyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& yyyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& yyzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& yyzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& yyzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& yyzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& yyzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& yyzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& yyzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& yyzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& yywx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& yywx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& yywy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& yywy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& yywz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& yywz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& yyww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& yyww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& yzxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& yzxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& yzxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& yzxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& yzxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& yzxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& yzxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& yzxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& yzyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& yzyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& yzyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& yzyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& yzyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& yzyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& yzyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& yzyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& yzzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& yzzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& yzzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& yzzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& yzzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& yzzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& yzzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& yzzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& yzwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& yzwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& yzwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& yzwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& yzwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& yzwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& yzww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& yzww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& ywxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& ywxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& ywxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& ywxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& ywxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& ywxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& ywxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& ywxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& ywyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& ywyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& ywyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& ywyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& ywyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& ywyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& ywyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& ywyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& ywzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& ywzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& ywzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& ywzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& ywzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& ywzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& ywzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& ywzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& ywwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& ywwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& ywwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& ywwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& ywwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& ywwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& ywww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& ywww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& zxxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& zxxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& zxxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& zxxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& zxxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& zxxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& zxxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& zxxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& zxyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& zxyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& zxyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& zxyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& zxyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& zxyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& zxyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& zxyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& zxzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& zxzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& zxzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& zxzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& zxzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& zxzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& zxzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& zxzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& zxwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& zxwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& zxwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& zxwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& zxwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& zxwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& zxww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& zxww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& zyxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& zyxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& zyxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& zyxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& zyxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& zyxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& zyxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& zyxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& zyyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& zyyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& zyyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& zyyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& zyyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& zyyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& zyyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& zyyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& zyzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& zyzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& zyzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& zyzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& zyzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& zyzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& zyzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& zyzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& zywx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& zywx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& zywy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& zywy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& zywz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& zywz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& zyww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& zyww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& zzxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& zzxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& zzxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& zzxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& zzxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& zzxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& zzxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& zzxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& zzyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& zzyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& zzyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& zzyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& zzyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& zzyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& zzyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& zzyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& zzzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& zzzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& zzzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& zzzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& zzzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& zzzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& zzzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& zzzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& zzwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& zzwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& zzwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& zzwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& zzwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& zzwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& zzww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& zzww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& zwxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& zwxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& zwxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& zwxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& zwxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& zwxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& zwxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& zwxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& zwyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& zwyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& zwyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& zwyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& zwyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& zwyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& zwyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& zwyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& zwzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& zwzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& zwzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& zwzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& zwzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& zwzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& zwzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& zwzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& zwwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& zwwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& zwwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& zwwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& zwwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& zwwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& zwww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& zwww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& wxxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& wxxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& wxxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& wxxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& wxxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& wxxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& wxxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& wxxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& wxyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& wxyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& wxyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& wxyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& wxyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& wxyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& wxyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& wxyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& wxzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& wxzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& wxzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& wxzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& wxzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& wxzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& wxzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& wxzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& wxwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& wxwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& wxwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& wxwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& wxwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& wxwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& wxww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& wxww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& wyxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& wyxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& wyxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& wyxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& wyxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& wyxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& wyxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& wyxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& wyyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& wyyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& wyyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& wyyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& wyyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& wyyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& wyyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& wyyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& wyzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& wyzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& wyzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& wyzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& wyzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& wyzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& wyzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& wyzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& wywx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& wywx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& wywy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& wywy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& wywz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& wywz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& wyww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& wyww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& wzxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& wzxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& wzxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& wzxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& wzxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& wzxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& wzxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& wzxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& wzyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& wzyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& wzyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& wzyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& wzyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& wzyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& wzyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& wzyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& wzzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& wzzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& wzzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& wzzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& wzzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& wzzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& wzzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& wzzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& wzwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& wzwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& wzwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& wzwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& wzwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& wzwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& wzww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& wzww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& wwxx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& wwxx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& wwxy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& wwxy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& wwxz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& wwxz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& wwxw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& wwxw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& wwyx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& wwyx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& wwyy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& wwyy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& wwyz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& wwyz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& wwyw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& wwyw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& wwzx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& wwzx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& wwzy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& wwzy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& wwzz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& wwzz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& wwzw() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& wwzw() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& wwwx() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& wwwx() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& wwwy() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& wwwy() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& wwwz() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& wwwz() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& wwww() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& wwww() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& rrrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& rrrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& rrrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& rrrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& rrrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& rrrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& rrra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& rrra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& rrgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& rrgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& rrgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& rrgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& rrgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& rrgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& rrga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& rrga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& rrbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& rrbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& rrbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& rrbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& rrbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& rrbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& rrba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& rrba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& rrar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& rrar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& rrag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& rrag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& rrab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& rrab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& rraa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& rraa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& rgrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& rgrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& rgrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& rgrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& rgrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& rgrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& rgra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& rgra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& rggr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& rggr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& rggg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& rggg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& rggb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& rggb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& rgga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& rgga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& rgbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& rgbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& rgbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& rgbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& rgbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& rgbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& rgba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& rgba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& rgar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& rgar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& rgag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& rgag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& rgab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& rgab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& rgaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& rgaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& rbrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& rbrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& rbrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& rbrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& rbrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& rbrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& rbra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& rbra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& rbgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& rbgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& rbgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& rbgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& rbgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& rbgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& rbga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& rbga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& rbbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& rbbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& rbbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& rbbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& rbbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& rbbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& rbba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& rbba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& rbar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& rbar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& rbag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& rbag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& rbab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& rbab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& rbaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& rbaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& rarr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& rarr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& rarg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& rarg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& rarb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& rarb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& rara() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& rara() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& ragr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& ragr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& ragg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& ragg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& ragb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& ragb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& raga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& raga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& rabr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& rabr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& rabg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& rabg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& rabb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& rabb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& raba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& raba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& raar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& raar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& raag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& raag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& raab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& raab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& raaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(0), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& raaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& grrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& grrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& grrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& grrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& grrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& grrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& grra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& grra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& grgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& grgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& grgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& grgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& grgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& grgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& grga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& grga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& grbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& grbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& grbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& grbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& grbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& grbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& grba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& grba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& grar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& grar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& grag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& grag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& grab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& grab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& graa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& graa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& ggrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& ggrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& ggrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& ggrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& ggrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& ggrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& ggra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& ggra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& gggr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& gggr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& gggg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& gggg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& gggb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& gggb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& ggga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& ggga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& ggbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& ggbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& ggbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& ggbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& ggbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& ggbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& ggba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& ggba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& ggar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& ggar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& ggag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& ggag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& ggab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& ggab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& ggaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& ggaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& gbrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& gbrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& gbrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& gbrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& gbrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& gbrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& gbra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& gbra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& gbgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& gbgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& gbgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& gbgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& gbgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& gbgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& gbga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& gbga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& gbbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& gbbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& gbbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& gbbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& gbbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& gbbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& gbba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& gbba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& gbar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& gbar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& gbag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& gbag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& gbab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& gbab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& gbaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& gbaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& garr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& garr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& garg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& garg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& garb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& garb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& gara() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& gara() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& gagr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& gagr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& gagg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& gagg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& gagb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& gagb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& gaga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& gaga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& gabr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& gabr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& gabg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& gabg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& gabb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& gabb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& gaba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& gaba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& gaar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& gaar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& gaag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& gaag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& gaab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& gaab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& gaaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(1), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& gaaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& brrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& brrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& brrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& brrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& brrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& brrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& brra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& brra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& brgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& brgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& brgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& brgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& brgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& brgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& brga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& brga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& brbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& brbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& brbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& brbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& brbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& brbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& brba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& brba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& brar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& brar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& brag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& brag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& brab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& brab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& braa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& braa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& bgrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& bgrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& bgrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& bgrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& bgrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& bgrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& bgra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& bgra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& bggr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& bggr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& bggg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& bggg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& bggb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& bggb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& bgga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& bgga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& bgbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& bgbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& bgbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& bgbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& bgbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& bgbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& bgba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& bgba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& bgar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& bgar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& bgag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& bgag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& bgab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& bgab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& bgaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& bgaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& bbrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& bbrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& bbrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& bbrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& bbrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& bbrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& bbra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& bbra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& bbgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& bbgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& bbgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& bbgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& bbgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& bbgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& bbga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& bbga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& bbbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& bbbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& bbbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& bbbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& bbbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& bbbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& bbba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& bbba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& bbar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& bbar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& bbag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& bbag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& bbab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& bbab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& bbaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& bbaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& barr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& barr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& barg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& barg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& barb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& barb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& bara() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& bara() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& bagr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& bagr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& bagg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& bagg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& bagb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& bagb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& baga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& baga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& babr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& babr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& babg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& babg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& babb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& babb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& baba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& baba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& baar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& baar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& baag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& baag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& baab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& baab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& baaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(2), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& baaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(0)>>
  return_t& arrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(0)>>
  const return_t& arrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(1)>>
  return_t& arrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(1)>>
  const return_t& arrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(2)>>
  return_t& arrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(2)>>
  const return_t& arrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(3)>>
  return_t& arra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(0), get_index(3)>>
  const return_t& arra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(0)>>
  return_t& argr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(0)>>
  const return_t& argr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(1)>>
  return_t& argg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(1)>>
  const return_t& argg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(2)>>
  return_t& argb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(2)>>
  const return_t& argb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(3)>>
  return_t& arga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(1), get_index(3)>>
  const return_t& arga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(0)>>
  return_t& arbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(0)>>
  const return_t& arbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(1)>>
  return_t& arbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(1)>>
  const return_t& arbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(2)>>
  return_t& arbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(2)>>
  const return_t& arbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(3)>>
  return_t& arba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(2), get_index(3)>>
  const return_t& arba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(0)>>
  return_t& arar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(0)>>
  const return_t& arar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(1)>>
  return_t& arag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(1)>>
  const return_t& arag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(2)>>
  return_t& arab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(2)>>
  const return_t& arab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(3)>>
  return_t& araa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(0),
                                    get_index(3), get_index(3)>>
  const return_t& araa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(0)>>
  return_t& agrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(0)>>
  const return_t& agrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(1)>>
  return_t& agrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(1)>>
  const return_t& agrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(2)>>
  return_t& agrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(2)>>
  const return_t& agrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(3)>>
  return_t& agra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(0), get_index(3)>>
  const return_t& agra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(0)>>
  return_t& aggr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(0)>>
  const return_t& aggr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(1)>>
  return_t& aggg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(1)>>
  const return_t& aggg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(2)>>
  return_t& aggb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(2)>>
  const return_t& aggb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(3)>>
  return_t& agga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(1), get_index(3)>>
  const return_t& agga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(0)>>
  return_t& agbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(0)>>
  const return_t& agbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(1)>>
  return_t& agbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(1)>>
  const return_t& agbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(2)>>
  return_t& agbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(2)>>
  const return_t& agbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(3)>>
  return_t& agba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(2), get_index(3)>>
  const return_t& agba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(0)>>
  return_t& agar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(0)>>
  const return_t& agar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(1)>>
  return_t& agag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(1)>>
  const return_t& agag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(2)>>
  return_t& agab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(2)>>
  const return_t& agab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(3)>>
  return_t& agaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(1),
                                    get_index(3), get_index(3)>>
  const return_t& agaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(0)>>
  return_t& abrr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(0)>>
  const return_t& abrr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(1)>>
  return_t& abrg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(1)>>
  const return_t& abrg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(2)>>
  return_t& abrb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(2)>>
  const return_t& abrb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(3)>>
  return_t& abra() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(0), get_index(3)>>
  const return_t& abra() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(0)>>
  return_t& abgr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(0)>>
  const return_t& abgr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(1)>>
  return_t& abgg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(1)>>
  const return_t& abgg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(2)>>
  return_t& abgb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(2)>>
  const return_t& abgb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(3)>>
  return_t& abga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(1), get_index(3)>>
  const return_t& abga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(0)>>
  return_t& abbr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(0)>>
  const return_t& abbr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(1)>>
  return_t& abbg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(1)>>
  const return_t& abbg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(2)>>
  return_t& abbb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(2)>>
  const return_t& abbb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(3)>>
  return_t& abba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(2), get_index(3)>>
  const return_t& abba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(0)>>
  return_t& abar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(0)>>
  const return_t& abar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(1)>>
  return_t& abag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(1)>>
  const return_t& abag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(2)>>
  return_t& abab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(2)>>
  const return_t& abab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(3)>>
  return_t& abaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(2),
                                    get_index(3), get_index(3)>>
  const return_t& abaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(0)>>
  return_t& aarr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(0)>>
  const return_t& aarr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(1)>>
  return_t& aarg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(1)>>
  const return_t& aarg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(2)>>
  return_t& aarb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(2)>>
  const return_t& aarb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(3)>>
  return_t& aara() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(0), get_index(3)>>
  const return_t& aara() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(0)>>
  return_t& aagr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(0)>>
  const return_t& aagr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(1)>>
  return_t& aagg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(1)>>
  const return_t& aagg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(2)>>
  return_t& aagb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(2)>>
  const return_t& aagb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(3)>>
  return_t& aaga() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(1), get_index(3)>>
  const return_t& aaga() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(0)>>
  return_t& aabr() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(0)>>
  const return_t& aabr() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(1)>>
  return_t& aabg() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(1)>>
  const return_t& aabg() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(2)>>
  return_t& aabb() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(2)>>
  const return_t& aabb() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(3)>>
  return_t& aaba() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(2), get_index(3)>>
  const return_t& aaba() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(0)>>
  return_t& aaar() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(0)>>
  const return_t& aaar() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(1)>>
  return_t& aaag() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(1)>>
  const return_t& aaag() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(2)>>
  return_t& aaab() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(2)>>
  const return_t& aaab() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(3)>>
  return_t& aaaa() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <
      class return_t = swizzled_vec<dataT, kElems, get_index(3), get_index(3),
                                    get_index(3), get_index(3)>>
  const return_t& aaaa() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

#endif  // SYCL_SIMPLE_SWIZZLES

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0)>>
  return_t& s0() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(0)>>
  const return_t& s0() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1)>>
  return_t& s1() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(1)>>
  const return_t& s1() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2)>>
  return_t& s2() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(2)>>
  const return_t& s2() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3)>>
  return_t& s3() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(3)>>
  const return_t& s3() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(4)>>
  return_t& s4() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(4)>>
  const return_t& s4() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(5)>>
  return_t& s5() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(5)>>
  const return_t& s5() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(6)>>
  return_t& s6() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(6)>>
  const return_t& s6() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(7)>>
  return_t& s7() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(7)>>
  const return_t& s7() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(8)>>
  return_t& s8() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(8)>>
  const return_t& s8() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(9)>>
  return_t& s9() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(9)>>
  const return_t& s9() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(10)>>
  return_t& sA() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(10)>>
  const return_t& sA() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(11)>>
  return_t& sB() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(11)>>
  const return_t& sB() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(12)>>
  return_t& sC() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(12)>>
  const return_t& sC() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(13)>>
  return_t& sD() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(13)>>
  const return_t& sD() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(14)>>
  return_t& sE() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(14)>>
  const return_t& sE() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(15)>>
  return_t& sF() {
    auto swizzledVec = reinterpret_cast<return_t*>(this);
    return *swizzledVec;
  }

  template <class return_t = swizzled_vec<dataT, kElems, get_index(15)>>
  const return_t& sF() const {
    auto swizzledVec = reinterpret_cast<const return_t*>(this);
    return *swizzledVec;
  }

  COMPUTECPP_CLANG_FORMAT_BARRIER

  template <int... kDestIndexesN>
  const swizzled_vec<dataT, kElems, get_index(kDestIndexesN)...>& swizzle()
      const {
    return (*(reinterpret_cast<const swizzled_vec<
                  dataT, kElems, get_index(kDestIndexesN)...>* const>(this)));
  }

  template <int... kDestIndexesN>
  swizzled_vec<dataT, kElems, get_index(kDestIndexesN)...>& swizzle() {
    return (
        *(reinterpret_cast<
            swizzled_vec<dataT, kElems, get_index(kDestIndexesN)...>*>(this)));
  }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<
                sizeof...(kIndexesN) == sizeof...(kIndexesRhsN), dataT>::type>
  swizzled_vec<dataT, kElems, kIndexesN...>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, typename E = typename std::enable_if<
                               sizeof...(kIndexesN) == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, kIndexesN...>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, kIndexesN...>::apply(*this,
                                                                       rhs);
    return *this;
  }

  /** @brief Constructs a vec object from the scalar value and then performs a
   * lhs swizzle operation on the this swizzled_vec object, assigning the new
   * vector object.
   * @param rhs The reference a scalar value to be assigned.
   * @return The this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...>& operator=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> rhsAsVec(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, rhsAsVec);
    return *this;
  }

  swizzled_vec<dataT, kElems, kIndexesN...>& operator++() {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, sizeof...(kIndexesN)> operator++(int) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    vec<dataT, sizeof...(kIndexesN)> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, kIndexesN...>& operator--() {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, sizeof...(kIndexesN)> operator--(int) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    vec<dataT, sizeof...(kIndexesN)> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return save;
  }

  vec<dataT, sizeof...(kIndexesN)> operator-() {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, sizeof...(kIndexesN)> operator~() {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator+=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator-=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator*=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator/=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator%=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator&=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator|=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator^=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator<<=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator>>=(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator+=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator-=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator*=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator/=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator%=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator&=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator|=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator^=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator<<=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, kIndexesN...> operator>>=(dataT rhs) {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesN),
                        kIndexesN...>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator+(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator-(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator*(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator/(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator%(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator&(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator|(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator^(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator<<(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator>>(
      const vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator+(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator-(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator*(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator/(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator%(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator&(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator|(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator^(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator<<(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, sizeof...(kIndexesN)> operator>>(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator&&(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator||(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator==(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator!=(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<=(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>=(vec<dataT, sizeof...(kIndexesN)>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator&&(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator||(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator==(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator!=(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<=(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another swizzled_vec
   * object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>=(swizzled_vec<dataT, kElems, kIndexesN...>& rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator&&(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator||(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator==(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator!=(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator<=(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator>=(dataT rhs) const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
      sizeof...(kIndexesN)>
  operator!() const {
    vec<dataT, sizeof...(kIndexesN)> thisAsVec{*this};
    return !thisAsVec;
  }

};  // class swizzled_vec

namespace detail {

/** @brief Overload for deducing swizzled_vec<T, N>.
 * @see deduce_type_impl_f.
 */
template <typename T, int N>
vec<deduce_type_t<T>, N> deduce_type_impl_f(swizzled_vec<T, N>);

/** @brief Overload for deducing swizzled_vec<T, 1>.
 * @see deduce_type_impl_f.
 */
template <typename T>
deduce_type_t<T> deduce_type_impl_f(swizzled_vec<T, 1>);

}  // namespace detail

#ifdef __SYCL_DEVICE_ONLY__

// One dimensional vectors are illegal in OpenCL, SPIR, and SPIR-V but with the
// implementation above we generate 1D vecs while doing operations on single
// access swizzles (i.e .x(), .y())
// By providing these template specialisations we lower the one-d
// vector to scalar and perform the operation on the scalar directly. This is
// only for device as the host implementation already performs the operations
// as a series of scalars.

// This class provides a common class for the specializations to inherit from
// Providing overloads for the assignment and assignement like operators.
// This class doesn't have doxygen because as a template specialization of an
// already documented class it doesn't appear in the documentation and just uses
// parents
template <typename dataT, int kElems, int kIndex>
class detail::swizzled_vec_intermediate<dataT, kElems, kIndex>
    : public mem_container_base<dataT, kElems> {
  using ActualType = swizzled_vec<dataT, kElems, kIndex>;

 public:
  swizzled_vec<dataT, kElems, kIndex>& operator=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator+=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() + rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator-=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() - rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator*=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() * rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator/=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() / rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator|=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() | rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator^=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() ^ rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator<<=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() << rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator>>=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() >> rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator&=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() & rhs);
    return *p;
  }
  swizzled_vec<dataT, kElems, kIndex>& operator%=(dataT rhs) {
    ActualType* p = static_cast<ActualType*>(this);
    p->set_swizzle_value(p->get_swizzle_value() % rhs);
    return *p;
  }
};

// We produce the following specializations for one element swizzles to force
// demotion to a scalar during codegen as 1 element vectors are not legal OpenCL

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s0>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s0> {
 public:
  swizzled_vec<dataT, kElems, detail::s0>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s0>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.x; }
  inline void set_swizzle_value(dataT v) { this->m_data.x = v; }
  operator dataT() const { return this->m_data.x; }
  operator dataT() { return this->m_data.x; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s0>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s0>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s0>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s0>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s0>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s0> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s0>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s0>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s0>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s0>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s0>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s0>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s0>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s0>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s0>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s0, detail::s0>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s0, detail::s0>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s0, detail::s0>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s0, detail::s0>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0, detail::s0>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s0, detail::s0,
                                      detail::s0, detail::s0>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s0, detail::s0, detail::s0,
                     detail::s0>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s0, detail::s0, detail::s0, detail::s0>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s1>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s1> {
 public:
  swizzled_vec<dataT, kElems, detail::s1>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s1>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s1; }
  inline void set_swizzle_value(dataT v) { this->m_data.s1 = v; }
  operator dataT() const { return this->m_data.s1; }
  operator dataT() { return this->m_data.s1; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s1>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s1>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s1>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s1>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s1>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s1> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s1>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s1>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s1>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s1>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s1>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s1>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s1>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s1>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s1>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s1, detail::s1>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s1, detail::s1>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s1, detail::s1>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s1, detail::s1>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1, detail::s1>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s1, detail::s1,
                                      detail::s1, detail::s1>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s1, detail::s1, detail::s1,
                     detail::s1>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s1, detail::s1, detail::s1, detail::s1>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s2>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s2> {
 public:
  swizzled_vec<dataT, kElems, detail::s2>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s2>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s2; }
  inline void set_swizzle_value(dataT v) { this->m_data.s2 = v; }
  operator dataT() const { return this->m_data.s2; }
  operator dataT() { return this->m_data.s2; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s2>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s2>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s2>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s2>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s2>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s2> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s2>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s2>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s2>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s2>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s2>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s2>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s2>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s2>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s2>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s2, detail::s2>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s2, detail::s2>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s2, detail::s2>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s2, detail::s2>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2, detail::s2>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s2, detail::s2,
                                      detail::s2, detail::s2>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s2, detail::s2, detail::s2,
                     detail::s2>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s2, detail::s2, detail::s2, detail::s2>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s3>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s3> {
 public:
  swizzled_vec<dataT, kElems, detail::s3>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s3>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s3; }
  inline void set_swizzle_value(dataT v) { this->m_data.s3 = v; }
  operator dataT() const { return this->m_data.s3; }
  operator dataT() { return this->m_data.s3; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s3>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s3>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s3>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s3>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s3>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s3> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s3>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s3>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s3>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s3>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s3>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s3>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s3>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s3>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s3>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s3, detail::s3>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s3, detail::s3>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s3, detail::s3>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s3, detail::s3>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3, detail::s3>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s3, detail::s3,
                                      detail::s3, detail::s3>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s3, detail::s3, detail::s3,
                     detail::s3>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s3, detail::s3, detail::s3, detail::s3>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s4>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s4> {
 public:
  swizzled_vec<dataT, kElems, detail::s4>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s4>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s4; }
  inline void set_swizzle_value(dataT v) { this->m_data.s4 = v; }
  operator dataT() const { return this->m_data.s4; }
  operator dataT() { return this->m_data.s4; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s4>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s4>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s4>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s4>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s4>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s4> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s4>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s4>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s4>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s4>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s4>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s4>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s4>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s4>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s4>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s4, detail::s4>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s4, detail::s4>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s4, detail::s4>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s4, detail::s4>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4, detail::s4>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s4, detail::s4,
                                      detail::s4, detail::s4>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s4, detail::s4, detail::s4,
                     detail::s4>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s4, detail::s4, detail::s4, detail::s4>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s5>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s5> {
 public:
  swizzled_vec<dataT, kElems, detail::s5>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s5>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s5; }
  inline void set_swizzle_value(dataT v) { this->m_data.s5 = v; }
  operator dataT() const { return this->m_data.s5; }
  operator dataT() { return this->m_data.s5; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s5>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s5>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s5>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s5>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s5>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s5> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s5>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s5>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s5>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s5>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s5>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s5>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s5>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s5>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s5>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s5, detail::s5>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s5, detail::s5>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s5, detail::s5>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s5, detail::s5>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5, detail::s5>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s5, detail::s5,
                                      detail::s5, detail::s5>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s5, detail::s5, detail::s5,
                     detail::s5>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s5, detail::s5, detail::s5, detail::s5>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s6>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s6> {
 public:
  swizzled_vec<dataT, kElems, detail::s6>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s6>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s6; }
  inline void set_swizzle_value(dataT v) { this->m_data.s6 = v; }
  operator dataT() const { return this->m_data.s6; }
  operator dataT() { return this->m_data.s6; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s6>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s6>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s6>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s6>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s6>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s6> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s6>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s6>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s6>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s6>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s6>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s6>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s6>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s6>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s6>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s6, detail::s6>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s6, detail::s6>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s6, detail::s6>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s6, detail::s6>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6, detail::s6>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s6, detail::s6,
                                      detail::s6, detail::s6>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s6, detail::s6, detail::s6,
                     detail::s6>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s6, detail::s6, detail::s6, detail::s6>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s7>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s7> {
 public:
  swizzled_vec<dataT, kElems, detail::s7>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s7>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s7; }
  inline void set_swizzle_value(dataT v) { this->m_data.s7 = v; }
  operator dataT() const { return this->m_data.s7; }
  operator dataT() { return this->m_data.s7; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s7>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s7>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s7>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s7>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s7>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s7> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s7>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s7>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s7>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s7>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s7>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s7>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s7>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s7>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s7>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s7, detail::s7>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s7, detail::s7>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s7, detail::s7>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s7, detail::s7>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7, detail::s7>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s7, detail::s7,
                                      detail::s7, detail::s7>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s7, detail::s7, detail::s7,
                     detail::s7>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s7, detail::s7, detail::s7, detail::s7>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s8>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s8> {
 public:
  swizzled_vec<dataT, kElems, detail::s8>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s8>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s8; }
  inline void set_swizzle_value(dataT v) { this->m_data.s8 = v; }
  operator dataT() const { return this->m_data.s8; }
  operator dataT() { return this->m_data.s8; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s8>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s8>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s8>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s8>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s8>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s8> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s8>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s8>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s8>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s8>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s8>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s8>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s8>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s8>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s8>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s8, detail::s8>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s8, detail::s8>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s8, detail::s8>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s8, detail::s8>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8, detail::s8>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s8, detail::s8,
                                      detail::s8, detail::s8>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s8, detail::s8, detail::s8,
                     detail::s8>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s8, detail::s8, detail::s8, detail::s8>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::s9>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::s9> {
 public:
  swizzled_vec<dataT, kElems, detail::s9>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::s9>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.s9; }
  inline void set_swizzle_value(dataT v) { this->m_data.s9 = v; }
  operator dataT() const { return this->m_data.s9; }
  operator dataT() { return this->m_data.s9; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::s9>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::s9>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::s9>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::s9>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::s9>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::s9> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::s9>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::s9>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s9>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s9>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s9>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s9>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s9>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s9>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s9>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::s9, detail::s9>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s9, detail::s9>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s9, detail::s9>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s9, detail::s9>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9, detail::s9>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s9, detail::s9,
                                      detail::s9, detail::s9>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::s9, detail::s9, detail::s9,
                     detail::s9>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::s9, detail::s9, detail::s9, detail::s9>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sA>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sA> {
 public:
  swizzled_vec<dataT, kElems, detail::sA>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sA>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sA; }
  inline void set_swizzle_value(dataT v) { this->m_data.sA = v; }
  operator dataT() const { return this->m_data.sA; }
  operator dataT() { return this->m_data.sA; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sA>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sA>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sA>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sA>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sA>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sA> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sA>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sA>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sA>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sA>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sA>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sA>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sA>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sA>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sA>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sA, detail::sA>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sA, detail::sA>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sA, detail::sA>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sA, detail::sA>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA, detail::sA>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sA, detail::sA,
                                      detail::sA, detail::sA>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sA, detail::sA, detail::sA,
                     detail::sA>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sA, detail::sA, detail::sA, detail::sA>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sB>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sB> {
 public:
  swizzled_vec<dataT, kElems, detail::sB>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sB>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sB; }
  inline void set_swizzle_value(dataT v) { this->m_data.sB = v; }
  operator dataT() const { return this->m_data.sB; }
  operator dataT() { return this->m_data.sB; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sB>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sB>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sB>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sB>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sB>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sB> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sB>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sB>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sB>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sB>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sB>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sB>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sB>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sB>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sB>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sB, detail::sB>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sB, detail::sB>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sB, detail::sB>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sB, detail::sB>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB, detail::sB>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sB, detail::sB,
                                      detail::sB, detail::sB>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sB, detail::sB, detail::sB,
                     detail::sB>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sB, detail::sB, detail::sB, detail::sB>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sC>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sC> {
 public:
  swizzled_vec<dataT, kElems, detail::sC>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sC>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sC; }
  inline void set_swizzle_value(dataT v) { this->m_data.sC = v; }
  operator dataT() const { return this->m_data.sC; }
  operator dataT() { return this->m_data.sC; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sC>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sC>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sC>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sC>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sC>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sC> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sC>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sC>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sC>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sC>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sC>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sC>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sC>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sC>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sC>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sC, detail::sC>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sC, detail::sC>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sC, detail::sC>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sC, detail::sC>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC, detail::sC>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sC, detail::sC,
                                      detail::sC, detail::sC>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sC, detail::sC, detail::sC,
                     detail::sC>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sC, detail::sC, detail::sC, detail::sC>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sD>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sD> {
 public:
  swizzled_vec<dataT, kElems, detail::sD>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sD>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sD; }
  inline void set_swizzle_value(dataT v) { this->m_data.sD = v; }
  operator dataT() const { return this->m_data.sD; }
  operator dataT() { return this->m_data.sD; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sD>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sD>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sD>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sD>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sD>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sD> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sD>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sD>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sD>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sD>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sD>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sD>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sD>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sD>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sD>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sD, detail::sD>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sD, detail::sD>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sD, detail::sD>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sD, detail::sD>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD, detail::sD>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sD, detail::sD,
                                      detail::sD, detail::sD>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sD, detail::sD, detail::sD,
                     detail::sD>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sD, detail::sD, detail::sD, detail::sD>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sE>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sE> {
 public:
  swizzled_vec<dataT, kElems, detail::sE>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sE>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sE; }
  inline void set_swizzle_value(dataT v) { this->m_data.sE = v; }
  operator dataT() const { return this->m_data.sE; }
  operator dataT() { return this->m_data.sE; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sE>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sE>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sE>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sE>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sE>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sE> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sE>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sE>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sE>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sE>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sE>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sE>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sE>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sE>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sE>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sE, detail::sE>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sE, detail::sE>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sE, detail::sE>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sE, detail::sE>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE, detail::sE>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sE, detail::sE,
                                      detail::sE, detail::sE>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sE, detail::sE, detail::sE,
                     detail::sE>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sE, detail::sE, detail::sE, detail::sE>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

template <typename dataT, int kElems>
class swizzled_vec<dataT, kElems, detail::sF>
    : public detail::swizzled_vec_intermediate<dataT, kElems, detail::sF> {
 public:
  swizzled_vec<dataT, kElems, detail::sF>& operator=(dataT rhs) {
    return detail::swizzled_vec_intermediate<dataT, kElems,
                                             detail::sF>::operator=(rhs);
  }
  inline dataT get_swizzle_value() { return this->m_data.sF; }
  inline void set_swizzle_value(dataT v) { this->m_data.sF = v; }
  operator dataT() const { return this->m_data.sF; }
  operator dataT() { return this->m_data.sF; }

  /** @brief Performs a rhs swizzle operation on the rhs swizzled_vec objects
   * and a lhs swizzle operation on the this swizzled_vec object and then
   * assigns the rhs result to the lhs result.
   * @tparam kElemsRhs The number of elements in the rhs swizzled_vec
   * object.
   * @tparam kIndexesRhsN The variadic argument packs for the rhs swizzle
   * indexes.
   * @param rhs The reference to the swizzled_vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs, int... kIndexesRhsN,
            typename E = typename std::enable_if<1 == sizeof...(kIndexesRhsN),
                                                 dataT>::type>
  swizzled_vec<dataT, kElems, detail::sF>& operator=(
      const swizzled_vec<dataT, kElemsRhs, kIndexesRhsN...>& rhs) {  // NOLINT
    vec<dataT, sizeof...(kIndexesRhsN)> newVec =
        detail::swizzle_rhs<dataT, sizeof...(kIndexesRhsN), kElems,
                            kIndexesRhsN...>::apply(rhs);
    detail::swizzle_lhs<dataT, kElems, sizeof...(kIndexesRhsN),
                        detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Performs a lhs swizzle operation on the this swizzled_vec object
   * and assigns to it the rhs vec object.
   * @tparam kElemsRhs The number of elements in the vec object.
   * @param rhs The reference to the vec object to be assigned.
   * @return The this object.
   */
  template <int kElemsRhs,
            typename E = typename std::enable_if<1 == kElemsRhs, dataT>::type>
  swizzled_vec<dataT, kElems, detail::sF>& operator=(
      const vec<dataT, kElemsRhs>& rhs) {
    detail::swizzle_lhs<dataT, kElems, kElemsRhs, detail::sF>::apply(*this,
                                                                     rhs);
    return *this;
  }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::size() instead.")
  size_t get_count() const { return 1; }

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use vec::byte_size() instead.")
  size_t get_size() const { return sizeof(dataT); }

#if SYCL_LANGUAGE_VERSION >= 202001
  size_t size() const noexcept { return 1; }

  size_t byte_size() const noexcept { return sizeof(dataT); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, 1> convert() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template convert<convertT, roundingMode>();
  }

  template <typename asT>
  asT as() const {
    vec<dataT, 1> newVec{*this};
    return newVec.template as<asT>();
  }

  swizzled_vec<dataT, kElems, detail::sF>& operator++() {
    vec<dataT, 1> newVec{*this};
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator++(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec += 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return save;
  }

  swizzled_vec<dataT, kElems, detail::sF>& operator--() {
    vec<dataT, 1> newVec{*this};
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  vec<dataT, 1> operator--(int) {
    vec<dataT, 1> newVec{*this};
    vec<dataT, 1> save = newVec;
    newVec -= 1;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return save;
  }

  vec<dataT, 1> operator-() {
    vec<dataT, 1> newVec{*this};
    newVec = -newVec;
    return newVec;
  }

  vec<dataT, 1> operator~() {
    vec<dataT, 1> newVec{*this};
    newVec = ~newVec;
    return newVec;
  }

  /** @brief Applies += to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator+=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator-=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator*=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator/=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator%=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator&=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator|=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator^=(const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator<<=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator>>=(
      const vec<dataT, 1>& rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies += to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator+=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec += rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies -= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator-=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec -= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies *= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator*=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec *= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies /= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator/=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec /= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies %= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator%=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec %= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies &= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator&=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec &= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies |= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator|=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec |= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies ^= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator^=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec ^= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies <<= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator<<=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec <<= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies >>= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return this object.
   */
  swizzled_vec<dataT, kElems, detail::sF> operator>>=(dataT rhs) {
    vec<dataT, 1> newVec{*this};
    newVec >>= rhs;
    detail::swizzle_lhs<dataT, kElems, 1, detail::sF>::apply(*this, newVec);
    return *this;
  }

  /** @brief Applies + to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and another vec object.
   * @param rhs The rhs vec reference.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(const vec<dataT, 1>& rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies + to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the + operation.
   */
  vec<dataT, 1> operator+(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec + rhs;
  }

  /** @brief Applies - to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the - operation.
   */
  vec<dataT, 1> operator-(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec - rhs;
  }

  /** @brief Applies * to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the * operation.
   */
  vec<dataT, 1> operator*(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec * rhs;
  }

  /** @brief Applies / to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the / operation.
   */
  vec<dataT, 1> operator/(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec / rhs;
  }

  /** @brief Applies % to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the % operation.
   */
  vec<dataT, 1> operator%(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec % rhs;
  }

  /** @brief Applies & to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the & operation.
   */
  vec<dataT, 1> operator&(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec & rhs;
  }

  /** @brief Applies | to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the | operation.
   */
  vec<dataT, 1> operator|(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec | rhs;
  }

  /** @brief Applies ^ to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the ^ operation.
   */
  vec<dataT, 1> operator^(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec ^ rhs;
  }

  /** @brief Applies << to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the << operation.
   */
  vec<dataT, 1> operator<<(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec << rhs;
  }

  /** @brief Applies >> to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return The result of the >> operation.
   */
  vec<dataT, 1> operator>>(dataT rhs) const {
    vec<dataT, 1> newVec{*this};
    return newVec >> rhs;
  }

  /** @brief Applies && to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and another vec object.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(vec<dataT, 1>& rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Applies && to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the && operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator&&(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec && rhs;
  }

  /** @brief Applies || to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the || operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator||(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec || rhs;
  }

  /** @brief Applies == to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the == operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator==(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec == rhs;
  }

  /** @brief Applies != to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the != operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec != rhs;
  }

  /** @brief Applies < to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the < operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec < rhs;
  }

  /** @brief Applies > to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the > operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec > rhs;
  }

  /** @brief Applies <= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the <= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator<=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec <= rhs;
  }

  /** @brief Applies >= to this swizzled_vec object and a scalar.
   * @param rhs The rhs scalar.
   * @return A new vector of correct type with the result of the >= operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator>=(dataT rhs) const {
    vec<dataT, 1> thisAsVec{*this};
    return thisAsVec >= rhs;
  }

  /** @brief Creates a new vec from the application of ! to each swizzled_vec
   * element.
   * @return A new vector of correct type with the result of the ! operation.
   */
  vec<typename detail::vec_ops::logical_return<sizeof(dataT)>::type, 1>
  operator!() const {
    vec<dataT, 1> thisAsVec{*this};
    return !thisAsVec;
  }

  swizzled_vec<dataT, kElems, detail::sF>& x() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sF>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sF>& x() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sF>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sF>& s0() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sF>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sF>& s0() const {
    auto swizzledVec =
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::sF>*>(this);
    return *swizzledVec;
  }

#ifdef SYCL_SIMPLE_SWIZZLES
  swizzled_vec<dataT, kElems, detail::sF, detail::sF>& xx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sF, detail::sF>*>(
            this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sF, detail::sF>& xx() const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sF, detail::sF>*>(this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF>& xxx() {
    auto swizzledVec = reinterpret_cast<
        swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF>& xxx()
      const {
    auto swizzledVec = reinterpret_cast<
        const swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF>*>(
        this);
    return *swizzledVec;
  }

  swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF, detail::sF>&
  xxxx() {
    auto swizzledVec =
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::sF, detail::sF,
                                      detail::sF, detail::sF>*>(this);
    return *swizzledVec;
  }

  const swizzled_vec<dataT, kElems, detail::sF, detail::sF, detail::sF,
                     detail::sF>&
  xxxx() const {
    auto swizzledVec = reinterpret_cast<const swizzled_vec<
        dataT, kElems, detail::sF, detail::sF, detail::sF, detail::sF>*>(this);
    return *swizzledVec;
  }
#endif  // SYCL_SIMPLE_SWIZZLES
};

#endif  // __SYCL_DEVICE_ONLY__

/** COMPUTECPP_DEV @endcond */

/** @cond COMPUTECPP_DEV */

/** @brief Adds a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The result of the addition.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator+(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return rhsCast + lhsCast;
}

/** @brief Subtracts a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The result of the subtraction.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator-(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast - rhsCast;
}

/** @brief Multiplies a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The result of the multiplication.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator*(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast * rhsCast;
}

/** @brief Divides a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The result of the division.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator/(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast / rhsCast;
}

/** @brief Bitwise XOR of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator^(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast ^ rhsCast;
}

/** @brief Modulo of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator%(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast % rhsCast;
}

/** @brief Bitwise OR of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator|(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast | rhsCast;
}

/** @brief Bitwise AND of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type operator&(
    dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast & rhsCast;
}

/** @brief Shift left of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type
operator<<(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast << rhsCast;
}

/** @brief Shift right of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return The this object.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type
operator>>(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  return lhsCast >> rhsCast;
}

/** @brief Logical AND of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator&&(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast && rhsCast);
  } else {
    return lhsCast && rhsCast;
  }
}

/** @brief Logical OR of a scalar to a swizzle vec object.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator||(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast || rhsCast);
  } else {
    return lhsCast || rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (==) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator==(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast == rhsCast);
  } else {
    return lhsCast == rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (!=) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator!=(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast != rhsCast);
  } else {
    return lhsCast != rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (>=) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator>=(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast >= rhsCast);
  } else {
    return lhsCast >= rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (>) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator>(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast > rhsCast);
  } else {
    return lhsCast > rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (<) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator<(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast < rhsCast);
  } else {
    return lhsCast < rhsCast;
  }
}

/** @brief Compares all elements of both vec objects and returns an int vector,
 * where the elements are -1 if the (<) operation returns true and 0 if it
 * returns false.
 * @param lhs The lhs scalar.
 * @param rhs The rhs vec reference.
 * @return Returns an int vector representing the result of the operation.
 */
template <typename dataT, int kElems, int... kIndexesN>
typename detail::swizzle_return_ty<
    typename detail::vec_ops::logical_return<sizeof(dataT)>::type,
    sizeof...(kIndexesN)>::Type
operator<=(dataT lhs, const swizzled_vec<dataT, kElems, kIndexesN...>& rhs) {
  // Here we cast both the lhs and rhs in order to use the resolved type
  // conversion and therefore avoid calling this operator recursively.
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type lhsCast(
      lhs);
  typename detail::swizzle_return_ty<dataT, sizeof...(kIndexesN)>::Type rhsCast(
      rhs);
  if (sizeof...(kIndexesN) == 1) {
    return -(lhsCast <= rhsCast);
  } else {
    return lhsCast <= rhsCast;
  }
}

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_IMPL_H_

////////////////////////////////////////////////////////////////////////////////
