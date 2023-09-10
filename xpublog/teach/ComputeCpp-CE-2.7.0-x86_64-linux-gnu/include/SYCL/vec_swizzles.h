/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_swizzles.h

 @brief This file contains the vector swizzle class definition used by @ref
 cl::sycl::vec.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_H_
#define RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_H_

#include "SYCL/common.h"
#include "SYCL/vec.h"

////////////////////////////////////////////////////////////////////////////////

namespace cl {

namespace sycl {

namespace detail {

/** Wrapper struct for the static apply function for rhs swizzle operations to
 * allow partial template specialization. Host Only.
 * @tparam dataT The data type.
 * @tparam kElemsRes The number of elements in the resultant vec object.
 * @tparam kElemsRhs The number of elements in the rhs swizzled_vec reference.
 * @tparam kIndexesN The variadic argument pack for swizzle indexes.
 */
template <typename dataT, int kElemsRes, int kElemsRhs, int... kIndexesN>
struct swizzle_rhs {
  /** @brief Alias for the swizzled_vec type used as rhs
   */
  using swizzled_vec_t = swizzled_vec<dataT, kElemsRhs, kIndexesN...>;

  /** @brief Assigns values from rhs to lhs, using swizzle index values
   *        as rhs indexes. Host only.
   * @param rhs The rhs swizzled_vec reference being assigned from.
   * @return The new constructed vec object.
   */
  static vec<dataT, kElemsRes> apply(const swizzled_vec_t& rhs) {
    vec<dataT, kElemsRes> newVec;
    for (int i = 0; i < kElemsRes; ++i) {
      int rhsIndex = swizzled_vec_t::get_index(i);
      newVec.set_value(i, rhs.get_value(rhsIndex));
    }
    return newVec;
  }
};

/** Wrapper struct for the static apply function for lhs swizzle operations to
 * allow partial template specialization. Host Only.
 * @tparam dataT The data type.
 * @tparam kElemsLhs The number of elements in the lhs swizzled_vec reference.
 * @tparam kElemsRhs The number of elements in the rhs vec reference.
 * @tparam kIndexesN The variadic argument pack for swizzle indexes.
 */
template <typename dataT, int kElemsLhs, int kElemsRhs, int... kIndexesN>
struct swizzle_lhs {
  /** @brief Alias for the swizzled_vec type used as lhs
   */
  using swizzled_vec_t = swizzled_vec<dataT, kElemsLhs, kIndexesN...>;

  /** @brief Assigns values from rhs to lhs, using swizzle index values
   *        as lhs indexes. Host only.
   * @param lhs The lhs swizzled_vec reference being assigned to.
   * @param rhs The rhs vec reference being assigned from.
   */
  static void apply(swizzled_vec_t& lhs, const vec<dataT, kElemsRhs>& rhs) {
    constexpr int size = sizeof...(kIndexesN);
    for (int i = 0; i < size; ++i) {
      int lhsIndex = swizzled_vec_t::get_index(i);
      lhs.set_value(lhsIndex, rhs.get_value(i));
    }
  }
};

#ifdef __SYCL_DEVICE_ONLY__

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::x> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::x>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.x;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::x> {
  static void apply(swizzled_vec<dataT, 1, detail::x>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.x = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 2, detail::x> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 2, detail::x>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.x;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 1, detail::x> {
  static void apply(swizzled_vec<dataT, 2, detail::x>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.x = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 2, detail::y> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 2, detail::y>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.y;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 1, detail::y> {
  static void apply(swizzled_vec<dataT, 2, detail::y>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.y = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 3, detail::x> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 3, detail::x>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.x;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 1, detail::x> {
  static void apply(swizzled_vec<dataT, 3, detail::x>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.x = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 3, detail::y> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 3, detail::y>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.y;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 1, detail::y> {
  static void apply(swizzled_vec<dataT, 3, detail::y>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.y = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 3, detail::z> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 3, detail::z>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.z;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 1, detail::z> {
  static void apply(swizzled_vec<dataT, 3, detail::z>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.z = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 4, detail::x> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 4, detail::x>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.x;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 1, detail::x> {
  static void apply(swizzled_vec<dataT, 4, detail::x>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.x = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 4, detail::y> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 4, detail::y>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.y;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 1, detail::y> {
  static void apply(swizzled_vec<dataT, 4, detail::y>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.y = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 4, detail::z> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 4, detail::z>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.z;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 1, detail::z> {
  static void apply(swizzled_vec<dataT, 4, detail::z>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.z = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 4, detail::w> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 4, detail::w>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.w;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 1, detail::w> {
  static void apply(swizzled_vec<dataT, 4, detail::w>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.w = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 1, detail::x, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 1, detail::x, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 2, detail::x, detail::x> {
  static void apply(swizzled_vec<dataT, 1, detail::x, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 2, detail::x, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 2, detail::x, detail::x> {
  static void apply(swizzled_vec<dataT, 2, detail::x, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 2, detail::x, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 2, detail::x, detail::y> {
  static void apply(swizzled_vec<dataT, 2, detail::x, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 2, detail::y, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 2, detail::y, detail::x> {
  static void apply(swizzled_vec<dataT, 2, detail::y, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 2, detail::y, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 2, detail::y, detail::y> {
  static void apply(swizzled_vec<dataT, 2, detail::y, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::x, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::x, detail::x> {
  static void apply(swizzled_vec<dataT, 3, detail::x, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::x, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::x, detail::y> {
  static void apply(swizzled_vec<dataT, 3, detail::x, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::x, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::x, detail::z> {
  static void apply(swizzled_vec<dataT, 3, detail::x, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::y, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::y, detail::x> {
  static void apply(swizzled_vec<dataT, 3, detail::y, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::y, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::y, detail::y> {
  static void apply(swizzled_vec<dataT, 3, detail::y, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::y, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::y, detail::z> {
  static void apply(swizzled_vec<dataT, 3, detail::y, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::z, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::z, detail::x> {
  static void apply(swizzled_vec<dataT, 3, detail::z, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::z, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::z, detail::y> {
  static void apply(swizzled_vec<dataT, 3, detail::z, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 3, detail::z, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 2, detail::z, detail::z> {
  static void apply(swizzled_vec<dataT, 3, detail::z, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::x, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::x, detail::x> {
  static void apply(swizzled_vec<dataT, 4, detail::x, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::x, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::x, detail::y> {
  static void apply(swizzled_vec<dataT, 4, detail::x, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::x, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::x, detail::z> {
  static void apply(swizzled_vec<dataT, 4, detail::x, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::x, detail::w> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::w>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.xw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::x, detail::w> {
  static void apply(swizzled_vec<dataT, 4, detail::x, detail::w>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.xw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::y, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::y, detail::x> {
  static void apply(swizzled_vec<dataT, 4, detail::y, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::y, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::y, detail::y> {
  static void apply(swizzled_vec<dataT, 4, detail::y, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::y, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::y, detail::z> {
  static void apply(swizzled_vec<dataT, 4, detail::y, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::y, detail::w> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::w>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.yw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::y, detail::w> {
  static void apply(swizzled_vec<dataT, 4, detail::y, detail::w>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.yw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::z, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::z, detail::x> {
  static void apply(swizzled_vec<dataT, 4, detail::z, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::z, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::z, detail::y> {
  static void apply(swizzled_vec<dataT, 4, detail::z, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::z, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::z, detail::z> {
  static void apply(swizzled_vec<dataT, 4, detail::z, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::z, detail::w> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::w>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.zw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::z, detail::w> {
  static void apply(swizzled_vec<dataT, 4, detail::z, detail::w>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.zw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::w, detail::x> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::x>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.wx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::w, detail::x> {
  static void apply(swizzled_vec<dataT, 4, detail::w, detail::x>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.wx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::w, detail::y> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::y>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.wy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::w, detail::y> {
  static void apply(swizzled_vec<dataT, 4, detail::w, detail::y>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.wy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::w, detail::z> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::z>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.wz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::w, detail::z> {
  static void apply(swizzled_vec<dataT, 4, detail::w, detail::z>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.wz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 2, 4, detail::w, detail::w> {
  static vec<dataT, 2> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::w>& rhs) {
    vec<dataT, 2> newVec;
    newVec.m_data = rhs.m_data.ww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 2, detail::w, detail::w> {
  static void apply(swizzled_vec<dataT, 4, detail::w, detail::w>& lhs,
                    const vec<dataT, 2>& rhs) {
    lhs.m_data.ww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 1, detail::x, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 1, detail::x, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 3, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 1, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::x, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::x, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::x, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::x, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::x, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::y, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::y, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::y, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 2, detail::y, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 2, detail::y, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 3, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::x, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::x, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::y, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::y, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 3, detail::z, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 3, detail::z, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 3, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::x, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::x, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::y, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::y, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::z, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::z, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::w, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::w, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::w, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::w, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::w, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::w, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::x, detail::w, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::x, detail::w, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.xww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::x, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.xww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::x, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::x, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::y, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::y, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::z, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::z, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::w, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::w, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.ywx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.ywx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::w, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::w, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.ywy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.ywy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::w, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::w, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.ywz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.ywz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::y, detail::w, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::y, detail::w, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.yww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::y, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.yww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::x, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::x, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::y, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::y, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::z, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::z, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::w, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::w, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::w, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::w, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::w, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::w, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::z, detail::w, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::z, detail::w, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.zww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::z, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.zww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::x, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::x, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::x, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::x, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::x, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::x, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::x, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::x, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::y, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::y, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::y, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::y, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::y, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::y, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::y, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::y, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::z, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::z, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::z, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::z, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::z, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::z, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::z, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::z, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::w, detail::x> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::w, detail::x>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::x>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::w, detail::y> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::w, detail::y>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::y>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::w, detail::z> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::w, detail::z>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.wwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::z>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.wwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 3, 4, detail::w, detail::w, detail::w> {
  static vec<dataT, 3> apply(
      const swizzled_vec<dataT, 4, detail::w, detail::w, detail::w>& rhs) {
    vec<dataT, 3> newVec;
    newVec.m_data = rhs.m_data.www;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 3, detail::w, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::w>& lhs,
      const vec<dataT, 3>& rhs) {
    lhs.m_data.www = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 1, detail::x, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 1, detail::x, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 4, detail::x, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 1, detail::x, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::x, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::x, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::x, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::x, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 2, detail::y, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 2, detail::y, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 2, 4, detail::y, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 2, detail::y, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::x, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::x, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::x, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::x, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::y, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::y, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::y, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::y, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 3, detail::z, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 3, detail::z, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 3, 4, detail::z, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 3, detail::z, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::x,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xxww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::x, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::x, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xxww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xywx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xywx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xywy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xywy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xywz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xywz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::y,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xyww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::y, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::y, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xyww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::z,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xzww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::z, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::z, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xzww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::x, detail::w,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.xwww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::x, detail::w, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::x, detail::w, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.xwww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::x,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yxww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::x, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::x, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yxww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yywx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yywx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yywy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yywy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yywz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yywz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::y,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yyww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::y, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::y, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yyww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::z,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.yzww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::z, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::z, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.yzww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::y, detail::w,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.ywww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::y, detail::w, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::y, detail::w, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.ywww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::x,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zxww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::x, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::x, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zxww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zywx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zywx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zywy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zywy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zywz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zywz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::y,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zyww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::y, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::y, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zyww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::z,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zzww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::z, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::z, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zzww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::z, detail::w,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.zwww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::z, detail::w, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::z, detail::w, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.zwww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::x,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wxww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::x, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::x, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wxww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wywx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wywx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wywy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wywy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wywz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wywz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::y,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wyww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::y, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::y, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wyww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::z,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wzww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::z, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::z, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wzww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::x, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwxx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::x, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwxx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::x, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwxy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::x, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwxy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::x, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwxz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::x, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwxz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::x, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwxw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::x, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::x, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwxw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::y, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwyx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::y, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwyx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::y, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwyy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::y, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwyy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::y, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwyz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::y, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwyz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::y, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwyw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::y, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::y, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwyw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::z, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwzx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::z, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwzx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::z, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwzy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::z, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwzy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::z, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwzz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::z, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwzz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::z, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwzw;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::z, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::z, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwzw = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::x> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::w, detail::x>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwwx;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::x> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::w, detail::x>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwwx = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::y> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::w, detail::y>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwwy;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::y> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::w, detail::y>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwwy = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::z> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::w, detail::z>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwwz;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::z> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::w, detail::z>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwwz = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::w> {
  static vec<dataT, 4> apply(const swizzled_vec<dataT, 4, detail::w, detail::w,
                                                detail::w, detail::w>& rhs) {
    vec<dataT, 4> newVec;
    newVec.m_data = rhs.m_data.wwww;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 4, 4, detail::w, detail::w, detail::w, detail::w> {
  static void apply(
      swizzled_vec<dataT, 4, detail::w, detail::w, detail::w, detail::w>& lhs,
      const vec<dataT, 4>& rhs) {
    lhs.m_data.wwww = rhs.m_data;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s4> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s4>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s4;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s4> {
  static void apply(swizzled_vec<dataT, 1, detail::s4>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s4 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s5> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s5>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s5;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s5> {
  static void apply(swizzled_vec<dataT, 1, detail::s5>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s5 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s6> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s6>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s6;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s6> {
  static void apply(swizzled_vec<dataT, 1, detail::s6>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s6 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s7> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s7>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s7;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s7> {
  static void apply(swizzled_vec<dataT, 1, detail::s7>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s7 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s8> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s8>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s8;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s8> {
  static void apply(swizzled_vec<dataT, 1, detail::s8>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s8 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::s9> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::s9>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.s9;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::s9> {
  static void apply(swizzled_vec<dataT, 1, detail::s9>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.s9 = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sA> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sA>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sA;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sA> {
  static void apply(swizzled_vec<dataT, 1, detail::sA>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sA = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sB> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sB>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sB;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sB> {
  static void apply(swizzled_vec<dataT, 1, detail::sB>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sB = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sC> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sC>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sC;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sC> {
  static void apply(swizzled_vec<dataT, 1, detail::sC>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sC = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sD> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sD>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sD;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sD> {
  static void apply(swizzled_vec<dataT, 1, detail::sD>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sD = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sE> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sE>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sE;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sE> {
  static void apply(swizzled_vec<dataT, 1, detail::sE>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sE = rhs.m_data.x;
  }
};

template <typename dataT>
struct swizzle_rhs<dataT, 1, 1, detail::sF> {
  static vec<dataT, 1> apply(const swizzled_vec<dataT, 1, detail::sF>& rhs) {
    vec<dataT, 1> newVec;
    newVec.m_data = rhs.m_data.sF;
    return newVec;
  }
};

template <typename dataT>
struct swizzle_lhs<dataT, 1, 1, detail::sF> {
  static void apply(swizzled_vec<dataT, 1, detail::sF>& lhs,
                    const vec<dataT, 1>& rhs) {
    lhs.m_data.sF = rhs.m_data.x;
  }
};

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail

}  // namespace sycl

}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_SWIZZLES_H_

////////////////////////////////////////////////////////////////////////////////
