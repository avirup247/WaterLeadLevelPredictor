/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_common.h

 @brief This file contains some common definitions required for @ref
 cl::sycl::vec.
*/

#ifndef RUNTIME_INCLUDE_SYCL_VEC_COMMON_H_
#define RUNTIME_INCLUDE_SYCL_VEC_COMMON_H_

#include "SYCL/common.h"
#include "SYCL/deduce.h"

////////////////////////////////////////////////////////////////////////////////

namespace cl {
namespace sycl {

namespace detail {

/* The un-named enumerations defined here are used in the swizzle macros in
   order to specify the swizzle indexes.
*/

enum { x, y, z, w };
enum { r, g, b, a };
enum { s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF };

/* The __sycl_vector template alias is used to define device side vector data
  storage type. It is defined by the data type with the ext_vector_type
  extension (clang extension) with the number of elements as its parameter.
*/

#ifdef __SYCL_DEVICE_ONLY__

/** Native Clang vector type.
 *  Used in builtins, so it should only use native device types for dataT.
 * @tparam dataT Underlying data type
 * @tparam kElems Number of vector elements
 */
template <typename dataT, int kElems>
using sycl_vector_native __attribute__((ext_vector_type(kElems))) = dataT;

/** Clang vector type
 * @tparam dataT Underlying data type. Gets translated via @ref deduce_type_t.
 * @tparam kElems Number of vector elements
 */
template <typename dataT, int kElems>
using __sycl_vector = sycl_vector_native<detail::deduce_type_t<dataT>, kElems>;

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail

/** @brief Available vector rounding modes
 */
enum class rounding_mode { automatic, rte, rtz, rtp, rtn };

/** @brief Struct with values that help with accessing vector members
 */
struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

/* Forward declarations for the vec and swizzled_vec classes. */

template <typename dataT, int kElems>
class vec;
template <typename dataT, int kElems, int... sourceTN>
class swizzled_vec;

namespace detail {

/** @brief Template class which represents a compile-time sequence of integer
 * indexes.
 *
 * @tparam kIndexes Variadic pack of integer indexes.
 */
template <int... kIndexes>
struct swizzle_pack {
  /** @brief Number of indexes in the pack
   */
  static constexpr int size = sizeof...(kIndexes);

  /** @brief Store the indexes as an array
   */
  static constexpr int indexes[size] = {kIndexes...};

  /** @brief Retrieves a specific index
   * @param indexPos The position of the index that we're interested in
   * @return Value of the index at the requested position
   */
  static constexpr int get(int indexPos) {
    return indexes[((indexPos >= size) ? (size - 1) : indexPos)];
  }
};

#if __cplusplus < 201703L
/** @brief Definition for the stored indexes.
 *
 *        Required even though it's constexpr because it could be used
 *        in a non-constexpr context
 * @tparam kIndexes Variadic pack of integer indexes.
 */
template <int... kIndexes>
constexpr int
    swizzle_pack<kIndexes...>::indexes[swizzle_pack<kIndexes...>::size];
#endif  // __cplusplus < 201703L

/** @brief Template alias to a swizzled_vec type that is the result of a
 * swizzled_vec transformed by a swizzle_pack.
 *
 * @tparam dataT The data type of the requested swizzled_vec type.
 * @tparam kElems The number of elements of the requested swizzled_vec type.
 * @tparam src_pack_t The swizzle_pack representing the swizzle indexes of the
 *         source swizzled_vec.
 * @tparam kDestIndexes Variadic pack of integer indexes representing the
 *         swizzle indexes of the transformation.
 */
template <typename dataT, int kElems, typename src_pack_t, int... kDestIndexes>
using transform_swizzled_vec_t =
    swizzled_vec<dataT, kElems, src_pack_t::get(kDestIndexes)...>;

/** @brief Struct containing the function for transforming a swizzled_vec
 *        by an integer pack
 *
 * @tparam kDestIndexes Variadic pack of integer indexes representing the
 *         swizzle indexes of the transformation.
 */
template <int... kDestIndexes>
struct transform_swizzle {
  /** @brief Alias for storing integer indexes of the source swizzled_vec into
   *        an integer pack
   * @tparam kSourceIndexes Variadic pack of integer indexes representing
   *         the swizzle indexes of the source swizzled_vec.
   */
  template <int... kSourceIndexes>
  using src_pack_t = swizzle_pack<kSourceIndexes...>;

  /** @brief Alias that performs the type transformation from one swizzled_vec
   *        to another
   * @tparam dataT The data type of the requested swizzled_vec type.
   * @tparam kElems The number of elements of the requested swizzled_vec type.
   * @tparam kSourceIndexes Variadic pack of integer indexes representing
   *         the swizzle indexes of the source swizzled_vec.
   */
  template <typename dataT, int kElems, int... kSourceIndexes>
  using return_t =
      transform_swizzled_vec_t<dataT, kElems, src_pack_t<kSourceIndexes...>,
                               kDestIndexes...>;

  /** @brief Transforms one swizzled_vec into another based on the destination
   *        indexes
   * @tparam dataT The data type of the requested swizzled_vec type.
   * @tparam kElems The number of elements of the requested swizzled_vec type.
   * @tparam kSourceIndexes Variadic pack of integer indexes representing
   *         the swizzle indexes of the source swizzled_vec.
   * @return New swizzled_vec swizzled based on the index transformation
   */
  template <typename dataT, int kElems, int... kSourceIndexes>
  static auto get(swizzled_vec<dataT, kElems, kSourceIndexes...> sourceVec)
      -> return_t<dataT, kElems, kSourceIndexes...> {
    return reinterpret_cast<return_t<dataT, kElems, kSourceIndexes...>&>(
        sourceVec);
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_COMMON_H_

////////////////////////////////////////////////////////////////////////////////
