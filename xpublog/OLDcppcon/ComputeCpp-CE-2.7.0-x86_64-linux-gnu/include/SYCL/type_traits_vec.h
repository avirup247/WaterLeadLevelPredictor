/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file type_traits_vec.h

  @brief Provides type traits from future C++ versions in C++11, as well as some
         in-house type traits
  @note This header is part of the implementation of the SYCL library and cannot
  be used independently.
*/
#ifndef RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_VEC_H_
#define RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_VEC_H_

#include "SYCL/type_traits.h"
#include "SYCL/vec.h"
#include <type_traits>

namespace cl {
namespace sycl {

template <typename dataT, int kElems>
class vec;
template <typename dataT, int kElems, int... sourceTN>
class swizzled_vec;

namespace detail {

/** @brief Extracts the value type for a vec, matches the type for a scalar.
 * @tparam T Type of the scalar
 */
template <typename T>
struct scalar_type {
  using type = T;
};

/** @brief Specialization for vec class
 * @tparam T Underlying type of the vec
 * @tparam N Number of vector elements
 */
template <typename T, int N>
struct scalar_type<::cl::sycl::vec<T, N>> {
  using type = T;
};

/** @brief Specialization for swizzled_vec class
 * @tparam T Underlying type of the swizzled_vec
 * @tparam kElems Number of original vec elements
 * @tparam Indexes The indexes used when accessing the original vec
 */
template <typename T, int kElems, int... Indexes>
struct scalar_type<::cl::sycl::swizzled_vec<T, kElems, Indexes...>> {
  using type = T;
};

/** @brief Retrieves the underlying type of a vec
 * @tparam T Type to collapse to a scalar
 */
template <typename T>
using scalar_type_t = typename scalar_type<T>::type;

/** @brief Retrieves the underlying vec from a swizzled_vec
 * @param V Type to collapse to a vec
 */
template <typename V>
struct collapse_swizzled_vec {
  /** @brief In the general case, just return the same type
   */
  using type = V;
};

/** @brief Retrieves the underlying vec from a swizzled_vec,
 *        specialization for swizzled_vec
 * @tparam T Underlying type of the swizzled_vec
 * @tparam kElems Number of original vec elements
 * @tparam Indexes The indexes used when accessing the original vec
 */
template <typename T, int kElems, int... Indexes>
struct collapse_swizzled_vec<swizzled_vec<T, kElems, Indexes...>> {
  /** @brief swizzled_vec collapses to a vec
   */
  using type = vec<T, sizeof...(Indexes)>;
};

/** @brief Retrieves the underlying vec from a swizzled_vec,
 *        specialization for 1-elem swizzled_vec
 * @tparam T Underlying type of the swizzled_vec
 * @tparam kElems Number of original vec elements
 * @tparam s0 The index used when accessing the original vec
 */
template <typename T, int kElems, int s0>
struct collapse_swizzled_vec<swizzled_vec<T, kElems, s0>> {
  /** @brief 1-elem swizzled_vec collapses to a scalar
   */
  using type = T;
};

/** @brief Transforms a 1-elem vec into a scalar
 * @tparam T Underlying type of the vec
 */
template <typename T>
struct collapse_swizzled_vec<vec<T, 1>> {
  /** @brief 1-elem vec collapses to a scalar
   */
  using type = T;
};

/** @brief Retrieves the underlying vec from a swizzled_vec
 * @tparam V Type to collapse to a vec
 */
template <typename V>
using collapse_swizzled_vec_t = typename collapse_swizzled_vec<V>::type;

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        When mixing vectors and scalars, a vector will be returned.
 *        Otherwise, common_type will be used.
 *
 * @tparam Ts Types to find the common return type for
 */
template <typename... Ts>
struct common_return_helper {
  using type = common_type_t<collapse_swizzled_vec_t<Ts>...>;
};

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        Specialization for vec as first parameter, underlying type of the vec
 *        as second parameter.
 *
 * @tparam T Underlying type of the vec
 * @tparam N Number of vector elements
 */
template <class T, int N>
struct common_return_helper<vec<T, N>, T> : common_return_helper<vec<T, N>> {};

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        Specialization for underlying type of the vec as first parameter,
 *        vec as second parameter.
 *
 * @tparam T Underlying type of the vec
 * @tparam N Number of vector elements
 */
template <class T, int N>
struct common_return_helper<T, vec<T, N>> : common_return_helper<vec<T, N>> {};

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        Specialization for getting the common return type
 *        of two 1-elem swizzled_vec parameters.
 *
 * @tparam T Underlying type of the swizzled_vec
 * @tparam kElemsX Number of original vec elements of the first swizzled_vec
 * @tparam s0x The index used when accessing the original vec
 *         of the first swizzled_vec
 * @tparam kElemsY Number of original vec elements of the second swizzled_vec
 * @tparam s0y The index used when accessing the original vec
 *         of the second swizzled_vec
 * @note Common return type is a scalar
 */
template <class T, int kElemsX, int kElemsY, int s0x, int s0y>
struct common_return_helper<swizzled_vec<T, kElemsX, s0x>,
                            swizzled_vec<T, kElemsY, s0y>>
    : common_return_helper<T> {};

#ifndef __SYCL_DEVICE_ONLY__

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        Host specialization for getting the common return type
 *        of a 1-elem vec parameter and a 1-elem swizzled_vec.
 *
 * @tparam T Underlying type of the vec and swizzled_vec
 * @tparam kElems Number of original vec elements of the swizzled_vec
 * @tparam s0 The index used when accessing the original vec
 *         of the swizzled_vec
 * @note Common return type is a scalar
 */
template <class T, int kElems, int s0>
struct common_return_helper<vec<T, 1>, swizzled_vec<T, kElems, s0>>
    : common_return_helper<T> {};

/** @brief Helper struct to retrieve the common return type of multiple types.
 *
 *        Host specialization for getting the common return type
 *        of a 1-elem swizzled_vec and a 1-elem vec parameter.
 *
 * @tparam T Underlying type of the vec and swizzled_vec
 * @tparam kElems Number of original vec elements of the swizzled_vec
 * @tparam s0 The index used when accessing the original vec
 *         of the swizzled_vec
 * @note Common return type is a scalar
 */
template <class T, int kElems, int s0>
struct common_return_helper<swizzled_vec<T, kElems, s0>, vec<T, 1>>
    : common_return_helper<T> {};

#endif  // __SYCL_DEVICE_ONLY__

/** @brief Retrieves the common return type of multiple types.
 *
 *        When mixing vectors and scalars, a vector will be returned.
 *        Otherwise, common_type will be used.
 *
 * @tparam Ts Types to find the common return type for
 */
template <typename... Ts>
using common_return = common_return_helper<decay_t<Ts>...>;

/** @brief Retrieves the common return type of multiple types.
 *
 *        When mixing vectors and scalars, a vector will be returned.
 *        Otherwise, common_type will be used.
 *
 * @tparam Ts Types to find the common return type for
 */
template <typename... Ts>
using common_return_t = typename common_return<Ts...>::type;

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_VEC_H_
