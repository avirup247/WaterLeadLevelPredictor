//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_GSL_DETAIL_TYPE_TRAITS_H_
#define RUNTIME_INCLUDE_GSL_DETAIL_TYPE_TRAITS_H_

#include <type_traits>
#include <utility>

namespace computecpp {
namespace gsl {

/** @brief See http://en.cppreference.com/w/cpp/types/add_pointer.
 */
template <typename T>
using add_pointer_t = typename std::add_pointer<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/decay.
 */
template <typename T>
using decay_t = typename std::decay<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/enable_if.
 */
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_const.
 */
template <typename T>
using remove_const_t = typename std::remove_const<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_pointer.
 */
template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_pointer.
 */
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

/** @brief Used for extracting value_type from a particular T, without needing
 * to rely on dependent type-names.
 */
template <typename T>
using value_type_t = typename T::value_type;

/** @brief Used for extracting pointer from a particular T, without needing to
 * rely on dependent type-names.
 */
template <typename T>
using pointer_t = typename T::pointer;

/** @brief Used for extracting the reference type a particular T.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <typename T>
using reference_t = decltype(*std::declval<T&>());

/** @brief See https://en.cppreference.com/w/cpp/types/conjunction.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <class...>
struct conjunction : std::true_type {};

/** @brief See https://en.cppreference.com/w/cpp/types/conjunction.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <class B1>
struct conjunction<B1> : B1 {};

/** @brief See https://en.cppreference.com/w/cpp/types/conjunction.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional<bool(B1::value), conjunction<Bn...>, B1>::type {};

/** @brief See https://en.cppreference.com/w/cpp/iterator/iter_t.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <typename T>
using iter_reference_t = decltype(*(std::declval<T>()));

/** @brief Used for extracting the return type of std::forward on a particular
 * T.
 * @note Definition is as it is to maintain consistency with upstream C++.
 */
template <typename T>
using forward_t = decltype(std::forward<T>(std::declval<T>()));

/** @brief Used to determine if an iterator type is writable.
 * @note Derived from http://eel.is/c++draft/iterator.concept.writable
 */
template <typename Iterator, typename T>
struct is_writable
    : conjunction<std::is_assignable<iter_reference_t<Iterator>, forward_t<T>>,
                  std::is_assignable<iter_reference_t<forward_t<Iterator>>,
                                     forward_t<T>>> {};

}  // namespace gsl
}  // namespace computecpp

#endif  // RUNTIME_INCLUDE_GSL_DETAIL_TYPE_TRAITS_H_
