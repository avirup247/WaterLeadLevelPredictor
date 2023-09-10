/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file type_traits.h

  @brief Provides type traits from future C++ versions in C++11, as well as some
         in-house type traits
  @note This header is part of the implementation of the SYCL library and cannot
  be used independently.
*/
#ifndef RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_H_
#define RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_H_

#include "SYCL/half_type.h"

#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {

/** @brief See https://en.cppreference.com/w/cpp/types/integral_constant.
 */
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

/** @brief See http://en.cppreference.com/w/cpp/types/enable_if.
 */
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/conditional.
 */
template <bool B, typename T1, typename T2>
using conditional_t = typename std::conditional<B, T1, T2>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/make_signed.
 */
template <typename T>
using make_signed_t = typename std::make_signed<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/make_unsigned.
 */
template <typename T>
using make_unsigned_t = typename std::make_unsigned<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/negation.
 */
template <typename T>
using negation = std::integral_constant<bool, !T::value>;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_cv.
 */
template <typename T>
using remove_const_t = typename std::remove_const<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_cv.
 */
template <typename T>
using remove_volatile_t = typename std::remove_volatile<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_cv.
 */
template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_reference.
 */
template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_cvref.
 */
template <typename T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

/** @brief See http://en.cppreference.com/w/cpp/types/remove_pointer.
 */
template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/underlying_type.
 */
template <typename T>
using underlying_type_t = typename std::underlying_type<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/decay.
 */
template <typename T>
using decay_t = typename std::decay<T>::type;

/** @brief See http://en.cppreference.com/w/cpp/types/void_t.
 */
template <typename...>
using void_t = void;

/////////// Codeplay-exclusive type_traits

/** @brief Extracts an element_type typename from an arbitrary type.
 * @tparam T The type to extract `element_type` from.
 * @tparam void Used to detect if typename `T::element_type` is a valid type.
 * @note If `typename T::element_type` is not a valid type, then `element_type`
 *       does not have a member.
 */
template <typename T, typename = void>
struct get_element_type {};

/** @brief Extracts an element_type typename from an arbitrary type.
 *        Specialization for when `typename T::element_type` is a valid type.
 * @tparam T The type to extract `element_type` from.
 */
template <typename T>
struct get_element_type<T, void_t<typename T::element_type>> {
  using type = typename T::element_type;
};

template <typename T>
using get_element_type_t = typename get_element_type<T>::type;

/** @brief Unary type function with a return type determined by whether or not
 *        the input is a signed type or not.
 *
 * @tparam T1 input type
 * @tparam T2 returned type if T1 is a signed type
 * @tparam T3 returned type if T1 is not a signed type
 */
template <typename T1, typename T2, typename T3>
using deduce_signedness_t =
    typename std::conditional<std::is_signed<T1>::value, T2, T3>::type;

/** @brief A library-based emulation of C++17 fold expressions
 */
template <bool...>
struct bool_pack {};

/** @brief Checks to see if T1 is present in the parameter pack.
 * @return false if T1 is present in the paramter pack, true otherwise.
 * @tparam T1 The subject type.
 * @tparam Ts... A list of types to be compared against.
 */
// TODO(Chris): Removed to avoid compiler bug in MSVC.
template <typename T1, typename... Ts>
using is_none_of =
    std::is_same<bool_pack<false, std::is_same<T1, Ts>::value...>,
                 bool_pack<std::is_same<T1, Ts>::value..., false>>;

/** @brief Checks to see if T1 is present in the parameter pack.
 * @tparam T1 The subject type.
 * @tparam Ts... A list of types to be compared against.
 */
template <typename T1, typename... Ts>
using is_one_of =
    negation<std::is_same<bool_pack<false, std::is_same<T1, Ts>::value...>,
                          bool_pack<std::is_same<T1, Ts>::value..., false>>>;

/** @brief Enables T1 if it is the same type as T2.
 * @tparam T1 The subject type.
 * @tparam T2 The objective type.
 */
template <typename T1, typename T2>
using requires_is_same_t = enable_if_t<std::is_same<T1, T2>::value, T1>;

/** @brief Checks whether the type is convertible to and from a half
 * @tparam T Type to check
 */
template <typename T>
using is_half_convertible =
    std::integral_constant<bool, (std::is_convertible<T, half>::value &&
                                  std::is_convertible<half, T>::value)>;

/** @brief Checks whether T is a custom half type
 *        - either half or convertible to it
 * @tparam T Type to check
 */
template <typename T>
using is_custom_half_type =
    std::integral_constant<bool, (std::is_same<half, decay_t<T>>::value ||
                                  (!std::is_floating_point<T>::value &&
                                   !std::is_integral<T>::value &&
                                   is_half_convertible<T>::value))>;

/** @brief Returns half if T is convertible to half
 * @tparam T Type to convert
 */
template <typename T>
struct common_half_type {
  using type =
      typename std::enable_if<cl::sycl::detail::is_half_convertible<T>::value,
                              half>::type;
};

/** @brief Returns half if T is convertible to half
 * @tparam T Type to convert
 */
template <typename T>
using common_half_type_t = typename common_half_type<T>::type;

/** @brief Helper struct to retrieve the common type of multiple types
 * @tparam Ts Types to find the common type for
 */
template <typename... Ts>
struct common_type_helper {
  using type = typename std::common_type<Ts...>::type;
};

/** @brief Helper struct to retrieve the common type of multiple types,
 *        specialization for half as first parameter
 * @tparam T Type used to find find common_type<half, T>
 */
template <typename T>
struct common_type_helper<half, T> : common_half_type<T> {};

/** @brief Helper struct to retrieve the common type of multiple types,
 *        specialization for half as second parameter
 * @tparam T Type used to find find common_type<T, half>
 */
template <typename T>
struct common_type_helper<T, half> : common_type_helper<half, T> {};

/** @brief Helper struct to retrieve the common type of multiple types,
 *        specialization for half as both parameters
 */
template <>
struct common_type_helper<half, half> {
  using type = half;
};

/** @brief Finds the common type of multiple types
 * @tparam Ts Types to find the common type for
 */
template <typename... Ts>
using common_type_t = typename common_type_helper<Ts...>::type;

/** @brief Checks whether the types are the same, ignoring references and
 *        cv-qualifiers
 * @tparam First The first type
 * @tparam Second The second type
 */
template <typename First, typename Second>
struct is_same_basic_type : std::is_same<decay_t<First>, decay_t<Second>> {};

/** @brief Deduces the cv-qualifiers necessary for a decaying a pointer to
 * void*.
 *
 * Given that `void* foo = bar` is not possible when `bar` is a
 * pointer-to-cv-qualified type, `void_ptr` deduces the correct type.
 * @note The primary template handles T* as well.
 */
template <typename T>
struct void_ptr {
  using type = void*;
};

/** @brief Deduces decay to void* for const types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<const T> {
  using type = const void*;
};

/** @brief Deduces decay to void* for pointer-to-const types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<const T*> {
  using type = const void*;
};

/** @brief Deduces decay to void* for volatile types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<volatile T> {
  using type = volatile void*;
};

/** @brief Deduces decay to void* for pointer-to-volatile types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<volatile T*> {
  using type = volatile void*;
};

/** @brief Deduces decay to void* for const volatile types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<const volatile T> {
  using type = const volatile void*;
};

/** @brief Deduces decay to void* for pointer-to-const-volatile types.
 * @seealso void_ptr<T>.
 */
template <typename T>
struct void_ptr<const volatile T*> {
  using type = const volatile void*;
};

template <typename T>
using void_ptr_t = typename void_ptr<T>::type;

/** @brief Checks if the provided type is a contiguous container. In that
 * - std::data(container) and std::size(container) are well formed
 * - return type of std::data(container) is convertible to T*
 */
template <typename T, typename Container>
using is_contiguous_container =
    detail::void_t<detail::enable_if_t<std::is_convertible<
                       detail::remove_pointer_t<decltype(
                           std::declval<Container>().data())> (*)[],
                       const T (*)[]>::value>,
                   decltype(std::declval<Container>().size())>;

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#if defined(__cpp_if_constexpr) && (__cpp_if_constexpr >= 201606)
#define COMPUTECPP_IF_CONSTEXPR_HELPER(cond) if constexpr (cond)
#else
#define COMPUTECPP_IF_CONSTEXPR_HELPER(cond)                                   \
  if (cl::sycl::detail::bool_constant<(cond)>::value)
#endif  // __cpp_if_constexpr

/** if statement where the condition is evaluated at compile time
 */
#define COMPUTECPP_IF_CONSTEXPR(cond) COMPUTECPP_IF_CONSTEXPR_HELPER(cond)

#endif  // RUNTIME_INCLUDE_SYCL_TYPE_TRAITS_H_
