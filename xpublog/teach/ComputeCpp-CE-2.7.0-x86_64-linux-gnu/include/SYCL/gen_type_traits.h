/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file gen_type_traits.h

  @brief Determines if a type is fit for being used in a built-in function.
*/
#ifndef RUNTIME_INCLUDE_SYCL_GEN_TYPE_TRAITS_H_
#define RUNTIME_INCLUDE_SYCL_GEN_TYPE_TRAITS_H_

#include "SYCL/half_type.h"
#include "SYCL/type_traits.h"
#include "SYCL/type_traits_vec.h"
#include <limits>
#include <utility>

namespace cl {
namespace sycl {
namespace detail {
namespace builtin {
/** @brief Deduces a trait based on a condition. A type 'is' something if the
 * condition is true; false otherwise.
 * @tparam B A Boolean condition.
 * @return std::true_type if B is true; std::false_type otherwise.
 */
template <bool B>
using is = std::integral_constant<bool, B>;

static_assert(is<true>::value, "is<true>::value should return true");

/** @brief Checks to see if N represents a size of a geometric vector,
 *        i.e. N is one of {2, 3, 4}.
 * @tparam N Size to check
 */
template <int N>
struct is_geometric_size
    : std::integral_constant<bool, ((N == 2) || (N == 3) || (N == 4))> {};

/** @brief Checks to see if V is a SYCL vector<T, N>, where N is one of {2, 3,
 * 4}.
 * @tparam V The type being inspected.
 * @tparam T The expected value_type of the vector.
 * @returns true if V is a SYCL vector<T, N>; false otherwise.
 */
template <typename V, typename T>
struct is_geovec : std::false_type {};

template <typename T, int N>
struct is_geovec<cl::sycl::vec<T, N>, T> : is_geometric_size<N> {};

template <typename T, int kElems, int... Indexes>
struct is_geovec<cl::sycl::swizzled_vec<T, kElems, Indexes...>, T>
    : is_geometric_size<sizeof...(Indexes)> {};

/** @brief Checks that T is an OpenCL scalar
 * @tparam T The parameterised type being inspected.
 * @tparam E An explicit type used to make sure that T is the Expected type. For
 * example, this type is both a floating-point _and_ a scalar.
 */
template <typename T, typename E>
struct is_scalar : std::false_type {};

template <>
struct is_scalar<cl::sycl::half, cl::sycl::half> : std::true_type {};

template <typename T>
struct is_scalar<T, cl::sycl::half> : is_custom_half_type<T> {};

template <>
struct is_scalar<float, float> : std::true_type {};

template <>
struct is_scalar<double, double> : std::true_type {};

template <>
struct is_scalar<char, char> : std::true_type {};

template <>
struct is_scalar<signed char, signed char> : std::true_type {};

template <>
struct is_scalar<unsigned char, unsigned char> : std::true_type {};

template <>
struct is_scalar<short, short> : std::true_type {};

template <>
struct is_scalar<unsigned short, unsigned short> : std::true_type {};

template <>
struct is_scalar<int, int> : std::true_type {};

template <>
struct is_scalar<unsigned int, unsigned int> : std::true_type {};

template <>
struct is_scalar<long, long> : std::true_type {};

template <>
struct is_scalar<unsigned long, unsigned long> : std::true_type {};

template <>
struct is_scalar<long long, long long> : std::true_type {};

template <>
struct is_scalar<unsigned long long, unsigned long long> : std::true_type {};

template <typename T>
struct is_scalar<vec<T, 1>, T> : std::true_type {};

template <typename T, int kElems, int s0>
struct is_scalar<swizzled_vec<T, kElems, s0>, T> : std::true_type {};

/** @brief Checks to see if V is a SYCL vector<T, N>, where N is one of {2, 3,
 * 4, 8, 16}.
 * @tparam V The type being inspected.
 * @tparam T The expected value_type of the vector.
 * @returns true if V is a SYCL vector<T, N>; false otherwise.
 */
template <typename V, typename T>
struct is_vec : is_geovec<V, T> {};

template <typename T>
struct is_vec<cl::sycl::vec<T, 8>, T> : std::true_type {};

template <typename T>
struct is_vec<cl::sycl::vec<T, 16>, T> : std::true_type {};

template <typename T, int kElems, int... Indexes>
struct is_vec<cl::sycl::swizzled_vec<T, kElems, Indexes...>, T>
    : is_vec<vec<T, sizeof...(Indexes)>, T> {};

/** @brief Checks if a type is either a valid vector or a valid scalar.
 *
 * @tparam T actual type
 * @tparam E expected underlying type
 */
template <typename T, typename E, typename = void>
struct is_gen : std::false_type {};

template <typename T, typename E>
struct is_gen<T, E, enable_if_t<is_vec<T, E>::value>> : std::true_type {};

template <typename T, typename E>
struct is_gen<T, E, enable_if_t<is_scalar<T, E>::value>> : std::true_type {};

/** @brief Checks that T is a vector of floats.
 * @tparam The type to be checked.
 */
template <typename T>
using is_floatn = is_vec<T, float>;

/** @brief Checks that T is a float scalar or float vector.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_genfloatf = is_gen<T, float>;

/** @brief Checks that T is a vector of doubles.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_doublen = is_vec<T, double>;

/** @brief Checks that T is a double scalar or double vector.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_genfloatd = is_gen<T, double>;

/** @brief Checks that T is a vector of half-precision floating-point numbers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_halfn = is_vec<T, half>;

/** @brief Checks that T is a half scalar or half vector.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_genfloath = is_gen<T, half>;

/** @brief Checks that T models one of genfloatf, genfloatd, or genfloath.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genfloat : std::false_type {};

template <typename T>
struct is_genfloat<T, enable_if_t<is_genfloath<T>::value>> : std::true_type {};

template <typename T>
struct is_genfloat<T, enable_if_t<is_genfloatf<T>::value>> : std::true_type {};

template <typename T>
struct is_genfloat<T, enable_if_t<is_genfloatd<T>::value>> : std::true_type {};

/** @brief Checks that T is a half scalar, or float scalar, or double scalar.
 * @tparam T The type to be checked.
 */
template <typename T1, typename T2, typename = void>
struct is_sgenfloat : std::false_type {};

template <typename T2>
struct is_sgenfloat<half, T2,
                    enable_if_t<std::is_same<half, scalar_type_t<T2>>::value>>
    : std::true_type {};
template <typename T2>
struct is_sgenfloat<float, T2,
                    enable_if_t<std::is_same<float, scalar_type_t<T2>>::value>>
    : std::true_type {};
template <typename T2>
struct is_sgenfloat<double, T2,
                    enable_if_t<std::is_same<double, scalar_type_t<T2>>::value>>
    : std::true_type {};

/** @brief Checks that T models gengeovec or is a scalar for cl::sycl::half.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_gengeohalf : std::false_type {};

template <typename T>
struct is_gengeohalf<T, enable_if_t<is_geovec<T, half>::value>>
    : std::true_type {};

template <>
struct is_gengeohalf<half, void> : std::true_type {};

/** @brief Checks that T models gengeovec or is a scalar for float.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_gengeofloat : std::false_type {};

template <typename T>
struct is_gengeofloat<T, enable_if_t<is_geovec<T, float>::value>>
    : std::true_type {};

template <>
struct is_gengeofloat<float, void> : std::true_type {};

/** @brief Checks that T models gengeovec or is a scalar for double.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_gengeodouble : std::false_type {};

template <typename T>
struct is_gengeodouble<T, enable_if_t<is_geovec<T, double>::value>>
    : std::true_type {};

template <>
struct is_gengeodouble<double, void> : std::true_type {};

/** @brief Checks whether type T satisfies any of the gengeo conditions
 *        for floating point types (half, float, or double)
 * @tparam T Type to check
 */
template <typename T>
struct is_gen_geo_anyfloat
    : std::integral_constant<bool,
                             (detail::builtin::is_gengeohalf<T>::value ||
                              detail::builtin::is_gengeofloat<T>::value ||
                              detail::builtin::is_gengeodouble<T>::value)> {};

/** @brief Checks that T is a vector of char.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_charn = is_vec<T, char>;

/** @brief Checks that T is a vector of signed char.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_scharn = is_vec<T, signed char>;

/** @brief Checks that T is a vector of unsigned char.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ucharn = is_vec<T, unsigned char>;

/** @brief Checks that T is a vector of signed char or is a signed char scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_igenchar : std::false_type {};

template <typename T>
struct is_igenchar<T, enable_if_t<is_scharn<T>::value>> : std::true_type {};

template <>
struct is_igenchar<signed char, void> : std::true_type {};

/** @brief Checks that T is a vector of unsigned char or is a unsigned char
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenchar : std::false_type {};

template <typename T>
struct is_ugenchar<T, enable_if_t<is_ucharn<T>::value>> : std::true_type {};

template <>
struct is_ugenchar<unsigned char, void> : std::true_type {};

/** @brief Checks that T models one of is_charn, igenchar, ugenchar, or is
 * a char scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genchar : std::false_type {};

template <typename T>
struct is_genchar<T, enable_if_t<is_charn<T>::value>> : std::true_type {};

template <typename T>
struct is_genchar<T, enable_if_t<is_igenchar<T>::value>> : std::true_type {};

template <typename T>
struct is_genchar<T, enable_if_t<is_ugenchar<T>::value>> : std::true_type {};

template <>
struct is_genchar<char, void> : std::true_type {};

/** @brief Checks that T is a vector of short integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_shortn = is_vec<T, short>;

/** @brief Checks that T either models shortn or is a short integer scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genshort : std::false_type {};

template <typename T>
struct is_genshort<T, enable_if_t<is_shortn<T>::value>> : std::true_type {};

template <>
struct is_genshort<short, void> : std::true_type {};

/** @brief Checks that T is a vector of unsigned short integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ushortn = is_vec<T, unsigned short>;

/** @brief Checks that T either models ushortn or is a unsigned short integer
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenshort : std::false_type {};

template <typename T>
struct is_ugenshort<T, enable_if_t<is_ushortn<T>::value>> : std::true_type {};

template <>
struct is_ugenshort<unsigned short, void> : std::true_type {};

/** @brief Checks that T is a vector of unsigned integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_uintn = is_vec<T, unsigned int>;

/** @brief Checks that T either models uintn or is an unsigned integer scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenint : std::false_type {};

template <typename T>
struct is_ugenint<T, enable_if_t<is_uintn<T>::value>> : std::true_type {};

template <>
struct is_ugenint<unsigned int, void> : std::true_type {};

/** @brief Checks that T is a vector of integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_intn = is_vec<T, int>;

/** @brief Checks that T either models intn or is a integer scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genint : std::false_type {};

template <typename T>
struct is_genint<T, enable_if_t<is_intn<T>::value>> : std::true_type {};

template <>
struct is_genint<int, void> : std::true_type {};

/** @brief Checks that T is a vector of unsigned long integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ulongn = is_vec<T, unsigned long>;

/** @brief Checks that T either models ulongn or is an unsigned long integer
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenlong : std::false_type {};

template <typename T>
struct is_ugenlong<T, enable_if_t<is_ulongn<T>::value>> : std::true_type {};

template <>
struct is_ugenlong<unsigned long, void> : std::true_type {};

/** @brief Checks that T is a vector of long integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_longn = is_vec<T, long>;

/** @brief Checks that T either models longn or is a long integer
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genlong : std::false_type {};

template <typename T>
struct is_genlong<T, enable_if_t<is_longn<T>::value>> : std::true_type {};

template <>
struct is_genlong<long, void> : std::true_type {};

/** @brief Checks that T is a vector of unsigned long long integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ulonglongn = is_vec<T, unsigned long long>;

/** @brief Checks that T either models ulonglongn or is an unsigned long long
 * integer
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenlonglong : std::false_type {};

template <typename T>
struct is_ugenlonglong<T, enable_if_t<is_ulonglongn<T>::value>>
    : std::true_type {};

template <>
struct is_ugenlonglong<unsigned long long, void> : std::true_type {};

/** @brief Checks that T is a vector of long long integers.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_longlongn = is_vec<T, long long>;

/** @brief Checks that T either models longlongn or is an long long integer
 * scalar.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_genlonglong : std::false_type {};

template <typename T>
struct is_genlonglong<T, enable_if_t<is_longlongn<T>::value>> : std::true_type {
};

template <>
struct is_genlonglong<long long, void> : std::true_type {};

/** @brief Checks that T models either is_genlong or is_genlonglong.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_igenlonginteger : std::false_type {};

template <typename T>
struct is_igenlonginteger<T, enable_if_t<is_genlong<T>::value>>
    : std::true_type {};

template <typename T>
struct is_igenlonginteger<T, enable_if_t<is_genlonglong<T>::value>>
    : std::true_type {};

/** @brief Checks that T models either is_ugenlong or is_ugenlonglong.
 * @tparam T The type to be checked.
 */
template <typename T, typename = void>
struct is_ugenlonginteger : std::false_type {};

template <typename T>
struct is_ugenlonginteger<T, enable_if_t<is_ugenlong<T>::value>>
    : std::true_type {};

template <typename T>
struct is_ugenlonginteger<T, enable_if_t<is_ugenlonglong<T>::value>>
    : std::true_type {};

/** @brief Number of bits in a SYCL byte.
 */
constexpr auto char_bit = 8;

/** @brief Determines the number of bits in a type.
 * @tparam T The type to be checked.
 */
template <typename T, bool = std::is_scalar<T>::value>
struct bitsize_of : std::integral_constant<std::size_t, sizeof(T) * char_bit> {
};

/** @ref See primary template for bitsize_of.
 */
template <typename T>
struct bitsize_of<T, false>
    : std::integral_constant<std::size_t,
                             sizeof(typename T::element_type) * char_bit> {};

// NOTE: is_geninteger and is_genintegerNbit defined below

/** @brief Checks that T models one of igenchar, igenshort, igenint, or
 * igenlonginteger.
 * @tparam The type to be checked.
 */
template <typename T, typename = void>
struct is_igeninteger : std::false_type {};

template <typename T>
struct is_igeninteger<T, enable_if_t<is_igenchar<T>::value>> : std::true_type {
};

template <typename T>
struct is_igeninteger<T, enable_if_t<is_genshort<T>::value>> : std::true_type {
};

template <typename T>
struct is_igeninteger<T, enable_if_t<is_genint<T>::value>> : std::true_type {};

template <typename T>
struct is_igeninteger<T, enable_if_t<is_igenlonginteger<T>::value>>
    : std::true_type {};

/** @brief Checks that T models igeninteger and has exactly N bits.
 * @tparam T The type to be checked.
 * @tparam N The number of bits in T.
 */
template <typename T, int N, typename = void>
struct is_igenintegerNbit : std::false_type {};

template <typename T, int N>
struct is_igenintegerNbit<
    T, N, enable_if_t<is_igeninteger<T>::value && bitsize_of<T>::value == N>>
    : std::true_type {};

/** @brief Checks that T models igeninteger and has exactly 8 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_igeninteger8bit = is_igenintegerNbit<T, 8>;

/** @brief Checks that T models igeninteger and has exactly 16 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_igeninteger16bit = is_igenintegerNbit<T, 16>;

/** @brief Checks that T models igeninteger and has exactly 32 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_igeninteger32bit = is_igenintegerNbit<T, 32>;

/** @brief Checks that T models igeninteger and has exactly 64 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_igeninteger64bit = is_igenintegerNbit<T, 64>;

/** @brief Checks that T models one of ugenchar, ugenshort, ugenint, or
 * ugenlonginteger.
 * @tparam The type to be checked.
 */
template <typename T, typename = void>
struct is_ugeninteger : std::false_type {};

template <typename T>
struct is_ugeninteger<T, enable_if_t<is_ugenchar<T>::value>> : std::true_type {
};

template <typename T>
struct is_ugeninteger<T, enable_if_t<is_ugenshort<T>::value>> : std::true_type {
};

template <typename T>
struct is_ugeninteger<T, enable_if_t<is_ugenint<T>::value>> : std::true_type {};

template <typename T>
struct is_ugeninteger<T, enable_if_t<is_ugenlonginteger<T>::value>>
    : std::true_type {};

/** @brief Checks that T models ugeninteger and has exactly N bits.
 * @tparam T The type to be checked.
 * @tparam N The number of bits in T.
 */
template <typename T, int N, typename = void>
struct is_ugenintegerNbit : std::false_type {};

template <typename T, int N>
struct is_ugenintegerNbit<
    T, N, enable_if_t<is_ugeninteger<T>::value && bitsize_of<T>::value == N>>
    : std::true_type {};

/** @brief Checks that T models ugeninteger and has exactly 8 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ugeninteger8bit = is_ugenintegerNbit<T, 8>;

/** @brief Checks that T models ugeninteger and has exactly 16 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ugeninteger16bit = is_ugenintegerNbit<T, 16>;

/** @brief Checks that T models ugeninteger and has exactly 32 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ugeninteger32bit = is_ugenintegerNbit<T, 32>;

/** @brief Checks that T models ugeninteger and has exactly 64 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_ugeninteger64bit = is_ugenintegerNbit<T, 64>;

/** @brief Checks that T models either is_genchar, igeninteger, or ugeninteger.
 * @tparam
 */
template <typename T>
using is_geninteger = is<is_genchar<T>::value || is_igeninteger<T>::value ||
                         is_ugeninteger<T>::value>;

/** @brief Checks that T models geninteger and has exactly N bits.
 * @tparam T The type to be checked.
 * @tparam N The number of bits in T.
 */
template <typename T, int N>
using is_genintegerNbit =
    is<is_geninteger<T>::value && bitsize_of<T>::value == N>;

/** @brief Checks that T models geninteger and has exactly 8 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_geninteger8bit = is_genintegerNbit<T, 8>;

/** @brief Checks that T models geninteger and has exactly 16 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_geninteger16bit = is_genintegerNbit<T, 16>;

/** @brief Checks that T models geninteger and has exactly 32 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_geninteger32bit = is_genintegerNbit<T, 32>;

/** @brief Checks that T models geninteger and has exactly 64 bits.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_geninteger64bit = is_genintegerNbit<T, 64>;

/** @brief Checks that T models geninteger and scalar.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_sgeninteger = is<is_geninteger<T>::value && std::is_scalar<T>::value>;

/** @brief Checks that T models genfloat or geninteger.
 * @tparam T The type to be checked.
 */
template <typename T>
using is_gentype = is<is_genfloat<T>::value || is_geninteger<T>::value>;

/** @brief Checks that Types model Test.
 * @tparam Test The test to conduct.
 * @tparam Types... The types to be checked.
 */
template <template <typename...> class Test, typename... Types>
struct all_xsigned_geninteger {
  static constexpr auto value = true;
};

/** @ref See primary template.
 */
template <template <typename...> class is_xsigned, typename T,
          typename... Types>
struct all_xsigned_geninteger<is_xsigned, T, Types...> {
  static constexpr auto value =
      is_xsigned<T>::value &&
      all_xsigned_geninteger<is_xsigned, Types...>::value;
};

/** @brief Checks that Types model igeninteger.
 * @tparam Types... The types to be checked.
 */
template <typename... Types>
using all_igeninteger = all_xsigned_geninteger<is_igeninteger, Types...>;

/** @brief Checks that Types model ugeninteger.
 * @tparam Types... The types to be checked.
 */
template <typename... Types>
using all_ugeninteger = all_xsigned_geninteger<is_ugeninteger, Types...>;

template <typename... Types>
struct have_same_width : std::true_type {};

template <typename T1, typename T2, typename... Types>
struct have_same_width<T1, T2, Types...>
    : std::integral_constant<bool,
                             bitsize_of<T1>::value == bitsize_of<T2>::value &&
                                 have_same_width<Types...>::value> {};
}  // namespace builtin
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_GEN_TYPE_TRAITS_H_
