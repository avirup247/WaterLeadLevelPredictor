/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/// `deduce_type_t` takes an arbitrary type and attempts to map it to a
/// `cl::sycl::cl_type`.
///
/// For example, `int` must have at least sixteen bits, but its width is
/// otherwise implementation-defined. deduce_type_t<int> will correctly deduce
/// whether `int` should be interpreted as `cl_short` (16-bit int), `cl_int`
/// (32-bit int), or `cl_long` (64-bit int).
///
/// `deduce_type_t` also correctly deduces pointer-to-cv-qualified types (such
/// as
/// `const int*`) and multi_ptr<T, AS>.
///
#ifndef RUNTIME_INCLUDE_SYCL_DEDUCE_H_
#define RUNTIME_INCLUDE_SYCL_DEDUCE_H_

#include "SYCL/cl_types.h"
#include "SYCL/type_traits.h"

namespace cl {
namespace sycl {
namespace detail {
/** @brief Deduces a compatible OpenCL type for a C++ type.
 *
 * Due to the added complexity, pointers-to-cv-qualified types are not handled
 * here, but are directly handled by `deduce_type`, which invokes
 * `deduce_type_impl`.
 *
 * @tparam T The C++ type.
 * @tparam Size the size of the type.
 * @note The primary template does not deduce a type.
 */
template <typename T, int Size>
struct deduce_type_impl {};

/** @brief Deduces a compatible OpenCL type for a C++ type.
 *
 * Pointers and references are not considered.
 * @tparam T The C++ type.
 */
template <typename T>
using deduce_type_impl_t =
    typename deduce_type_impl<T, sizeof(remove_pointer_t<T>)>::type;

/** @ref deduce_type_impl
 */
template <typename T, int Size>
struct deduce_type_impl<T&, Size> {
  using type = deduce_type_impl_t<T>&;
};

/** @ref deduce_type_impl
 */
template <typename T, int Size>
struct deduce_type_impl<const T, Size> {
  using type = const deduce_type_impl_t<remove_const_t<T>>;
};

/** @ref deduce_type_impl
 */
template <typename T, int Size>
struct deduce_type_impl<volatile T, Size> {
  using type = volatile deduce_type_impl_t<remove_volatile_t<T>>;
};

/** @ref deduce_type_impl
 */
template <typename T, int Size>
struct deduce_type_impl<const volatile T, Size> {
  using type = const volatile deduce_type_impl_t<remove_cv_t<T>>;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<char, 1> {
  using type = conditional_t<std::is_signed<char>::value, cl_char, cl_uchar>;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<signed char, 1> {
  using type = cl_char;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned char, 1> {
  using type = cl_uchar;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<short, 2> {
  using type = cl_short;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned short, 2> {
  using type = cl_ushort;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<int, 2> {
  using type = cl_short;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned int, 2> {
  using type = cl_ushort;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<int, 4> {
  using type = cl_int;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned int, 4> {
  using type = cl_uint;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<int, 8> {
  using type = cl_long;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned int, 8> {
  using type = cl_ulong;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<long, 4> {
  using type = cl_int;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned long, 4> {
  using type = cl_uint;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<long, 8> {
  using type = cl_long;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned long, 8> {
  using type = cl_ulong;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<long long, 8> {
  using type = cl_long;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<unsigned long long, 8> {
  using type = cl_ulong;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<half, 2> {
  using type = cl_half;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<float, 2> {
  using type = cl_half;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<float, 4> {
  using type = cl_float;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<double, 4> {
  using type = cl_float;
};

/** @ref deduce_type_impl
 */
template <>
struct deduce_type_impl<double, 8> {
  using type = cl_double;
};

/** @brief Convenience function to help implicit conversions be deducible using
 * deduce_type_impl_t.
 *
 * Due to implicit conversions, these overloads permit types that are compatible
 * with SYCL types to be deducible as if they are SYCL types.
 *
 * @note This function may only exist in unevaluated contexts, similarly to
 * std::declval.
 * @seealso std::declval.
 */
unsigned char deduce_type_impl_f(bool);
deduce_type_impl_t<char> deduce_type_impl_f(char);
deduce_type_impl_t<signed char> deduce_type_impl_f(signed char);
deduce_type_impl_t<unsigned char> deduce_type_impl_f(unsigned char);
deduce_type_impl_t<short> deduce_type_impl_f(short);
deduce_type_impl_t<unsigned short> deduce_type_impl_f(unsigned short);
deduce_type_impl_t<int> deduce_type_impl_f(int);
deduce_type_impl_t<unsigned int> deduce_type_impl_f(unsigned int);
deduce_type_impl_t<long> deduce_type_impl_f(long);
deduce_type_impl_t<unsigned long> deduce_type_impl_f(unsigned long);
deduce_type_impl_t<long long> deduce_type_impl_f(long long);
deduce_type_impl_t<unsigned long long> deduce_type_impl_f(unsigned long long);
deduce_type_impl_t<half> deduce_type_impl_f(half);
deduce_type_impl_t<float> deduce_type_impl_f(float);
deduce_type_impl_t<double> deduce_type_impl_f(double);

/** @brief Deduces a SYCL-friendly type from a SYCL-compatible type.
 *
 * For example, all 16-bit integers map to cl_short, and all pointers to const
 * 64-bit integers map to const cl_long*.
 */
template <typename T>
struct deduce_type;

template <typename T>
using deduce_type_t = typename deduce_type<T>::type;

/** @brief Deduces the type that is possibly not a C++ fundamental or SYCL type
 * as a compatible SYCL type.
 */
template <typename T>
using deduce_impl_t = decltype(deduce_type_impl_f(std::declval<T>()));

/** @brief Deduces a SYCL-friendly type from a SYCL-compatible type.
 *
 * For example, all 16-bit integers map to cl_short, and all pointers to const
 * 64-bit integers map to const cl_long*.
 */
template <typename T>
struct deduce_type {
  using type = deduce_impl_t<T>;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<T&> {
  using type = deduce_type_t<T>&;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<const T&> {
  using type = const deduce_type_t<T>&;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<volatile T&> {
  using type = volatile deduce_type_t<T>&;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<const volatile T&> {
  using type = const volatile deduce_type_t<T>&;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<T*> {
  using type = deduce_type_t<T>*;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<const T*> {
  using type = const deduce_type_t<T>*;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<volatile T*> {
  using type = volatile deduce_type_t<T>*;
};

/** @ref deduce_type
 */
template <typename T>
struct deduce_type<const volatile T*> {
  using type = const volatile deduce_type_t<T>*;
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEDUCE_H_
