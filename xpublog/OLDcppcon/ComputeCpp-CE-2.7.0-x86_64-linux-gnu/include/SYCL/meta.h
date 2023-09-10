/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/
#ifndef RUNTIME_INCLUDE_SYCL_META_H_
#define RUNTIME_INCLUDE_SYCL_META_H_

#include "SYCL/cl_types.h"
#include "SYCL/vec.h"

namespace cl {
namespace sycl {
namespace detail {
/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_short double_width_cast(const cl_char from) noexcept {
  return static_cast<cl_short>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_short, N> double_width_cast(
    const ::cl::sycl::vec<cl_char, N> from) noexcept {
  return from.template convert<cl_short, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_ushort double_width_cast(const cl_uchar from) noexcept {
  return static_cast<cl_ushort>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_ushort, N> double_width_cast(
    const ::cl::sycl::vec<cl_uchar, N> from) noexcept {
  return from.template convert<cl_ushort, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_int double_width_cast(const cl_short from) noexcept {
  return static_cast<cl_int>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_int, N> double_width_cast(
    const ::cl::sycl::vec<cl_short, N> from) noexcept {
  return from.template convert<cl_int, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_uint double_width_cast(const cl_ushort from) noexcept {
  return static_cast<cl_uint>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_uint, N> double_width_cast(
    const ::cl::sycl::vec<cl_ushort, N> from) noexcept {
  return from.template convert<cl_uint, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_long double_width_cast(const cl_int from) noexcept {
  return static_cast<cl_long>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_long, N> double_width_cast(
    const ::cl::sycl::vec<cl_int, N> from) noexcept {
  return from.template convert<cl_long, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_ulong double_width_cast(const cl_uint from) noexcept {
  return static_cast<cl_ulong>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_ulong, N> double_width_cast(
    const ::cl::sycl::vec<cl_uint, N> from) noexcept {
  return from.template convert<cl_ulong, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_long double_width_cast(const cl_long from) noexcept {
  return static_cast<cl_long>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_long, N> double_width_cast(
    const ::cl::sycl::vec<cl_long, N> from) noexcept {
  return from;
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_ulong double_width_cast(const cl_ulong from) noexcept {
  return static_cast<cl_ulong>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_ulong, N> double_width_cast(
    const ::cl::sycl::vec<cl_ulong, N> from) noexcept {
  return from;
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
inline cl_float double_width_cast(const half from) noexcept {
  return static_cast<cl_float>(from);
}

template <int N>
inline ::cl::sycl::vec<cl_float, N> double_width_cast(
    const ::cl::sycl::vec<half, N> from) noexcept {
  return from.template convert<cl_float, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_double double_width_cast(const cl_float from) noexcept {
  return static_cast<cl_double>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_double, N> double_width_cast(
    const ::cl::sycl::vec<cl_float, N> from) noexcept {
  return from.template convert<cl_double, rounding_mode::automatic>();
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
constexpr cl_double double_width_cast(const cl_double from) noexcept {
  return static_cast<cl_double>(from);
}

/** @brief Promotes the type of the input parameter to a type with double its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with double the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_double, N> double_width_cast(
    const ::cl::sycl::vec<cl_double, N> from) noexcept {
  return from;
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_char halve_width_cast(const cl_char from) noexcept {
  return static_cast<cl_char>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_char, N> halve_width_cast(
    const ::cl::sycl::vec<cl_char, N> from) noexcept {
  return from;
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_uchar halve_width_cast(const cl_uchar from) noexcept {
  return static_cast<cl_uchar>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_uchar, N> halve_width_cast(
    const ::cl::sycl::vec<cl_uchar, N> from) noexcept {
  return from;
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_char halve_width_cast(const cl_short from) noexcept {
  return static_cast<cl_char>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_char, N> halve_width_cast(
    const ::cl::sycl::vec<cl_short, N> from) noexcept {
  return from.template convert<cl_char, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_uchar halve_width_cast(const cl_ushort from) noexcept {
  return static_cast<cl_uchar>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_uchar, N> halve_width_cast(
    const ::cl::sycl::vec<cl_ushort, N> from) noexcept {
  return from.template convert<cl_uchar, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_short halve_width_cast(const cl_int from) noexcept {
  return static_cast<cl_short>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_short, N> halve_width_cast(
    const ::cl::sycl::vec<cl_int, N> from) noexcept {
  return from.template convert<cl_short, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_ushort halve_width_cast(const cl_uint from) noexcept {
  return static_cast<cl_ushort>(from);
}

template <int N>
inline ::cl::sycl::vec<cl_ushort, N> halve_width_cast(
    const ::cl::sycl::vec<cl_uint, N> from) noexcept {
  return from.template convert<cl_ushort, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_int halve_width_cast(const cl_long from) noexcept {
  return static_cast<cl_int>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_int, N> halve_width_cast(
    const ::cl::sycl::vec<cl_long, N> from) noexcept {
  return from.template convert<cl_int, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_uint halve_width_cast(const cl_ulong from) noexcept {
  return static_cast<cl_uint>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_uint, N> halve_width_cast(
    const ::cl::sycl::vec<cl_ulong, N> from) noexcept {
  return from.template convert<cl_uint, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
inline half halve_width_cast(const half from) noexcept {
  return static_cast<half>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<half, N> halve_width_cast(
    const ::cl::sycl::vec<half, N> from) noexcept {
  return from.template convert<half, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
inline half halve_width_cast(const cl_float from) noexcept {
  return static_cast<half>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<half, N> halve_width_cast(
    const ::cl::sycl::vec<cl_float, N> from) noexcept {
  return from.template convert<half, rounding_mode::automatic>();
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
constexpr cl_float halve_width_cast(const cl_double from) noexcept {
  return static_cast<cl_float>(from);
}

/** @brief Narrows the type of the input parameter to a type with half its
 * width.
 * @param from The type to convert.
 * @return Same data, promoted to a type with half the width.
 */
template <int N>
inline ::cl::sycl::vec<cl_float, N> halve_width_cast(
    const ::cl::sycl::vec<cl_double, N> from) noexcept {
  return from.template convert<cl_float, rounding_mode::automatic>();
}
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_META_H_
