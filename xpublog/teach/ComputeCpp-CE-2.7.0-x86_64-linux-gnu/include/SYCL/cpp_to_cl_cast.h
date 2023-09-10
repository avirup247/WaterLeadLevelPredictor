/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/
#ifndef RUNTIME_INCLUDE_SYCL_CPP_TO_CL_CAST_H_
#define RUNTIME_INCLUDE_SYCL_CPP_TO_CL_CAST_H_

#include "SYCL/addrspace_cast.h"
#include "SYCL/deduce.h"
#include "SYCL/half_type.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/type_traits.h"
#include "SYCL/type_traits_vec.h"
#include "SYCL/vec.h"
#include "computecpp/gsl/gsl"

namespace cl {
namespace sycl {
namespace detail {

/** @brief  Converts a cl::sycl::vec<T, N> type to the equivalent OpenCL type.
 * @tparam T A C++ integral type.
 * @tparam N The dimension of the vector.
 * @tparam U vec<cl_char, N> if T is signed char, or if T is char on an
 * implementation where std::is_signed<char>::value == true;
 *           vec<cl_uchar, N> if T is unsigned char, or if T is char on an
 * implementation where std::is_signed<char>::value == false;
 *           vec<cl_short, N> if T is a signed 16-bit integer;
 *           vec<cl_ushort, N> if T is an unsigned 16-bit integer;
 *           vec<cl_int, N> if T is a signed 32-bit integer;
 *           vec<cl_uint, N> if T is an unsigned 32-bit integer;
 *           vec<cl_long, N> if T is a signed 64-bit integer
 *           vec<cl_ulong, N> if T is an unsigned 64-bit integer
 * @param t The object to be converted.
 * @return An object of type U with an equivalent value to t.
 */
#if defined __SYCL_DEVICE_ONLY__
template <typename T, int N>
auto cpp_to_cl_cast(const cl::sycl::vec<T, N>& v) noexcept
    -> decltype(std::declval<cl::sycl::vec<deduce_type_t<T>, N>>().get_data()) {
  return v.template as<cl::sycl::vec<deduce_type_t<T>, N>>().get_data();
}
#else
template <typename T, int N>
auto cpp_to_cl_cast(const cl::sycl::vec<T, N>& v) noexcept
    -> cl::sycl::vec<deduce_type_t<T>, N> {
  return v.template as<cl::sycl::vec<deduce_type_t<T>, N>>();
}
#endif  // defined __SYCL_DEVICE_ONLY__

/** @brief  Converts a C++ fundamental type to the equivalent OpenCL type.
 * @tparam T A C++ integral type.
 * @tparam U cl_char if T is signed char, or if T is char on an implementation
 * where std::is_signed<char>::value == true;
 *           cl_uchar if T is unsigned char, or if T is char on an
 * implementation where std::is_signed<char>::value == false;
 *           cl_short if T is a signed 16-bit integer;
 *           cl_ushort if T is an unsigned 16-bit integer;
 *           cl_int if T is a signed 32-bit integer;
 *           cl_uint if T is an unsigned 32-bit integer;
 *           cl_long if T is a signed 64-bit integer
 *           cl_ulong if T is an unsigned 64-bit integer
 * @param t The object to be converted.
 * @return An object of type U with an equivalent value to t.
 */
template <typename T>
constexpr auto cpp_to_cl_cast(const T t) noexcept -> deduce_type_t<T> {
  return static_cast<deduce_type_t<T>>(t);
}

/** @brief Converts cl::sycl::vec<T, N> to the equivalent OpenCL type.
 *
 * Specific overload for cl::sycl::vec<cl::sycl::half, N>. This is an identity
 * function, as there is no other C++ half type in C++11.
 *
 * @tparam N The dimension of the vector.
 * @param v The object to be converted.
 * @return `v`
 */
#if defined __SYCL_DEVICE_ONLY__
template <int N>
constexpr auto cpp_to_cl_cast(
    const ::cl::sycl::vec<::cl::sycl::half, N>& v) noexcept
    -> decltype(v.get_data()) {
  return v.get_data();
}
#else
template <int N>
constexpr cl::sycl::vec<cl::sycl::half, N> cpp_to_cl_cast(
    const cl::sycl::vec<cl::sycl::half, N>& v) noexcept {
  return v;
}
#endif  // defined __SYCL_DEVICE_ONLY__

/** @brief  Converts a multi_ptr to a raw pointer, preserving the address space.
 * @tparam P Pointer type.
 * @tparam AddressSpace Pointer address space.
 * @param t The object to be converted.
 * @return The underlying pointer of p.
 */
#if defined __SYCL_DEVICE_ONLY__
template <typename T, int N, access::address_space AddressSpace>
auto cpp_to_cl_cast(
    ::cl::sycl::multi_ptr<::cl::sycl::vec<T, N>, AddressSpace> p) noexcept
    -> decltype(multi_ptr_get_internal_type(
        std::declval<multi_ptr<
            typename std::remove_pointer<decltype(p->get_data_ptr())>::type,
            AddressSpace>>())) {
  using P = typename std::remove_pointer<decltype(p->get_data_ptr())>::type;
  using multi_ptr_t = multi_ptr<P, AddressSpace>;
#ifdef __SYCL_COMPUTECPP_ASP__
  void_ptr_t<P> result = multi_ptr_get_internal_type<P, AddressSpace>(
      cl::sycl::make_ptr<P, AddressSpace>(p->get_data_ptr()));
  return detail::reinterpret_addrspace_cast<decltype(
      multi_ptr_get_internal_type(std::declval<multi_ptr_t>()))>(result);
#else   // Offload
  using P = typename std::remove_pointer<decltype(p->get_data_ptr())>::type;
  using multi_ptr_t = multi_ptr<P, AddressSpace>;
  void_ptr_t<P> result = multi_ptr_get_internal_type<P, AddressSpace>(
      cl::sycl::make_ptr<P, AddressSpace>(p->get_data_ptr()));
  return static_cast<decltype(
      multi_ptr_get_internal_type(std::declval<multi_ptr_t>()))>(result);
#endif  // __SYCL_COMPUTECPP_ASP__
}
#endif  // defined __SYCL_DEVICE_ONLY__

/** @brief Converts multi_ptr<P, AddressSpace> to a SYCL-appropriate P2*.
 *
 * Address space is preserved, but P may be transformed (e.g. from `int` to
 * `cl_int`).
 *
 * In the "no-asp" mode, this pointer will be qualified with an address space as
 * if that mode were disabled.
 * @tparam P The type that the multi_ptr points to.
 * @tparam AddressSpace The address-space that the pointee occupies.
 * @param p The object to cast.
 */
template <typename P, access::address_space AddressSpace>
auto cpp_to_cl_cast(multi_ptr<P, AddressSpace> p) noexcept -> decltype(
    multi_ptr_get_internal_type(std::declval<deduce_type_t<decltype(p)>>())) {
#ifdef __SYCL_COMPUTECPP_ASP__
  // First, remove CV qualifiers
  void_ptr_t<P> result = detail::reinterpret_addrspace_cast<void_ptr_t<P>>(
      multi_ptr_get_internal_type<P, AddressSpace>(p));
  // Then cast it to the appropriate type
  return detail::reinterpret_addrspace_cast<decltype(
      multi_ptr_get_internal_type(std::declval<deduce_type_t<decltype(p)>>()))>(
      result);
#else   // Offload
  void_ptr_t<P> result = multi_ptr_get_internal_type<P, AddressSpace>(p);
  return static_cast<decltype(multi_ptr_get_internal_type(
      std::declval<deduce_type_t<decltype(p)>>()))>(result);
#endif  // __SYCL_COMPUTECPP_ASP__
}

/** @brief Converts swizzled_vec<T, N> to the equivalent OpenCL type
 *
 * @tparam T The underlying type of the swizzled_vec
 * @tparam kElems The number of elements of the swizzled_vec
 * @tparam Indexes The indexes used to access the original vec
 *         of the swizzled_vec
 * @param v The object to be converted
 * @return swizzled_vec cast to an OpenCL type
 */
template <typename T, int kElems, int... Indexes>
auto cpp_to_cl_cast(
    const cl::sycl::swizzled_vec<T, kElems, Indexes...>& v) noexcept
    -> decltype(cpp_to_cl_cast(
        static_cast<detail::common_return_t<vec<T, sizeof...(Indexes)>,
                                            decltype(v)>>(v))) {
  return cpp_to_cl_cast(
      static_cast<
          detail::common_return_t<vec<T, sizeof...(Indexes)>, decltype(v)>>(v));
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl
#endif  // RUNTIME_INCLUDE_SYCL_CPP_TO_CL_CAST_H_
