/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file abacus_types.h
 *
 * @brief This file contains the scalar and vector types used for SYCL
 * host/abacus interop. The SYCL host is following the C++ standard definitions
 * for primitive types and abacus is following the OpenCL C 1.2 standard
 * definitions. These types reside in the abacus namespace.
 */
#ifndef RUNTIME_INCLUDE_SYCL_ABACUS_TYPES_H_
#define RUNTIME_INCLUDE_SYCL_ABACUS_TYPES_H_

#ifndef __SYCL_DEVICE_ONLY__

#include "SYCL/cl_types.h"
#include "SYCL/cl_vec_types.h"

namespace abacus {

#include "abacus/abacus_config"  // NOLINT due to this header specifically
                                 // is added here so that all the types
                                 // are defined within the abacus
                                 // namespace

using cl_char = abacus_char;
using cl_uchar = abacus_uchar;
using cl_short = abacus_short;
using cl_ushort = abacus_ushort;
using cl_int = abacus_int;
using cl_uint = abacus_uint;
using cl_long = abacus_long;
using cl_ulong = abacus_ulong;
using cl_float = abacus_float;
using cl_double = abacus_double;

using cl_char2 = abacus_char2;
using cl_char3 = abacus_char3;
using cl_char4 = abacus_char4;
using cl_char8 = abacus_char8;
using cl_char16 = abacus_char16;

using cl_uchar2 = abacus_uchar2;
using cl_uchar3 = abacus_uchar3;
using cl_uchar4 = abacus_uchar4;
using cl_uchar8 = abacus_uchar8;
using cl_uchar16 = abacus_uchar16;

using cl_short2 = abacus_short2;
using cl_short3 = abacus_short3;
using cl_short4 = abacus_short4;
using cl_short8 = abacus_short8;
using cl_short16 = abacus_short16;

using cl_ushort2 = abacus_ushort2;
using cl_ushort3 = abacus_ushort3;
using cl_ushort4 = abacus_ushort4;
using cl_ushort8 = abacus_ushort8;
using cl_ushort16 = abacus_ushort16;

using cl_int2 = abacus_int2;
using cl_int3 = abacus_int3;
using cl_int4 = abacus_int4;
using cl_int8 = abacus_int8;
using cl_int16 = abacus_int16;

using cl_uint2 = abacus_uint2;
using cl_uint3 = abacus_uint3;
using cl_uint4 = abacus_uint4;
using cl_uint8 = abacus_uint8;
using cl_uint16 = abacus_uint16;

using cl_long2 = abacus_vector<abacus_long, 2>;
using cl_long3 = abacus_vector<abacus_long, 3>;
using cl_long4 = abacus_vector<abacus_long, 4>;
using cl_long8 = abacus_vector<abacus_long, 8>;
using cl_long16 = abacus_vector<abacus_long, 16>;

using cl_ulong2 = abacus_vector<abacus_ulong, 2>;
using cl_ulong3 = abacus_vector<abacus_ulong, 3>;
using cl_ulong4 = abacus_vector<abacus_ulong, 4>;
using cl_ulong8 = abacus_vector<abacus_ulong, 8>;
using cl_ulong16 = abacus_vector<abacus_ulong, 16>;

using cl_float2 = abacus_float2;
using cl_float3 = abacus_float3;
using cl_float4 = abacus_float4;
using cl_float8 = abacus_float8;
using cl_float16 = abacus_float16;

using cl_double2 = abacus_double2;
using cl_double3 = abacus_double3;
using cl_double4 = abacus_double4;
using cl_double8 = abacus_double8;
using cl_double16 = abacus_double16;

template <typename T>
struct convert_abacus_sycl;

template <>
struct convert_abacus_sycl<char> {
  using abacus_type = char;
  using abacus_utype = unsigned char;
  using sycl_utype = cl::sycl::cl_uchar;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_char> {
  using abacus_type = signed char;
  using abacus_utype = unsigned char;
  using sycl_utype = cl::sycl::cl_uchar;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_uchar> {
  using abacus_type = unsigned char;
  using abacus_utype = unsigned char;
  using sycl_utype = cl::sycl::cl_uchar;
};

template <>
struct convert_abacus_sycl<bool> : convert_abacus_sycl<unsigned char> {};

template <>
struct convert_abacus_sycl<cl::sycl::cl_short> {
  using abacus_type = abacus_short;
  using abacus_utype = abacus_ushort;
  using sycl_utype = cl::sycl::cl_ushort;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_ushort> {
  using abacus_type = abacus_ushort;
  using abacus_utype = abacus_ushort;
  using sycl_utype = cl::sycl::cl_ushort;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_uint> {
  using abacus_type = abacus_uint;
  using abacus_utype = abacus_uint;
  using sycl_utype = cl::sycl::cl_uint;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_int> {
  using abacus_type = int;
  using abacus_utype = unsigned int;
  using sycl_utype = cl::sycl::cl_uint;
};

template <>
struct convert_abacus_sycl<long> {
  using abacus_type = cl_long;
  using abacus_utype = cl_ulong;
  using sycl_utype = cl::sycl::cl_ulong;
};

template <>
struct convert_abacus_sycl<long long> {
  using abacus_type = cl_long;
  using abacus_utype = cl_ulong;
  using sycl_utype = cl::sycl::cl_ulong;
};

template <>
struct convert_abacus_sycl<unsigned long> {
  using abacus_type = abacus_ulong;
  using abacus_utype = abacus_ulong;
  using sycl_utype = cl::sycl::cl_ulong;
};

template <>
struct convert_abacus_sycl<unsigned long long> {
  using abacus_type = abacus_ulong;
  using abacus_utype = abacus_ulong;
  using sycl_utype = cl::sycl::cl_ulong;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_float> {
  using abacus_type = abacus_float;
  using abacus_utype = abacus_float;
  using sycl_utype = cl::sycl::cl_float;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_double> {
  using abacus_type = abacus_double;
  using abacus_utype = abacus_double;
  using sycl_utype = cl::sycl::cl_double;
};

template <>
struct convert_abacus_sycl<cl::sycl::cl_half> {
  using abacus_type = cl::sycl::cl_float;
  using abacus_utype = cl::sycl::cl_float;
  using sycl_utype = cl::sycl::cl_float;
};

template <typename T, int dims>
struct convert_abacus_sycl<cl::sycl::vec<T, dims>> {
 private:
  using converted_t = convert_abacus_sycl<T>;

 public:
  using abacus_type = abacus_vector<typename converted_t::abacus_type, dims>;
  using abacus_utype = abacus_vector<typename converted_t::abacus_utype, dims>;
  using sycl_utype = cl::sycl::vec<typename converted_t::sycl_utype, dims>;
};

/** @brief Converts a SYCL type to an Abacus type.
 * @tparam T Any SYCL type.
 */
template <typename T>
struct sycl_to_abacus;

/** @brief Converts an Abacus type to a SYCL type.
 * @tparam T Any Abacus type.
 */
template <typename T>
struct abacus_to_sycl;

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_char> {
  using type = abacus_char;
};

/** @ref sycl_to_abacus.
 */
template <>
struct abacus_to_sycl<abacus_char> {
  using type = cl::sycl::cl_char;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_uchar> {
  using type = abacus_uchar;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_uchar> {
  using type = cl::sycl::cl_uchar;
};

/** @ref abacus_to_sycl.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_short> {
  using type = abacus_short;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_short> {
  using type = cl::sycl::cl_short;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_ushort> {
  using type = abacus_ushort;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_ushort> {
  using type = cl::sycl::cl_ushort;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_int> {
  using type = abacus_int;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_int> {
  using type = cl::sycl::cl_int;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_uint> {
  using type = abacus_uint;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_uint> {
  using type = cl::sycl::cl_uint;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_long> {
  using type = abacus_long;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_long> {
  using type = cl::sycl::cl_long;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_ulong> {
  using type = abacus_ulong;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_ulong> {
  using type = cl::sycl::cl_ulong;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_half> {
  using type = abacus_float;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_float> {
  using type = abacus_float;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_float> {
  using type = cl::sycl::cl_float;
};

/** @ref sycl_to_abacus.
 */
template <>
struct sycl_to_abacus<cl::sycl::cl_double> {
  using type = abacus_double;
};

/** @ref abacus_to_sycl.
 */
template <>
struct abacus_to_sycl<abacus_double> {
  using type = cl::sycl::cl_double;
};

/** @ref sycl_to_abacus, specialized for vec
 */
template <typename T, int N>
struct sycl_to_abacus<cl::sycl::vec<T, N>> {
  using type = abacus_vector<typename sycl_to_abacus<T>::type, N>;
};

/** @ref sycl_to_abacus, specialized for 1-elem vec
 */
template <typename T>
struct sycl_to_abacus<cl::sycl::vec<T, 1>> : sycl_to_abacus<T> {};

/** @ref abacus_to_sycl, specialized for abacus_vector
 */
template <typename T, int N>
struct abacus_to_sycl<abacus_vector<T, N>> {
  using type = cl::sycl::vec<typename abacus_to_sycl<T>::type, N>;
};

/** @ref sycl_to_abacus.
 */
template <typename T>
using sycl_to_abacus_t = typename sycl_to_abacus<T>::type;

/** @ref abacus_to_sycl.
 */
template <typename T>
using abacus_to_sycl_t = typename sycl_to_abacus<T>::type;
}  // namespace abacus

#endif  // __SYCL_DEVICE_ONLY__
#endif  //  RUNTIME_INCLUDE_SYCL_ABACUS_TYPES_H_
