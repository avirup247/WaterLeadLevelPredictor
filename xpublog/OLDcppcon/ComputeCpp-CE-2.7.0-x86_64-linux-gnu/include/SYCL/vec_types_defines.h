/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_types_defines.h

 @brief This file contains the vector types used only on SYCL host or on SYCL
 device.
 These are defined in the SYCL 1.2 specification and are residing in the
 cl::sycl namespace, but these types are not suggested for usage, as they
 cannot interop between host/device.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_TYPES_DEFINES_H_
#define RUNTIME_INCLUDE_SYCL_VEC_TYPES_DEFINES_H_

#include "SYCL/common.h"
#include "SYCL/half_type.h"
#include "SYCL/vec.h"

namespace cl {
namespace sycl {

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

// Vector types for the half data type, it is outside the
// __SYCL_DEVICE _ONLY statement because cl::sycl::half
// already contains its own #ifdef statement
using half2 = vec<half, 2>;
using half3 = vec<half, 3>;
using half4 = vec<half, 4>;
using half8 = vec<half, 8>;
using half16 = vec<half, 16>;
#ifndef __SYCL_DEVICE_ONLY__

using char2 = vec<char, 2>;
using char3 = vec<char, 3>;
using char4 = vec<char, 4>;
using char8 = vec<char, 8>;
using char16 = vec<char, 16>;

using schar2 = vec<signed char, 2>;
using schar3 = vec<signed char, 3>;
using schar4 = vec<signed char, 4>;
using schar8 = vec<signed char, 8>;
using schar16 = vec<signed char, 16>;

using uchar2 = vec<unsigned char, 2>;
using uchar3 = vec<unsigned char, 3>;
using uchar4 = vec<unsigned char, 4>;
using uchar8 = vec<unsigned char, 8>;
using uchar16 = vec<unsigned char, 16>;

using short2 = vec<short, 2>;
using short3 = vec<short, 3>;
using short4 = vec<short, 4>;
using short8 = vec<short, 8>;
using short16 = vec<short, 16>;

using ushort2 = vec<unsigned short, 2>;
using ushort3 = vec<unsigned short, 3>;
using ushort4 = vec<unsigned short, 4>;
using ushort8 = vec<unsigned short, 8>;
using ushort16 = vec<unsigned short, 16>;

using int2 = vec<int, 2>;
using int3 = vec<int, 3>;
using int4 = vec<int, 4>;
using int8 = vec<int, 8>;
using int16 = vec<int, 16>;

using uint2 = vec<unsigned int, 2>;
using uint3 = vec<unsigned int, 3>;
using uint4 = vec<unsigned int, 4>;
using uint8 = vec<unsigned int, 8>;
using uint16 = vec<unsigned int, 16>;

using long2 = vec<long, 2>;
using long3 = vec<long, 3>;
using long4 = vec<long, 4>;
using long8 = vec<long, 8>;
using long16 = vec<long, 16>;

using ulong2 = vec<unsigned long, 2>;
using ulong3 = vec<unsigned long, 3>;
using ulong4 = vec<unsigned long, 4>;
using ulong8 = vec<unsigned long, 8>;
using ulong16 = vec<unsigned long, 16>;

using longlong2 = vec<long long, 2>;
using longlong3 = vec<long long, 3>;
using longlong4 = vec<long long, 4>;
using longlong8 = vec<long long, 8>;
using longlong16 = vec<long long, 16>;

using ulonglong2 = vec<unsigned long long, 2>;
using ulonglong3 = vec<unsigned long long, 3>;
using ulonglong4 = vec<unsigned long long, 4>;
using ulonglong8 = vec<unsigned long long, 8>;
using ulonglong16 = vec<unsigned long long, 16>;

#else
using char2 = vec<char, 2>;
using char3 = vec<char, 3>;
using char4 = vec<char, 4>;
using char8 = vec<char, 8>;
using char16 = vec<char, 16>;

using schar2 = vec<signed char, 2>;
using schar3 = vec<signed char, 3>;
using schar4 = vec<signed char, 4>;
using schar8 = vec<signed char, 8>;
using schar16 = vec<signed char, 16>;

using uchar2 = vec<unsigned char, 2>;
using uchar3 = vec<unsigned char, 3>;
using uchar4 = vec<unsigned char, 4>;
using uchar8 = vec<unsigned char, 8>;
using uchar16 = vec<unsigned char, 16>;

using short2 = vec<short, 2>;
using short3 = vec<short, 3>;
using short4 = vec<short, 4>;
using short8 = vec<short, 8>;
using short16 = vec<short, 16>;

using ushort2 = vec<unsigned short, 2>;
using ushort3 = vec<unsigned short, 3>;
using ushort4 = vec<unsigned short, 4>;
using ushort8 = vec<unsigned short, 8>;
using ushort16 = vec<unsigned short, 16>;

using int2 = vec<int, 2>;
using int3 = vec<int, 3>;
using int4 = vec<int, 4>;
using int8 = vec<int, 8>;
using int16 = vec<int, 16>;

using uint2 = vec<unsigned int, 2>;
using uint3 = vec<unsigned int, 3>;
using uint4 = vec<unsigned int, 4>;
using uint8 = vec<unsigned int, 8>;
using uint16 = vec<unsigned int, 16>;

using long2 = vec<long, 2>;
using long3 = vec<long, 3>;
using long4 = vec<long, 4>;
using long8 = vec<long, 8>;
using long16 = vec<long, 16>;

using ulong2 = vec<unsigned long, 2>;
using ulong3 = vec<unsigned long, 3>;
using ulong4 = vec<unsigned long, 4>;
using ulong8 = vec<unsigned long, 8>;
using ulong16 = vec<unsigned long, 16>;

using longlong2 = vec<long long, 2>;
using longlong3 = vec<long long, 3>;
using longlong4 = vec<long long, 4>;
using longlong8 = vec<long long, 8>;
using longlong16 = vec<long long, 16>;

using ulonglong2 = vec<unsigned long long, 2>;
using ulonglong3 = vec<unsigned long long, 3>;
using ulonglong4 = vec<unsigned long long, 4>;
using ulonglong8 = vec<unsigned long long, 8>;
using ulonglong16 = vec<unsigned long long, 16>;

#endif  //  __SYCL_DEVICE_ONLY__

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_VEC_TYPES_DEFINES_H_
