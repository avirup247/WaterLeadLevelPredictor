/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
    @file cl_types.h

    @brief This file contains the vector types used for SYCL/OpenCL interop.

        @detail These are defined in the SYCL 1.2 specification and are residing
   in the
    cl::sycl namespace.
*/
#ifndef RUNTIME_INCLUDE_SYCL_CL_VEC_TYPES_H_
#define RUNTIME_INCLUDE_SYCL_CL_VEC_TYPES_H_

#include "SYCL/cl_types.h"
#include "SYCL/vec.h"

namespace cl {
namespace sycl {

/// @brief char vector interop type
using cl_char2 = vec<cl_char, 2>;
/// @brief char vector interop type
using cl_char3 = vec<cl_char, 3>;
/// @brief char vector interop type
using cl_char4 = vec<cl_char, 4>;
/// @brief char vector interop type
using cl_char8 = vec<cl_char, 8>;
/// @brief char vector interop type
using cl_char16 = vec<cl_char, 16>;

/// @brief unsigned char vector interop type
using cl_uchar2 = vec<cl_uchar, 2>;
/// @brief unsigned char vector interop type
using cl_uchar3 = vec<cl_uchar, 3>;
/// @brief unsigned char vector interop type
using cl_uchar4 = vec<cl_uchar, 4>;
/// @brief unsigned char vector interop type
using cl_uchar8 = vec<cl_uchar, 8>;
/// @brief unsigned char vector interop type
using cl_uchar16 = vec<cl_uchar, 16>;

/// @brief short vector interop type
using cl_short2 = vec<cl_short, 2>;
/// @brief short vector interop type
using cl_short3 = vec<cl_short, 3>;
/// @brief short vector interop type
using cl_short4 = vec<cl_short, 4>;
/// @brief short vector interop type
using cl_short8 = vec<cl_short, 8>;
/// @brief short vector interop type
using cl_short16 = vec<cl_short, 16>;

/// @brief unsigned short vector interop type
using cl_ushort2 = vec<cl_ushort, 2>;
/// @brief unsigned short vector interop type
using cl_ushort3 = vec<cl_ushort, 3>;
/// @brief unsigned short vector interop type
using cl_ushort4 = vec<cl_ushort, 4>;
/// @brief unsigned short vector interop type
using cl_ushort8 = vec<cl_ushort, 8>;
/// @brief unsigned short vector interop type
using cl_ushort16 = vec<cl_ushort, 16>;

/// @brief int vector interop type
using cl_int2 = vec<cl_int, 2>;
/// @brief int vector interop type
using cl_int3 = vec<cl_int, 3>;
/// @brief int vector interop type
using cl_int4 = vec<cl_int, 4>;
/// @brief int vector interop type
using cl_int8 = vec<cl_int, 8>;
/// @brief int vector interop type
using cl_int16 = vec<cl_int, 16>;

/// @brief unsigned int vector interop type
using cl_uint2 = vec<cl_uint, 2>;
/// @brief unsigned int vector interop type
using cl_uint3 = vec<cl_uint, 3>;
/// @brief unsigned int vector interop type
using cl_uint4 = vec<cl_uint, 4>;
/// @brief unsigned int vector interop type
using cl_uint8 = vec<cl_uint, 8>;
/// @brief unsigned int vector interop type
using cl_uint16 = vec<cl_uint, 16>;

/// @brief long vector interop type
using cl_long2 = vec<cl_long, 2>;
/// @brief long vector interop type
using cl_long3 = vec<cl_long, 3>;
/// @brief long vector interop type
using cl_long4 = vec<cl_long, 4>;
/// @brief long vector interop type
using cl_long8 = vec<cl_long, 8>;
/// @brief long vector interop type
using cl_long16 = vec<cl_long, 16>;

/// @brief unsigned long vector interop type
using cl_ulong2 = vec<cl_ulong, 2>;
/// @brief unsigned long vector interop type
using cl_ulong3 = vec<cl_ulong, 3>;
/// @brief unsigned long vector interop type
using cl_ulong4 = vec<cl_ulong, 4>;
/// @brief unsigned long vector interop type
using cl_ulong8 = vec<cl_ulong, 8>;
/// @brief unsigned long vector interop type
using cl_ulong16 = vec<cl_ulong, 16>;

/// @brief float vector interop type
using cl_float2 = vec<cl_float, 2>;
/// @brief float vector interop type
using cl_float3 = vec<cl_float, 3>;
/// @brief float vector interop type
using cl_float4 = vec<cl_float, 4>;
/// @brief float vector interop type
using cl_float8 = vec<cl_float, 8>;
/// @brief float vector interop type
using cl_float16 = vec<cl_float, 16>;

/// @brief double vector interop type
using cl_double2 = vec<cl_double, 2>;
/// @brief double vector interop type
using cl_double3 = vec<cl_double, 3>;
/// @brief double vector interop type
using cl_double4 = vec<cl_double, 4>;
/// @brief double vector interop type
using cl_double8 = vec<cl_double, 8>;
/// @brief double vector interop type
using cl_double16 = vec<cl_double, 16>;

/// @brief 2 dimensional half vector interop type
using cl_half2 = vec<cl_half, 2>;
/// @brief 3 dimensional half vector interop type
using cl_half3 = vec<cl_half, 3>;
/// @brief 4 dimensional half vector interop type
using cl_half4 = vec<cl_half, 4>;
/// @brief 8 dimensional half vector interop type
using cl_half8 = vec<cl_half, 8>;
/// @brief 16 dimensional half vector interop type
using cl_half16 = vec<cl_half, 16>;

#if defined(__SYCL_DEVICE_ONLY__)
namespace detail {

/// @brief float vector interop type
using cl_float2 = __sycl_vector<cl_float, 2>;
/// @brief float vector interop type
using cl_float3 = __sycl_vector<cl_float, 3>;
/// @brief float vector interop type
using cl_float4 = __sycl_vector<cl_float, 4>;
/// @brief float vector interop type
using cl_float8 = __sycl_vector<cl_float, 8>;
/// @brief float vector interop type
using cl_float16 = __sycl_vector<cl_float, 16>;

/// @brief double vector interop type
using cl_double2 = __sycl_vector<cl_double, 2>;
/// @brief double vector interop type
using cl_double3 = __sycl_vector<cl_double, 3>;
/// @brief double vector interop type
using cl_double4 = __sycl_vector<cl_double, 4>;
/// @brief double vector interop type
using cl_double8 = __sycl_vector<cl_double, 8>;
/// @brief double vector interop type
using cl_double16 = __sycl_vector<cl_double, 16>;

/// @brief char vector interop type
using cl_char2 = __sycl_vector<cl_char, 2>;
/// @brief char vector interop type
using cl_char3 = __sycl_vector<cl_char, 3>;
/// @brief char vector interop type
using cl_char4 = __sycl_vector<cl_char, 4>;
/// @brief char vector interop type
using cl_char8 = __sycl_vector<cl_char, 8>;
/// @brief char vector interop type
using cl_char16 = __sycl_vector<cl_char, 16>;

/// @brief unsigned char vector interop type
using cl_uchar2 = __sycl_vector<cl_uchar, 2>;
/// @brief unsigned char vector interop type
using cl_uchar3 = __sycl_vector<cl_uchar, 3>;
/// @brief unsigned char vector interop type
using cl_uchar4 = __sycl_vector<cl_uchar, 4>;
/// @brief unsigned char vector interop type
using cl_uchar8 = __sycl_vector<cl_uchar, 8>;
/// @brief unsigned char vector interop type
using cl_uchar16 = __sycl_vector<cl_uchar, 16>;

/// @brief short vector interop type
using cl_short2 = __sycl_vector<cl_short, 2>;
/// @brief short vector interop type
using cl_short3 = __sycl_vector<cl_short, 3>;
/// @brief short vector interop type
using cl_short4 = __sycl_vector<cl_short, 4>;
/// @brief short vector interop type
using cl_short8 = __sycl_vector<cl_short, 8>;
/// @brief short vector interop type
using cl_short16 = __sycl_vector<cl_short, 16>;

/// @brief unsigned short vector interop type
using cl_ushort2 = __sycl_vector<cl_ushort, 2>;
/// @brief unsigned short vector interop type
using cl_ushort3 = __sycl_vector<cl_ushort, 3>;
/// @brief unsigned short vector interop type
using cl_ushort4 = __sycl_vector<cl_ushort, 4>;
/// @brief unsigned short vector interop type
using cl_ushort8 = __sycl_vector<cl_ushort, 8>;
/// @brief unsigned short vector interop type
using cl_ushort16 = __sycl_vector<cl_ushort, 16>;

/// @brief int vector interop type
using cl_int2 = __sycl_vector<cl_int, 2>;
/// @brief int vector interop type
using cl_int3 = __sycl_vector<cl_int, 3>;
/// @brief int vector interop type
using cl_int4 = __sycl_vector<cl_int, 4>;
/// @brief int vector interop type
using cl_int8 = __sycl_vector<cl_int, 8>;
/// @brief int vector interop type
using cl_int16 = __sycl_vector<cl_int, 16>;

/// @brief unsigned int vector interop type
using cl_uint2 = __sycl_vector<cl_uint, 2>;
/// @brief unsigned int vector interop type
using cl_uint3 = __sycl_vector<cl_uint, 3>;
/// @brief unsigned int vector interop type
using cl_uint4 = __sycl_vector<cl_uint, 4>;
/// @brief unsigned int vector interop type
using cl_uint8 = __sycl_vector<cl_uint, 8>;
/// @brief unsigned int vector interop type
using cl_uint16 = __sycl_vector<cl_uint, 16>;

/// @brief long vector interop type
using cl_long2 = __sycl_vector<cl_long, 2>;
/// @brief long vector interop type
using cl_long3 = __sycl_vector<cl_long, 3>;
/// @brief long vector interop type
using cl_long4 = __sycl_vector<cl_long, 4>;
/// @brief long vector interop type
using cl_long8 = __sycl_vector<cl_long, 8>;
/// @brief long vector interop type
using cl_long16 = __sycl_vector<cl_long, 16>;

/// @brief unsigned long vector interop type
using cl_ulong2 = __sycl_vector<cl_ulong, 2>;
/// @brief unsigned long vector interop type
using cl_ulong3 = __sycl_vector<cl_ulong, 3>;
/// @brief unsigned long vector interop type
using cl_ulong4 = __sycl_vector<cl_ulong, 4>;
/// @brief unsigned long vector interop type
using cl_ulong8 = __sycl_vector<cl_ulong, 8>;
/// @brief unsigned long vector interop type
using cl_ulong16 = __sycl_vector<cl_ulong, 16>;

/// @brief half vector interop type
using cl_half2 = __sycl_vector<cl_half, 2>;
/// @brief half vector interop type
using cl_half3 = __sycl_vector<cl_half, 3>;
/// @brief half vector interop type
using cl_half4 = __sycl_vector<cl_half, 4>;
/// @brief half vector interop type
using cl_half8 = __sycl_vector<cl_half, 8>;
/// @brief half vector interop type
using cl_half16 = __sycl_vector<cl_half, 16>;

}  // namespace detail
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CL_TYPES_H_
