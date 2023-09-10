/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
    @file cl_types.h

    @brief This file contains the scalar types used for SYCL/OpenCL interop.

        @detail These are defined in the SYCL 1.2 specification and are residing
   in the
    cl::sycl namespace.
*/

#ifndef RUNTIME_INCLUDE_SYCL_CL_TYPES_H_
#define RUNTIME_INCLUDE_SYCL_CL_TYPES_H_

#include "SYCL/common.h"
#include "SYCL/half_type.h"

namespace cl {
namespace sycl {

/// @brief Boolean interop type
using cl_bool = bool;
/// @brief char (1 byte) interop type
using cl_char = signed char;
/// @brief unsigned char (1 byte) interop type
using cl_uchar = unsigned char;
/// @brief short (2 bytes) interop type
using cl_short = int16_t;
/// @brief unsigned short (2 bytes) interop type
using cl_ushort = uint16_t;
/// @brief int (4 bytes) interop type
using cl_int = int;
/// @brief unsigned int (4 bytes) interop type
using cl_uint = unsigned int;
/// @brief float interop type
using cl_float = float;
/// @brief double interop type
using cl_double = double;

/// @brief long (8 bytes) interop type
using cl_long = int64_t;
/// @brief unsigned long (8 bytes) interop type
using cl_ulong = uint64_t;

/// @brief half scalar type
using cl_half = half;

#if defined(__SYCL_DEVICE_ONLY__)

namespace detail {

/// @brief char (1 byte) interop type
using cl_char = signed char;
/// @brief unsigned char (1 byte) interop type
using cl_uchar = unsigned char;
/// @brief short (2 byte) interop type
using cl_short = int16_t;
/// @brief unsigned short (2 byte) interop type
using cl_ushort = uint16_t;
/// @brief int (4 byte) interop type
using cl_int = int;
/// @brief unsigned int (4 byte) interop type
using cl_uint = unsigned int;
/// @brief long (8 byte) interop type
using cl_long = int64_t;
/// @brief unsigned long (8 byte) interop type
using cl_ulong = uint64_t;
/// @brief half interop type
using cl_half = half;
/// @brief float interop type
using cl_float = float;
/// @brief double interop type
using cl_double = double;

}  // namespace detail
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CL_TYPES_H_
