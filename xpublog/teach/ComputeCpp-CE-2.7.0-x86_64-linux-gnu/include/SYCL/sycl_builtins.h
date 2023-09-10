/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file sycl_builtins.h

  @brief This file is an unified header file which includes all required header
  files for the sycl built-ins functions.

  @note This header is part of the implementation of the SYCL library and cannot
  be used independently. The SYCL library entry point is sycl/sycl.hpp.
*/

#ifndef RUNTIME_INCLUDE_SYCL_SYCL_BUILTINS_H_
#define RUNTIME_INCLUDE_SYCL_SYCL_BUILTINS_H_

/// @cond COMPUTECPP_DEV
#define ABACUS_LIBRARY_STATIC 1
/// COMPUTECPP_DEV @endcond

#include "SYCL/cpp_to_cl_cast.h"
#include "SYCL/gen_type_traits.h"
#include "SYCL/type_traits.h"

#ifdef signbit
#warning "C99 Macro definition of signbit found. " \
    " This conflicts with SYCL builtins so its disabled."
#undef signbit
#endif  // signbit

#include "SYCL/builtins/math_common.h"
#include "SYCL/builtins/math_floating_point.h"
#include "SYCL/builtins/math_fp_half_precision.h"
#include "SYCL/builtins/math_fp_native.h"
#include "SYCL/builtins/math_geometric.h"
#include "SYCL/builtins/math_integral.h"
#include "SYCL/builtins/math_relational.h"

#endif  // RUNTIME_INCLUDE_SYCL_SYCL_BUILTINS_H_
