/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_macros.h

 @brief This file contains some common internal helper macros used by @ref
 cl::sycl::vec to define swizzle operations.
*/

#ifndef RUNTIME_INCLUDE_SYCL_VEC_MACROS_H_
#define RUNTIME_INCLUDE_SYCL_VEC_MACROS_H_

#include "SYCL/common.h"

////////////////////////////////////////////////////////////////////////////////

/* The COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1 macro takes 1, parameter representing
   a swizzle index and constructs a simple swizzled method. The macro is used
   within the different specializations of the mem_container class to define the
   simple swizzle methods based on the index parameter that is passed to the
   macro. The simple swizzle methods take no arguments and return a reference to
   a swizzled_vec object that is templated by the swizzle index value. This is
   done by performing a reinterpret_cast on the this pointer from a vec type to
   a swizzled_vec type. This is possible due to both the vec type and the
   swizzled_vec type deriving from the same base class with no additional
   members, therefore making both types identical in data layout but different
   in functionality. The index values are assigned using the const variables
   defined in the detail namespace.

   For example COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(x) would create the
   method
   'swizzled_vec<dataT, kElems, 0> &x()' which performs a cast
   from 'vec<datat, kElems>' to 'swizzled_vec<dataT, kElems, 0>'
*/

/** @cond COMPUTECPP_DEV */
#define COMPUTECPP_DEFINE_SIMPLE_SWIZZLE_1(s0)                                 \
  swizzled_vec<dataT, kElems, detail::s0>& s0() {                              \
    auto swizzledVec =                                                         \
        reinterpret_cast<swizzled_vec<dataT, kElems, detail::s0>*>(this);      \
    return *swizzledVec;                                                       \
  }                                                                            \
                                                                               \
  const swizzled_vec<dataT, kElems, detail::s0>& s0() const {                  \
    auto swizzledVec =                                                         \
        reinterpret_cast<const swizzled_vec<dataT, kElems, detail::s0>*>(      \
            this);                                                             \
    return *swizzledVec;                                                       \
  }

/** COMPUTECPP_DEV @endcond */

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_MACROS_H_

////////////////////////////////////////////////////////////////////////////////
