/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file accessor_host_args.h
  @brief Internal file used by the @ref cl::sycl::accessor_base class
*/

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_HOST_ARGS_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_HOST_ARGS_H_

#include "SYCL/base.h"
#include "SYCL/index_array.h"

namespace cl {
namespace sycl {
namespace detail {

/*******************************************************************************
    host_accessor_args (host side)
*******************************************************************************/

/**

This file contains:
- @ref host_accessor_args
*/

/**
  @brief Structure to hold host side accessor arguments.
  This struct contains host only fields for accessors.
  Structures device_arg_container (device equivalent to host_accessor_args) has
  an attribute to tell the compiler to refer to this class when emitting the
  stubfile.
*/
struct host_arg_container {
  /**
  @brief Shared pointer to the internal detail accessor object.
  */
  daccessor_shptr m_impl;

  /**
  @brief Raw pointer to the host memory of the accessor.
  */
  void* m_hostDataPtr;

  /** Cached value of the stored range
   */
  index_array m_storeRange;
};

/******************************************************************************/

}  // namespace detail
}  // namespace sycl
}  // namespace cl

/******************************************************************************/

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_HOST_ARGS_H_

/******************************************************************************/
