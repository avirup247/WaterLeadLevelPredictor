/*****************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/**
  @file host_compiler_macros.h

  @brief Contains macros related to the host compiler
 */
#ifndef RUNTIME_INCLUDE_SYCL_HOST_COMPILER_MACROS_H_
#define RUNTIME_INCLUDE_SYCL_HOST_COMPILER_MACROS_H_

/// @cond COMPUTECPP_DEV

#ifdef _MSC_VER
#define COMPUTECPP_PRAGMA_HELPER(message) __pragma(message)
#else
#define COMPUTECPP_PRAGMA_HELPER(message) _Pragma(#message)
#endif  // _MSC_VER

/** @brief Provides a way to specify pragmas
 */
#define COMPUTECPP_PRAGMA(message) COMPUTECPP_PRAGMA_HELPER(message)

// The following block provides the these macros:
// COMPUTECPP_HOST_COMPILER_STRING - Name of the compiler used
// COMPUTECPP_HOST_CXX_DIAGNOSTIC - Provides a way to specify host diagnostics
//    for any compiler
// COMPUTECPP_GNU_CXX_DIAGNOSTIC - Diagnostics for GNU-like compilers
//    (GCC and Clang)
// COMPUTECPP_GCC_CXX_DIAGNOSTIC - Diagnostics for GCC
// COMPUTECPP_CLANG_CXX_DIAGNOSTIC - Diagnostics for Clang
// COMPUTECPP_MSVC_CXX_DIAGNOSTIC - Diagnostics for Visual Studio compiler

#if defined(__clang__)

#define COMPUTECPP_HOST_COMPILER_STRING "clang " __clang_version__
#define COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)                                \
  COMPUTECPP_PRAGMA(clang diagnostic message)
#define COMPUTECPP_GNU_CXX_DIAGNOSTIC(message)                                 \
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)
#define COMPUTECPP_CLANG_CXX_DIAGNOSTIC(message)                               \
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)

#elif defined(__GNUC__) || defined(__GNUG__)

#define COMPUTECPP_HOST_COMPILER_STRING "gcc " __VERSION__
#define COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)                                \
  COMPUTECPP_PRAGMA(GCC diagnostic message)
#define COMPUTECPP_GNU_CXX_DIAGNOSTIC(message)                                 \
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)
#define COMPUTECPP_GCC_CXX_DIAGNOSTIC(message)                                 \
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)

#elif defined(_MSC_VER)

#define COMPUTECPP_STRINGIFY_HELPER(x) #x
#define COMPUTECPP_STRINGIFY(x) COMPUTECPP_STRINGIFY_HELPER(x)
#define COMPUTECPP_HOST_COMPILER_STRING                                        \
  "MSVC " COMPUTECPP_STRINGIFY(_MSC_FULL_VER)

#define COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)                                \
  COMPUTECPP_PRAGMA(warning(message))
#define COMPUTECPP_MSVC_CXX_DIAGNOSTIC(message)                                \
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)

#else

#if defined(__INTEL_COMPILER)
#define COMPUTECPP_HOST_COMPILER_STRING __VERSION__
#else
#define COMPUTECPP_HOST_COMPILER_STRING "unknown host compiler"
#endif  // __INTEL_COMPILER

#endif

#ifndef COMPUTECPP_HOST_CXX_DIAGNOSTIC
#define COMPUTECPP_HOST_CXX_DIAGNOSTIC(message)
#endif

#ifndef COMPUTECPP_GNU_CXX_DIAGNOSTIC
#define COMPUTECPP_GNU_CXX_DIAGNOSTIC(message)
#endif

#ifndef COMPUTECPP_GCC_CXX_DIAGNOSTIC
#define COMPUTECPP_GCC_CXX_DIAGNOSTIC(message)
#endif

#ifndef COMPUTECPP_CLANG_CXX_DIAGNOSTIC
#define COMPUTECPP_CLANG_CXX_DIAGNOSTIC(message)
#endif

#ifndef COMPUTECPP_MSVC_CXX_DIAGNOSTIC
#define COMPUTECPP_MSVC_CXX_DIAGNOSTIC(message)
#endif

/// COMPUTECPP_DEV @endcond

#endif  // RUNTIME_INCLUDE_SYCL_HOST_COMPILER_MACROS_H_
