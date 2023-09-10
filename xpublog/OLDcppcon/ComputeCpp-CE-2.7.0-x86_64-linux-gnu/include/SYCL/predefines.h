/*****************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

********************************************************************/

/** @file predefines.h
 *
 * @brief This file contains common internal runtime pre-processor definitions
 * and the pre-processor definitions required by the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_PREDEFINES_H_
#define RUNTIME_INCLUDE_SYCL_PREDEFINES_H_

/// @cond COMPUTECPP_DEV

#define ABACUS_LIBRARY_STATIC 1

// This file is generated from CMake and it contains some common version macros
#include "version.h"

#include "SYCL/host_compiler_macros.h"
#include <computecpp_export.h>

/** Homogenize the operating system identification macro
 */
#if defined(WINDOWS) || defined(_WIN32) || defined(_WIN64)
#define COMPUTECPP_WINDOWS
#if defined(_WIN64)
#define COMPUTECPP_ENV_64
#define COMPUTECPP_PTR_SIZE 8
#else  // _WIN64
#define COMPUTECPP_ENV_32
#define COMPUTECPP_PTR_SIZE 4
#endif  // _WIN64
#endif  // defined(WINDOWS) || defined(_WIN32) || defined(_WIN64)
#ifdef __linux__
#if defined(__x86_64__) || defined(__ppc64__) || defined(__aarch64__)
#define COMPUTECPP_ENV_64
#define COMPUTECPP_PTR_SIZE 8
#else  // 64__
#define COMPUTECPP_ENV_32
#define COMPUTECPP_PTR_SIZE 4
#endif  // 64__
#define COMPUTECPP_LINUX
#endif  // __linux__

#if defined(__GNUC__) && !defined(__llvm__)
#if (__GNUC__ < 5)
#define COMPUTECPP_GCC_PRE_5
#if (__GNUC_MINOR__ < 9)
#define COMPUTECPP_GCC_PRE_4_9
#endif  // (__GNUC_MINOR__ < 9)
#endif  // (__GNUC__ < 5)
#endif  // defined(__GNUC__) && !defined(__llvm__)

#ifdef __MINGW32__
#define COMPUTECPP_THREAD_LOCAL __thread
#else
#define COMPUTECPP_THREAD_LOCAL thread_local
#endif  // __MINGW32__

/** When there is a mismatch between the host and the device bitness,
 * we need to ensure the alignment on both platforms matches.
 * To guarantee that, we force it to be aligned to 8, which works on
 * both combinations.
 */
#if defined(COMPUTECPP_ENV_32) && defined(__DEVICE_SPIR64__)
#define COMPUTECPP_PTR_SIZE 8
#elif defined(COMPUTECPP_ENV_64) && defined(__DEVICE_SPIR32__)
#define COMPUTECPP_PTR_SIZE 8
#endif  // defined(COMPUTECPP_ENV_32) && defined(__DEVICE_SPIR64__)

// Used to suppress "calling convention __stdcall ignored for this target"
#if defined(COMPUTECPP_WINDOWS) && defined(__SYCL_DEVICE_ONLY__)
#define COMPUTECPP_CL_API_CALL
#else
#define COMPUTECPP_CL_API_CALL CL_API_CALL
#endif

/// Default SYCL 1.2.1 version
#define COMPUTECPP_SYCL_VERSION_121 201703
/// Default SYCL 2020 version
#define COMPUTECPP_SYCL_VERSION_2020 202002

/** Used to prevent clang-format from badly formatting code
 *  on the following lines in files that are used as input for generated code
 *  (.h.in extension)
 */
#define COMPUTECPP_CLANG_FORMAT_BARRIER

/// COMPUTECPP_DEV @endcond

/** Set the the default value of SYCL_LANGUAGE_VERSION to SYCL version 1.2.1
 */
#ifndef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION COMPUTECPP_SYCL_VERSION_121
#endif  // SYCL_LANGUAGE_VERSION

// For compatibility we allow the user to set the version to 2017
#if (SYCL_LANGUAGE_VERSION == 2017)
#undef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION COMPUTECPP_SYCL_VERSION_121
#endif  // SYCL_LANGUAGE_VERSION

// For compatibility we allow the user to set the version to 2020
#if (SYCL_LANGUAGE_VERSION == 2020)
#undef SYCL_LANGUAGE_VERSION
#define SYCL_LANGUAGE_VERSION COMPUTECPP_SYCL_VERSION_2020
#endif  // SYCL_LANGUAGE_VERSION

/** Set the value of CL_SYCL_LANGUAGE_VERSION based on the value of
 * SYCL_LANGUAGE_VERSION.
 * The CL_SYCL_LANGUAGE_VERSION macro is deprecated in SYCL 2020
 */
#ifndef CL_SYCL_LANGUAGE_VERSION
#if (SYCL_LANGUAGE_VERSION >= 201700) && (SYCL_LANGUAGE_VERSION < 202000)
#define CL_SYCL_LANGUAGE_VERSION 121
#elif (SYCL_LANGUAGE_VERSION >= 202000)
#define CL_SYCL_LANGUAGE_VERSION 2020
#endif  // SYCL_LANGUAGE_VERSION
#endif  // CL_SYCL_LANGUAGE_VERSION

/// @cond COMPUTECPP_DEV

/* This macro is used for methods that are used in the unit tests and hence need
 * to be
 * specified virtual, but are not required to be virtual for the normal
 * operation of the runtime.
 */
#ifdef COMPUTECPP_BUILD_UNIT_TEST
#define COMPUTECPP_TEST_VIRTUAL virtual
#define COMPUTECPP_TEST_OVERRIDE override
#else
#define COMPUTECPP_TEST_VIRTUAL
#define COMPUTECPP_TEST_OVERRIDE final
#endif  // COMPUTECPP_BUILD_UNIT_TEST

#if defined(__cpp_constexpr) && (__cpp_constexpr >= 201304)
#define COMPUTECPP_CONSTEXPR_CPP14_HELPER constexpr
#else
#define COMPUTECPP_CONSTEXPR_CPP14_HELPER inline
#endif  // __cpp_constexpr

/** constexpr specifier if C++14 constexpr is supported.
 *  If not, it's just inline.
 */
#define COMPUTECPP_CONSTEXPR_CPP14 COMPUTECPP_CONSTEXPR_CPP14_HELPER

/** Define a symbol constexpr (if supported) and also export the symbol.
 * @note Only useful for maintaining ABI
 */
#define COMPUTECPP_CONSTEXPR_EXPORT COMPUTECPP_EXPORT COMPUTECPP_CONSTEXPR_CPP14

#if !defined(_MSC_VER) && defined(ComputeCpp_EXPORTS)
#define COMPUTECPP_ABI_CONSTEXPR_HELPER COMPUTECPP_EXPORT
#else
#define COMPUTECPP_ABI_CONSTEXPR_HELPER COMPUTECPP_CONSTEXPR_EXPORT
#endif  // !_MSC_VER && ComputeCpp_EXPORTS

/** @brief In some cases @ref COMPUTECPP_CONSTEXPR_EXPORT is not enough
 *  for exporting the symbol, typically when inline functions are hidden.
 * @note Only useful for maintaining ABI
 */
#define COMPUTECPP_ABI_CONSTEXPR COMPUTECPP_ABI_CONSTEXPR_HELPER

/// COMPUTECPP_DEV @endcond

namespace cl {
namespace sycl {
namespace detail {

/** @internal
 * ComputeCpp internal error codes. Use to report meaning full errors.
 */
enum class cpp_error_code : unsigned int {
  CPP_NO_ERROR = 0,
  // General Errors
  UNKNOWN_ERROR = 1,
  OUT_OF_HOST_MEMORY = 2,
  RETAIN_CL_OBJECT_ERROR = 3,
  RELEASE_CL_OBJECT_ERROR = 4,
  HOST_MEMORY_ALLOCATION_ERROR = 5,
  NOT_SUPPORTED_ERROR = 6,
  UNREACHABLE_ERROR = 7,
  SYCL_OBJECTS_STILL_ALIVE = 8,
  TARGET_ENV_ERROR = 9,
  TARGET_FORMAT_ERROR = 10,
  NULLPTR_ERROR = 11,
  INVALID_CONFIG_FILE = 12,
  INVALID_CONFIG_OPTION = 13,

  // Program/Kernel issues
  BUILD_PROGRAM_ERROR = 100,
  CREATE_KERNEL_ERROR = 101,
  KERNEL_NOT_FOUND_ERROR = 102,
  GET_PROGRAM_INFO_ERROR = 103,
  GET_KERNEL_INFO_ERROR = 104,
  TARGET_NOT_FOUND_ERROR = 105,
  DEVICE_NOT_FOUND_ERROR = 106,
  CREATE_PROGRAM_FROM_BINARY_ERROR = 107,
  LINK_PROGRAM_ERROR = 108,
  KERNEL_BUILD_ERROR = 109,
  CREATE_PROGRAM_FROM_SOURCE_ERROR = 110,
  CL_SET_KERNEL_ARGUMENT_ERROR = 111,
  RETAIN_KERNEL_ERROR = 112,
  RELEASE_KERNEL_ERROR = 113,
  INVALID_CL_PROGRAM_ERROR = 114,
  DEVICE_UNSUPPORTED_EXTENSIONS_ERROR = 115,
  INVALID_CL_KERNEL_ERROR = 116,
  COMPILE_PROGRAM_ERROR = 117,
  BINARY_NOT_FOUND_ERROR = 118,

  // Execution/Transaction/Command group errors
  CREATE_IMPLICIT_QUEUE_ERROR = 200,
  WAIT_FOR_EVENT_ERROR = 201,
  GET_EVENT_INFO_ERROR = 202,
  SET_USER_EVENT_STATUS_ERROR = 203,
  CREATE_CONTEXT_ERROR = 204,
  CREATE_USER_EVENT_ERROR = 205,
  QUEUE_FINISH_ERROR = 206,
  QUEUE_FLUSH_ERROR = 207,
  ENQUEUE_ERROR = 208,
  RETAIN_CONTEXT_ERROR = 209,
  RELEASE_CONTEXT_ERROR = 210,
  CONTEXT_WITH_NO_DEVICES_ERROR = 211,
  RETAIN_CL_EVENT_ERROR = 212,
  INVALID_CL_EVENT_ERROR = 213,
  RELEASE_CL_EVENT_ERROR = 214,
  SET_CALLBACK_ERROR = 215,
  COMMAND_GROUP_SUBMIT_ERROR = 216,
  KERNEL_EXECUTION_ERROR = 217,
  TRANSACTION_ADD_KERNEL_PARAM_ERROR = 218,
  COMMAND_GROUP_SYNTAX_ERROR = 219,
  MAXIMUM_DEVICES_PER_CONTEXT_ERROR = 220,
  NO_COMMAND_GROUP_AVAILABLE_ERROR = 221,
  ACCESSOR_OUTSIDE_COMMAND_GROUP_ERROR = 222,
  HOST_ACCESSOR_IN_COMMAND_GROUP_ERROR = 223,
  INSUFFICIENT_MEMORY_ON_SUBMIT_ERROR = 224,

  // Local/Workgroup problems
  GET_WORKGROUP_INFO_ERROR = 300,
  WORK_GROUP_SIZE_ERROR = 301,

  // Runtime classes errors
  GET_INFO_ERROR = 400,
  CREATE_SUBDEVICE_ERROR = 401,
  CREATE_DEVICE_ERROR = 402,
  RELEASE_DEVICE_ERROR = 403,
  GET_CL_MEM_OBJ_INFO_ERROR = 404,
  INVALID_CL_MEM_OBJ_ERROR = 405,
  INVALID_CONTEXT_ERROR = 406,
  CREATE_QUEUE_ERROR = 407,
  QUERY_NUMBER_OF_PLATFORMS_ERROR = 408,
  QUERY_PLATFORM_ERROR = 409,
  ACCESSOR_ARGUMENTS_ERROR = 410,
  INCORRECT_ACCESSOR_TYPE_ERROR = 411,
  PROPERTY_ERROR = 412,
  INVALID_CL_DEVICE_ERROR = 413,
  INVALID_CL_QUEUE_ERROR = 414,
  INVALID_CL_PLATFORM_ERROR = 415,
  NO_PROFILING_INFO_ERROR = 416,
  CANNOT_LOAD_CL_FUNCTION_POINTER = 417,

  // Buffers/images/samplers errors
  CREATE_BUFFER_ERROR = 500,
  CREATE_SUBBUFFER_ERROR = 501,
  CREATE_IMAGE_ERROR = 502,
  CREATE_SAMPLER_ERROR = 503,
  RELEASE_MEM_OBJECT_ERROR = 504,
  RETAIN_MEM_OBJECT_ERROR = 505,
  GET_CL_MEM_ERROR = 506,
  CREATE_NDRANGE_ERROR = 507,
  MEMORY_OBJECT_UNAVAILABLE_ERROR = 508,
  INVALID_OBJECT_ERROR = 509,
  NULL_BUFFER_ERROR = 510,
  USM_ALLOCATION_ERROR = 511,
  USM_DEVICE_FOR_POINTER_NOT_FOUND = 512,

  // Profiling API errors
  PROFILING_ENTRY_NOT_FOUND_ERROR = 600,
  CANNOT_WRITE_PROFILING_OUTPUT = 601,
  SET_EVENT_CALLBACK_ERROR = 602,
  JSON_PROFILING_ERROR = 603,

  // Extensions
  EXT_ONCHIP_MEMORY_ERROR = 900,
  EXT_SET_PLANE_ERROR = 901,
  EXT_SUBGROUP_INFO_ERROR = 902,
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#ifndef __SYCL_DEVICE_ONLY__

#ifdef _MSC_VER

#ifdef _MSC_FULL_VER

#if _MSC_FULL_VER < 170051025
#error "SYCL requires c++11 support"
#endif

#endif  // _MSC_FULL_VER

#else
#if __cplusplus <= 199711L
#error "SYCL requires c++11 support"
#endif
#endif  // _MSC_VER

#endif  //  __SYCL_DEVICE_ONLY__

/*
 * Suppress MSVC warnings
 */
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(push)

// Disable some warnings
// because impl objects used in export classes
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(disable : 4251)
// because impl objects used in export classes
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(disable : 4275)
// unchecked iterators
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(disable : 4996)

// Disable some non-standard extensions
// non-constant aggregate initializer
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(error : 4204)
// class rvalue used as lvalue
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(error : 4238)
// binding temporaries to references
COMPUTECPP_MSVC_CXX_DIAGNOSTIC(error : 4239)

/// @cond COMPUTECPP_DEV

/**
@internal
@def COMPUTECPP_VALID_KERNEL_ARG_IF(Condition, Message)
Macro which expands to the attribute computecpp::valid_kernel_arg_if
with the given condition and message. If a value of a type with this parameter
is offloaded to a kernel, if the condition is false an error will be raised,
including the given message in the form of a note.
This is used, for example, to prevent offloading of host accessors to device
code.
@param Condition True if this type can be offloaded, false otherwise.
@param Message Message to print after the error, to clarify what the issue might
be.
*/
#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_VALID_KERNEL_ARG_IF(Condition, Message)
#else
#define COMPUTECPP_VALID_KERNEL_ARG_IF(Condition, Message)                     \
  [[computecpp::valid_kernel_arg_if((Condition), (Message))]]
#endif  // __SYCL_DEVICE_ONLY__

// Suppress security warnings
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif  // _CRT_SECURE_NO_WARNINGS
#endif  // _MSC_VER

namespace cl {
namespace sycl {
namespace detail {
/// parameter_kind.
/// \brief Represents the parameter kind of a SYCL Kernel as understood
//         by the SYCL device compiler. See device compiler documentation
//         for details.
enum class parameter_kind : unsigned {
  /// \brief Represents an invalid parameter.
  invalid = 0,

  /// \brief Represents a normal parameter that should not get
  /// any special treatment (it's a standard layout struct,
  /// or a built-in type).
  normal = 1,

  /// \brief Represents any kind of pointer.
  pointer = 2,

  /// \brief Represents any OpenCL image built-in type (type
  /// BuiltinType::OCLImage1d, OCLImage1dArray, OCLImage1dBuffer,
  /// OCLImage2d, OCLImage2dArray, OCLImage3d).
  ocl_image = 3,

  /// \brief Represents an OpenCL sampler parameter (type
  /// BuiltinType::OCLSampler).
  ocl_sampler = 4,

  /// \brief Represents an OpenCL event parameter (type
  /// BuiltinType::OCLEvent).
  ocl_event = 5
};

/**
  The \ref{COMPUTECPP_CONVERT_ATTR} macro is used to define the opencl_convert
  attribute for the device side, this attribute is used to tell the device
  compiler that the struct should be flattened into it's containing fields when
  being passed to the kernel function.

  The \ref{COMPUTECPP_CONVERT_ATTR_PLACEHOLDER} macro performs the same role as
  COMPUTECPP_CONVERT_ATTR, but is also a way to distinguish a placeholder
  accessor from a regular one.
*/

/** @brief Distinguishes the different types of accessors
 */
enum class parameter_class : int {
  user_defined = 0, /**< Represents an accessor to a used defined type */
  value = 1,        /**< Represents a value accessor */
  placeholder = 2,  /**< Represents a placeholder accessor */
  stream = 3,       /**< Represents a stream */
  sampler = 4,      /**< Represents a sampler */
  usm_wrapper = 5,  /**< Represents a USM pointer wrapper */
};
}  // namespace detail
}  // namespace sycl
}  // namespace cl

#ifdef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_CONVERT_ATTR                                                \
  __attribute__((opencl_convert(::cl::sycl::detail::parameter_class::value)))
#define COMPUTECPP_CONVERT_ATTR_PLACEHOLDER                                    \
  __attribute__(                                                               \
      (opencl_convert(::cl::sycl::detail::parameter_class::placeholder)))
#define COMPUTECPP_CONVERT_ATTR_STREAM                                         \
  __attribute__((opencl_convert(::cl::sycl::detail::parameter_class::stream)))
#define COMPUTECPP_CONVERT_ATTR_SAMPLER                                        \
  __attribute__((opencl_convert(::cl::sycl::detail::parameter_class::sampler)))
#define COMPUTECPP_CONVERT_ATTR_USM_WRAPPER                                    \
  __attribute__(                                                               \
      (opencl_convert(::cl::sycl::detail::parameter_class::usm_wrapper)))

#else
#define COMPUTECPP_CONVERT_ATTR
#define COMPUTECPP_CONVERT_ATTR_PLACEHOLDER
#define COMPUTECPP_CONVERT_ATTR_STREAM
#define COMPUTECPP_CONVERT_ATTR_SAMPLER
#define COMPUTECPP_CONVERT_ATTR_USM_WRAPPER
#endif

/*******************************************************************************
    COMPUTECPP_ENABLE_IF macro
*******************************************************************************/

/**
@internal
@def COMPUTECPP_ENABLE_IF(typeName, condition)
Macro for specifying a condition to the method of a template class which
determines whether or not the method is enabled. The macro should be used in
place of a template parameter to a template class method. The condition must be
a compile time boolean expression and is used as the parameter to an
std::enable_if in order to remove the method from overload resolution if the
condition is not met according to the rules of SFINAE. A minor addition to this
is the first parameter to the macro, this parameter must be a typename from the
class template arguments, the reason for this is that SFINAE can only work if
the enable_if is dependant on template arguments of the method itself, rather
than from just the class. So this macro adds an additional template argument to
the method, uses the typename specified as it's default argument and performs a
logical AND with an is_same comparison between the method template argument
typename and the typename specified. This means that the method overload is then
dependant on a method template argument, however that argument is then resolved
to the default at the point of overload resolution, making the is_same
comparison always true and SFINAE can allow the overload to be dropped without
an error.
@param typeName Specify a typename from the class template arguments.
@param condition Specify a compile time boolean expression describing the
context in which the method overload should be enabled.
*/
#define COMPUTECPP_ENABLE_IF(typeName, condition)                              \
  typename overloadDependantT = typeName,                                      \
           typename = typename std::enable_if <                                \
                          std::is_same<typeName, overloadDependantT>::value && \
                      (condition) > ::type

/**
 * @brief Introduces a template parameter that can enable a function definition
 *        based on the provided condition.
 *
 * Similar to @p COMPUTECPP_ENABLE_IF, except that this works with non-type
 * template parameters.
 * It actually introduces two template parameters, and one of them is defaulted,
 * so this would typically be used as the last template parameter.
 *
 * @param parameter Name of the non-type template parameter
 * @param condition When to enable the function definition
 * @internal
 */
#define COMPUTECPP_ENABLE_IF_VAL(parameter, condition)                         \
  decltype(parameter) dependantParameter = parameter,                          \
                      typename std::enable_if <                                \
                              (dependantParameter == parameter) &&             \
                          (condition),                                         \
                      decltype(parameter) > ::type = 0

/// COMPUTECPP_DEV @endcond

#ifdef COMPUTECPP_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#endif  // COMPUTECPP_WINDOWS

#if defined(_MSC_VER) ||                                                       \
    (defined(__has_cpp_attribute) && __has_cpp_attribute(deprecated))
#define COMPUTECPP_DEPRECATED_API_HELPER(reason) [[deprecated(reason)]]
#else
#define COMPUTECPP_DEPRECATED_API_HELPER(reason)                               \
  __attribute__((deprecated(reason)))
#endif  // __has_cpp_attribute(deprecated)

// defining COMPUTECPP_IGNORE_COMPUTECPP_DEPRECATED_API ignores deprecations.
#ifdef COMPUTECPP_IGNORE_COMPUTECPP_DEPRECATED_API
#define COMPUTECPP_DEPRECATED_API(reason)
#else
/** Marks an interface as deprecated
 * @param reason Information about the deprecation
 */
#define COMPUTECPP_DEPRECATED_API(reason)                                      \
  COMPUTECPP_DEPRECATED_API_HELPER(reason)
#endif  // COMPUTECPP_IGNORE_COMPUTECPP_DEPRECATED_API

#if SYCL_LANGUAGE_VERSION >= 202001
/** COMPUTECPP_DEPRECATED_BY_SYCL_202001(message)
 * @brief Macro which marks an interface as deprecated due to being deprecated
 * or removed in the SYCL 2020 specification. Redirects to
 * COMPUTECPP_DEPRECATED_API if the chosen version specified by
 * SYCL_LANGUAGE_VERSION is greater than or equal to 2020.
 */
#define COMPUTECPP_DEPRECATED_BY_SYCL_202001(message)                          \
  COMPUTECPP_DEPRECATED_API(                                                   \
      "Deprecated or removed in SYCL 2020 provisional. " message)
#else
#define COMPUTECPP_DEPRECATED_BY_SYCL_202001(message)
#endif  // SYCL_LANGUAGE_VERSION >= 202001

#if SYCL_LANGUAGE_VERSION >= 201703
/** COMPUTECPP_DEPRECATED_BY_SYCL_2017(message)
 * @brief Macro which marks an interface as deprecated due to being deprecated
 * or removed in the SYCL 1.2.1 specification. Redirects to
 * COMPUTECPP_DEPRECATED_API if the chosen version specified by
 * SYCL_LANGUAGE_VERSION is greater than to SYCL 1.2.1.
 */
#define COMPUTECPP_DEPRECATED_BY_SYCL_201703(message)                          \
  COMPUTECPP_DEPRECATED_API(                                                   \
      "Deprecated or removed in SYCL 1.2.1 (2017). " message)
#else
#define COMPUTECPP_DEPRECATED_BY_SYCL_201703(message)
#endif  // SYCL_LANGUAGE_VERSION >= 201703

/** COMPUTECPP_DEPRECATED_BY_SYCL_VER(syclVer, message)
 * @brief Macro which marks an interface as deprecated due to being deprecated
 * or removed in the SYCL 1.2.1 specification. Redirects to either
 * COMPUTECPP_DEPRECATED_BY_SYCL_ and the version number syclVer.
 */
#define COMPUTECPP_DEPRECATED_BY_SYCL_VER(syclVer, message)                    \
  COMPUTECPP_DEPRECATED_BY_SYCL_##syclVer(message)

#if SYCL_LANGUAGE_VERSION >= 202001
#define COMPUTECPP_INLINE_EXPERIMENTAL inline
#else
#define COMPUTECPP_INLINE_EXPERIMENTAL
#endif  // SYCL_LANGUAGE_VERSION

#ifdef __SYCL_DEVICE_ONLY__
/** @brief Using this attribute on a class ensures that objects of that class
 * are allocated in the private address space.
 */
#ifndef __SYCL_COMPUTECPP_ASP__
#define COMPUTECPP_PRIVATE_MEMORY_ATTR __attribute__((__offload_private))
#else  // __SYCL_COMPUTECPP_ASP__
// ASP doesn't support this attribute yet
#define COMPUTECPP_PRIVATE_MEMORY_ATTR
#endif  // __SYCL_COMPUTECPP_ASP__
#else   // Non-device code
#define COMPUTECPP_PRIVATE_MEMORY_ATTR
#endif  // __SYCL_DEVICE_ONLY__

/// COMPUTECPP_DEV @endcond

/** @brief Adding manually cmath header.
 * This sanitizes the potential C Macros defines and ensures all
 * math dependencies are satisfied further down the line.
 */
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#endif  // RUNTIME_INCLUDE_SYCL_PREDEFINES_H_

////////////////////////////////////////////////////////////////////////////////
