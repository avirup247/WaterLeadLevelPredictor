/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_ASSERT_H_
#define RUNTIME_INCLUDE_SYCL_ASSERT_H_

/// @cond COMPUTECPP_DEV

#if defined(__assume) || defined(_MSC_VER)
#define COMPUTECPP_BUILTIN_UNREACHABLE() __assume(false)
#elif defined(__builtin_unreachable) || defined(__GNUC__)
#define COMPUTECPP_BUILTIN_UNREACHABLE() __builtin_unreachable()
#elif defined(__has_builtin)
#if __has_builtin(__builtin_unreachable)
#define COMPUTECPP_BUILTIN_UNREACHABLE() __builtin_unreachable()
#endif  // __has_builtin(__builtin_unreachable)
#endif  // __assume || _MSC_VER
#ifndef COMPUTECPP_BUILTIN_UNREACHABLE
#define COMPUTECPP_BUILTIN_UNREACHABLE()
#endif  // COMPUTECPP_BUILTIN_UNREACHABLE

#if defined(NDEBUG) || defined(__SYCL_DEVICE_ONLY__)
#if defined(__assume) || defined(_MSC_VER)
#define COMPUTECPP_ASSERT_HELPER(condition, message) __assume(condition)
#else
#define COMPUTECPP_ASSERT_HELPER(condition, message) ((void)0)
#endif  // __assume  || _MSC_VER
#else
#include <cassert>
#define COMPUTECPP_ASSERT_HELPER(condition, message)                           \
  assert((condition) && (message))
#endif  // NDEBUG || __SYCL_DEVICE_ONLY__

#if defined(NDEBUG)
#define COMPUTECPP_UNREACHABLE_HELPER(message) COMPUTECPP_BUILTIN_UNREACHABLE()
#else
#define COMPUTECPP_UNREACHABLE_HELPER(message)                                 \
  do {                                                                         \
    COMPUTECPP_ASSERT_HELPER(false, (message));                                \
    COMPUTECPP_BUILTIN_UNREACHABLE();                                          \
  } while (false)
#endif  // NDEBUG

/// COMPUTECPP_DEV @endcond

/** Asserts that a statement is true,
 *  otherwise prints a message and aborts the program
 * @param condition Statement to evaluate, must be convertible to boolean
 * @param message Message to print on failure
 * @note The statement is not evaluated in Release mode
 */
#define COMPUTECPP_ASSERT(condition, message)                                  \
  COMPUTECPP_ASSERT_HELPER((condition), (message))

/** Indicates that the point of calling should not be reachable.
 *  If it is, it's most likely an internal programming error.
 * @param message Message to print on failure
 * @note Reaching this code in Release mode is Undefined Behavior
 */
#define COMPUTECPP_UNREACHABLE(message) COMPUTECPP_UNREACHABLE_HELPER((message))

#endif  // RUNTIME_INCLUDE_SYCL_ASSERT_H_
