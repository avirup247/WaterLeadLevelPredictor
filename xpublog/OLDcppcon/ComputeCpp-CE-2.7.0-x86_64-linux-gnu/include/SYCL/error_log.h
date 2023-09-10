/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file error_log.h
 *
 * @brief Provides error and message logging macros.
 */
#ifndef RUNTIME_INCLUDE_SYCL_ERROR_LOG_H_
#define RUNTIME_INCLUDE_SYCL_ERROR_LOG_H_

#include "SYCL/common.h"  // for log_type

#include <system_error>

/// @cond COMPUTECPP_DEV

namespace cl {
namespace sycl {
namespace detail {
class context;

/**Triggers a log in the ComputeCpp runtime.
 *
 * This provides a single entry point into the error handling of ComputeCpp, and
 * can be used for any log or error. Depending on the type of log specified a
 * message will be logged or an exception thrown.
 *
 * Typically this function should not be called directly, but through one of
 * the error/log handling macros, e.g. @p COMPUTECPP_LOG and
 * @p COMPUTECPP_CL_ERROR_CODE.
 *
 * @param logType The type of log that is being triggered.
 * @param filePath The path to the file where the log was triggered.
 * @param lineNumber The line number where the log was triggered.
 * @param openCLErrorCode An optional OpenCL error code that is associated with
 *                        the log.
 * @param contextPointer An optional pointer to a context associated with the
 *                       log for handling asynchronous errors.
 * @param extraInformation An optional additional message to include in the log.
 */
COMPUTECPP_EXPORT void trigger_sycl_log(log_type logType, const char* filePath,
                                        int lineNumber, int openclErrorCode,
                                        detail::cpp_error_code cppErrorCode,
                                        const detail::context* contextPointer,
                                        const char* extraInformation);

/**Triggers a log in the ComputeCpp runtime.
 *
 * This provides a single entry point into the error handling of ComputeCpp, and
 * can be used for any log or error. Depending on the type of log specified a
 * message will be logged or an exception thrown.
 *
 * Typically this function should not be called directly, but through one of
 * the error/log handling macros, e.g. @p COMPUTECPP_LOG and
 * @p COMPUTECPP_CL_ERROR_CODE.
 *
 * @param logType The type of log that is being triggered.
 * @param filePath The path to the file where the log was triggered.
 * @param lineNumber The line number where the log was triggered.
 * @param openCLErrorCode An optional OpenCL error code that is associated with
 *                        the log.
 * @param contextPointer An optional pointer to a context associated with the
 *                       log for handling asynchronous errors.
 * @param extraInformation An optional additional message to include in the log.
 * @param errc A SYCL error code integer value.
 */
COMPUTECPP_EXPORT void trigger_sycl_log(log_type logType, const char* filePath,
                                        int lineNumber, int openclErrorCode,
                                        detail::cpp_error_code cppErrorCode,
                                        const detail::context* contextPointer,
                                        const char* extraInformation, int errc);

/** Triggers a log in the ComputeCpp runtime.
 * @ref trigger_sycl_log
 */
inline void trigger_sycl_log(log_type logType, const char* filePath,
                             int lineNumber, int openclErrorCode,
                             detail::cpp_error_code cppErrorCode,
                             const detail::context* contextPointer,
                             const std::string& extraInformation) {
  trigger_sycl_log(logType, filePath, lineNumber, openclErrorCode, cppErrorCode,
                   contextPointer, extraInformation.c_str());
}
/** Triggers a log in the ComputeCpp runtime.
 * @ref trigger_sycl_log
 */
inline void trigger_sycl_log(log_type logType, const char* filePath,
                             int lineNumber, int openclErrorCode,
                             detail::cpp_error_code cppErrorCode,
                             const detail::context* contextPointer,
                             const std::string& extraInformation, int errc) {
  trigger_sycl_log(logType, filePath, lineNumber, openclErrorCode, cppErrorCode,
                   contextPointer, extraInformation.c_str(), errc);
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl

/* Macros for outputting a log to standard output or standard error. */
#ifndef __SYCL_DEVICE_ONLY__
/// @internal Internal log macro
#define COMPUTECPP_NOT_IMPLEMENTED(message)                                    \
  cl::sycl::detail::trigger_sycl_log(                                          \
      cl::sycl::log_type::not_implemented, __FILE__, __LINE__, CL_SUCCESS,     \
      cl::sycl::detail::cpp_error_code::NOT_SUPPORTED_ERROR, nullptr,          \
      message);

/// @internal Internal log macro
#define COMPUTECPP_LOG(logMessage)                                             \
  cl::sycl::detail::trigger_sycl_log(                                          \
      cl::sycl::log_type::info, __FILE__, __LINE__, 0,                         \
      cl::sycl::detail::cpp_error_code::CPP_NO_ERROR, nullptr, logMessage);

/// @internal Internal warning macro
#define COMPUTECPP_WARNING(logMessage)                                         \
  cl::sycl::detail::trigger_sycl_log(                                          \
      cl::sycl::log_type::warning, __FILE__, __LINE__, 0,                      \
      cl::sycl::detail::cpp_error_code::CPP_NO_ERROR, nullptr, logMessage);

/// @internal Internal log macro
#define COMPUTECPP_CL_ERROR_CODE(openclErrorCode, cppErrorCode, contextPtr)    \
  cl::sycl::detail::trigger_sycl_log(cl::sycl::log_type::error, __FILE__,      \
                                     __LINE__, openclErrorCode, cppErrorCode,  \
                                     contextPtr, nullptr);

/// @internal Internal log macro
#define COMPUTECPP_CL_ERROR_CODE_MSG(openclErrorCode, cppErrorCode,            \
                                     contextPtr, extraInformation)             \
  cl::sycl::detail::trigger_sycl_log(cl::sycl::log_type::error, __FILE__,      \
                                     __LINE__, openclErrorCode, cppErrorCode,  \
                                     contextPtr, extraInformation);

/// @internal Internal log macro
#define COMPUTECPP_ERROR_CODE(openclErrorCode, cppErrorCode, contextPtr,       \
                              stdErrorCode)                                    \
  cl::sycl::detail::trigger_sycl_log(                                          \
      cl::sycl::log_type::error, __FILE__, __LINE__, openclErrorCode,          \
      cppErrorCode, contextPtr, nullptr, static_cast<int>(stdErrorCode));

/// @internal Internal log macro
#define COMPUTECPP_ERROR_CODE_MSG(openclErrorCode, cppErrorCode, contextPtr,   \
                                  extraInformation, stdErrorCode)              \
  cl::sycl::detail::trigger_sycl_log(cl::sycl::log_type::error, __FILE__,      \
                                     __LINE__, openclErrorCode, cppErrorCode,  \
                                     contextPtr, extraInformation,             \
                                     static_cast<int>(stdErrorCode));

#else  // __SYCL_DEVICE_ONLY__
#define COMPUTECPP_NOT_IMPLEMENTED(...)
#define COMPUTECPP_LOG(...)
#define COMPUTECPP_WARNING(...)
#define COMPUTECPP_CL_ERROR_CODE(...)
#define COMPUTECPP_CL_ERROR_CODE_MSG(...)
#define COMPUTECPP_ERROR_CODE(...)
#define COMPUTECPP_ERROR_CODE_MSG(...)
#endif  // __SYCL_DEVICE_ONLY__

/// COMPUTECPP_DEV @endcond

#endif  // RUNTIME_INCLUDE_SYCL_ERROR_LOG_H_
