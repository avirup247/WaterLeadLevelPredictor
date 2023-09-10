/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file error.h
 *
 * @brief Provides SYCL exception and error code types.
 */
#ifndef RUNTIME_INCLUDE_SYCL_ERROR_H_
#define RUNTIME_INCLUDE_SYCL_ERROR_H_
#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/context.h"
#include "SYCL/host_compiler_macros.h"
#include "SYCL/include_opencl.h"
#include "SYCL/predefines.h"

#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <system_error>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class context;
namespace detail {
struct sycl_log;

/** @brief Enumeration of SYCL runtime error codes. Detail version for use in
 * separate implementations.
 */
enum class errc {
  runtime,
  kernel,
  accessor,
  nd_range,
  event,
  kernel_argument,
  build,
  invalid,
  memory_allocation,
  platform,
  profiling,
  feature_not_supported,
  kernel_not_supported,
  backend_mismatch,
};

/** @brief Converts an error code into the equivalent string message for use in
 * error categories.
 * @param errorCode The error code to get the message for.
 * @return The message associated with the error code.
 */
COMPUTECPP_EXPORT std::string errc_to_str(errc errorCode) noexcept;

/** @brief Error category for host backend errors. As error conditions are not
 * supported in SYCL the message() method is overridden but non-functional.
 */
class host_error_category : public std::error_category {
 public:
  host_error_category() noexcept = default;
  host_error_category(const host_error_category&) = delete;

  const char* name() const noexcept override { return "host"; }
  std::string message(int condition) const override {
    (void)condition;
    return "message() not yet implemented for the host error category";
  }
};

/** @brief Error category for opencl backend errors. As error conditions are not
 * supported in SYCL the message() method is overridden but non-functional.
 */
class opencl_error_category : public std::error_category {
 public:
  opencl_error_category() noexcept = default;
  opencl_error_category(const opencl_error_category&) = delete;

  const char* name() const noexcept override { return "opencl"; }
  std::string message(int condition) const override {
    (void)condition;
    return "message() not yet implemented for the OpenCL error category";
  }
};

/** @brief Error category for SYCL errors. As error conditions are not
 * supported in SYCL the message() method is overridden but non-functional.
 */
class sycl_error_category : public std::error_category {
 public:
  sycl_error_category() noexcept = default;
  sycl_error_category(const sycl_error_category&) = delete;

  const char* name() const noexcept override { return "sycl"; }
  std::string message(int condition) const override {
    return errc_to_str(static_cast<errc>(condition));
  }
};

/** @brief Detail function which returns a const ref to the static sycl
 * error_category. Allows use in seperate error implementation.
 * @return const ref to static std::error_category instance
 */
inline const std::error_category& make_sycl_category() noexcept {
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wexit-time-destructors")
  static detail::sycl_error_category sycl_category;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  return sycl_category;
}
/** @brief Detail function for making a standard error code from a SYCL error
 * code
 * @param ec The sycl error code to use
 * @return A standard error code containing the integer value of errc
 */
inline std::error_code make_error_code(errc ec) {
  return std::error_code(static_cast<int>(ec), make_sycl_category());
}
}  // namespace detail

#if SYCL_LANGUAGE_VERSION >= 202001

/** @brief Enumeration of SYCL runtime error codes.
 */
enum class errc {
  runtime = static_cast<int>(detail::errc::runtime),
  kernel = static_cast<int>(detail::errc::kernel),
  accessor = static_cast<int>(detail::errc::accessor),
  nd_range = static_cast<int>(detail::errc::nd_range),
  event = static_cast<int>(detail::errc::event),
  kernel_argument = static_cast<int>(detail::errc::kernel_argument),
  build = static_cast<int>(detail::errc::build),
  invalid = static_cast<int>(detail::errc::invalid),
  memory_allocation = static_cast<int>(detail::errc::memory_allocation),
  platform = static_cast<int>(detail::errc::platform),
  profiling = static_cast<int>(detail::errc::profiling),
  feature_not_supported = static_cast<int>(detail::errc::feature_not_supported),
  kernel_not_supported = static_cast<int>(detail::errc::kernel_not_supported),
  backend_mismatch = static_cast<int>(detail::errc::backend_mismatch),
};

/** @brief Shortcut for the error code type of the given backend
 */
template <backend b>
using errc_for = typename backend_traits<b>::errc;

/** @brief Get the SYCL error category. The returned value here is a const ref
 * to a static instance of the category as per other standard library error
 * categories.
 * @return const std::error_category&
 */
inline const std::error_category& sycl_category() noexcept {
  return detail::make_sycl_category();
}

/** @brief Creates an std::error_code from a sycl::errc value. The associated
 * error category will always be a sycl_category().
 * @param e The SYCL error code
 * @return std::error_code
 */
inline std::error_code make_error_code(errc e) noexcept {
  return std::error_code(static_cast<int>(e), sycl_category());
}
#endif  // SYCL_LANGUAGE_VERSION >= 202001

/** @brief SYCL exception class, defined Section 3.2 of the specification,
 * for general SYCL error.
 *
 * This implementation adds extra methods to those defined in the
 * specification to provide additional information to the user.
 */
class COMPUTECPP_EXPORT exception {
 public:
  /// @cond COMPUTECPP_DEV
  /** @brief Constructs a exception from a sycl_log.
   * @param syclLog The sycl_log to be associated with the error.
   * @param context Shared pointer to a detail context if applies
   */
  explicit exception(std::unique_ptr<detail::sycl_log>&& syclLog,
                     dcontext_shptr context = nullptr);
  /// COMPUTECPP_DEV @endcond

#if SYCL_LANGUAGE_VERSION >= 202001
  /** @brief Constructs an exception from an error code and message.
   * @param errorCode The error code associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(std::error_code errorCode, const char* whatArg)
      : exception(detail::impl_constructor_tag{}, errorCode, whatArg) {}

  /** @brief Constructs an exception from an error code and message.
   * @param errorCode The error code associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(std::error_code errorCode, const std::string& whatArg)
      : exception(errorCode, whatArg.c_str()) {}

  /** @brief Constructs an exception from an error code.
   * @param errorCode The error code associated with the exception.
   */
  inline exception(std::error_code errorCode) : exception(errorCode, "") {}

  /** @brief Constructs an exception from an error code value, error category
   * and message.
   * @param errorValue The integer error code value associated with the
   * exception.
   * @param errorCategory The error category associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(int errorValue, const std::error_category& errorCategory,
                   const char* whatArg)
      : exception(detail::impl_constructor_tag{},
                  std::error_code(errorValue, errorCategory), whatArg) {}

  /** @brief Constructs an exception from an error code value, error category
   * and message.
   * @param errorValue The integer error code value associated with the
   * exception.
   * @param errorCategory The error category associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(int errorValue, const std::error_category& errorCategory,
                   const std::string& whatArg)
      : exception(errorValue, errorCategory, whatArg.c_str()) {}

  /** @brief Constructs an exception from an error code value and error
   * category.
   * @param errorValue The integer error code value associated with the
   * exception.
   * @param errorCategory The error category associated with the exception.
   */
  inline exception(int errorValue, const std::error_category& errorCategory)
      : exception(errorValue, errorCategory, "") {}

  /** @brief Constructs an exception from a sycl context, error code and
   * message.
   * @param ctx The sycl context associated with the exception.
   * @param errorCode The error code associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(context ctx, std::error_code errorCode, const char* whatArg)
      : exception(detail::impl_constructor_tag{}, ctx, errorCode, whatArg) {}

  /** @brief Constructs an exception from a sycl context, error code and
   * message.
   * @param ctx The sycl context associated with the exception.
   * @param errorCode The error code associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(context ctx, std::error_code errorCode,
                   const std::string& whatArg)
      : exception(ctx, errorCode, whatArg.c_str()) {}

  /** @brief Constructs an exception from a sycl context and error code.
   * @param ctx The sycl context associated with the exception.
   * @param errorCode The error code associated with the exception.
   */
  inline exception(context ctx, std::error_code errorCode)
      : exception(ctx, errorCode, "") {}

  /** @brief Constructs an exception from a sycl context, error code value,
   * error category and message.
   * @param ctx The sycl context associated with the exception.
   * @param errorValue The error code value associated with the exception.
   * @param errorCategory The error category associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(context ctx, int errorValue,
                   const std::error_category& errorCategory,
                   const char* whatArg)
      : exception(detail::impl_constructor_tag{}, ctx,
                  std::error_code(errorValue, errorCategory), whatArg) {}

  /** @brief Constructs an exception from a sycl context, error code value,
   * error category and message.
   * @param ctx The sycl context associated with the exception.
   * @param errorValue The error code value associated with the exception.
   * @param errorCategory The error category associated with the exception.
   * @param whatArg The message associated with the exception.
   */
  inline exception(context ctx, int errorValue,
                   const std::error_category& errorCategory,
                   const std::string& whatArg)
      : exception(ctx, errorValue, errorCategory, whatArg.c_str()) {}

  /** @brief Constructs an exception from a sycl context, error code value,
   * and error category.
   * @param ctx The sycl context associated with the exception.
   * @param errorValue The error code value associated with the exception.
   * @param errorCategory The error category associated with the exception.
   */
  inline exception(context ctx, int errorValue,
                   const std::error_category& errorCategory)
      : exception(ctx, errorValue, errorCategory, "") {}

  /** @brief Gets the error code associated with this exception
   * @return const ref to an std::error_code
   */
  const std::error_code& code() const noexcept { return get_log_errc(); }

  /** @brief Gets the error category associated with this exception. Equivalent
   * to calling code().category()
   * @return const ref to a static std::error_category instance
   */
  const std::error_category& category() const noexcept {
    return get_log_errc().category();
  }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** @brief Overload of std::runtime_error::what() which returns the message
   * associated with the error.
   * @return The message associated with the error.
   */
  const char* what() const noexcept;

  /** @brief Reports whether the exception has a context associated with it
   * @return True if a context is associated with this exception
   */
  bool has_context() const noexcept;

  /** @brief Returns the SYCL context that is associated with this SYCL
   * exception
   *
   * If no context is associated with this exception, it throws a new exception.
   *
   * @return Context that is associated with this exception
   */
  cl::sycl::context get_context() const;

  /** @brief Returns the OpenCL error code. Value extracted directly from
   * the OpenCL header.
   * @return int OpenCL error code
   */
  cl_int get_cl_code() const;

  /// @cond COMPUTECPP_DEV

  /** @brief Returns the SYCL error message.
   * @return The SYCL error message. The pointer is valid for the lifetime of
   * the exception; if required after that a copy of the null terminated string
   * must be made.
   */
  const char* get_description() const;

  /** @brief Returns the file name that trigger the error.
   * @return The file name.
   */
  const char* get_file_name() const;

  /** @brief Returns the line number that trigger the error.
   * @return The line number.
   */
  int get_line_number() const;

  /** @brief Returns an internal ComputeCpp error code from the error
   * @return The ComputeCpp specific error code representing the error
   */
  detail::cpp_error_code get_cpp_error_code() const;

  /** @brief Returns the name of the OpenCL error macro
   * @return const char * Name of the macro in human-readable-format
   */
  const char* get_cl_error_message() const;

  /// COMPUTECPP_DEV @endcond

 protected:
  /// @cond COMPUTECPP_DEV
  /** @brief Pointer to sycl_log containing the message and other information.
   *
   * Note: Either this must be a copyable pointer or an explicit copy
   * constructor needs to be provided for exceptions to allow the use of
   * `std::make_exception_ptr` which takes an excpetion by value.
   */
  std::shared_ptr<detail::sycl_log> m_syclLog;

  /* m_context.
   * If the SYCL exception was caused by a context, this will hold
   * a pointer to the context that caused the problem.
   */
  dcontext_shptr m_context;
  /// COMPUTECPP_DEV @endcond

  exception(detail::impl_constructor_tag, std::error_code errorCode,
            const char* whatArg);

  exception(detail::impl_constructor_tag, context ctx,
            std::error_code errorCode, const char* whatArg);

  /** @brief Returns the error_code from inside the member
   * sycl_log to avoid leaking the full log definition.
   * @return const ref to an std::error_code
   */
  const std::error_code& get_log_errc() const noexcept;
};

#if SYCL_LANGUAGE_VERSION >= 202001

/** @brief Get the error category associated with the backend b. The returned
 * value here is a const ref to a static instance of the category as per other
 * standard library error categories.
 * @return const std::error_category&
 */
template <backend b>
const std::error_category& error_category_for() noexcept;

// Specializations of error_category_for for supported backends.

template <>
inline const std::error_category& error_category_for<backend::host>() noexcept {
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wexit-time-destructors")
  static detail::host_error_category cat;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  return cat;
}

template <>
inline const std::error_category&
error_category_for<backend::opencl>() noexcept {
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wexit-time-destructors")
  static detail::opencl_error_category cat;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)
  return cat;
}
#endif  // SYCL_LANGUAGE_VERSION >= 202001

namespace detail {

enum class exception_types {
  runtime,
  kernel,
  accessor,
  nd_range,
  event,
  invalid_parameter,
  device,
  compile_program,
  link_program,
  invalid_object,
  memory_allocation,
  platform_error,
  profiling,
  feature_not_supported
};

template <exception_types type, typename Base>
class exception_implementation : public Base {
 public:
  using Base::Base;
};

}  // namespace detail

/** @brief Base SYCL runtime error group. Sub-classes of this error
 * represent a runtime specific error.
 */
class runtime_error
    : public detail::exception_implementation<detail::exception_types::runtime,
                                              exception> {
 public:
  using exception_implementation::exception_implementation;
};

/** @brief Represents an error that occurred before or while enqueuing a SYCL
 * kernel.
 */
class kernel_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error regarding \ref cl::sycl::accessor objects
 * defined.
 */

class accessor_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to a provided nd_range.
 */
class nd_range_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to a \ref cl::sycl::event.
 */
class event_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Represents an error related to SYCL kernel parameters.
 */
class invalid_parameter_error : public runtime_error {
 public:
  using runtime_error::runtime_error;
};

/** @brief Base SYCL device error group. Sub-classes of this error
 * represent a device specific error.
 */
class device_error
    : public detail::exception_implementation<detail::exception_types::runtime,
                                              exception> {
 public:
  using exception_implementation::exception_implementation;
};

/** @brief Represents an error that happened during compilation.
 */
class compile_program_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an error that happened during linking.
 */
class link_program_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an error regarding any memory object being used by a
 * kernel.
 */
class invalid_object_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents a memory allocation error.
 */
class memory_allocation_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents a platform related error.
 */
class platform_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an issue related to profiling (can only be raised if
 * profiling is enabled).
 */
class profiling_error : public device_error {
 public:
  using device_error::device_error;
};

/** @brief Represents an exception when an optional feature
 * or extension is used in a kernel but its not
 * available on the device the SYCL kernel is
 * being enqueued on.
 */
class feature_not_supported : public device_error {
 public:
  using device_error::device_error;
};

}  // namespace sycl
}  // namespace cl

#if SYCL_LANGUAGE_VERSION >= 202001
/** Specializations of is_error_code_enum for supported backend's error code
 * types. OpenCL backend is omitted because ::errc is int and specializations of
 * std types must be for user-defined types only. Host seems to work even though
 * its error type is void, this might not also be allowed.
 */
namespace std {
// TODO: Add specializations for backend_traits::errc.

template <>
struct is_error_code_enum<cl::sycl::errc> : public true_type {};

}  // namespace std
#endif  // SYCL_LANGUAGE_VERSION >= 202001
#endif  // RUNTIME_INCLUDE_SYCL_ERROR_H_
