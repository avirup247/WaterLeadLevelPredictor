/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file context.h

  @brief This file contains the API for the @ref cl::sycl::context class
*/
#ifndef RUNTIME_INCLUDE_SYCL_CONTEXT_H_
#define RUNTIME_INCLUDE_SYCL_CONTEXT_H_

#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/device.h"
#include "SYCL/exception_list.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"
#include "SYCL/platform.h"
#include "SYCL/predefines.h"
#include "SYCL/property.h"  // IWYU pragma: keep

#include <cstddef>
#include <memory>
#include <system_error>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class context;
class device_selector;

namespace detail {

template <>
struct opencl_backend_traits<sycl::context> {
 private:
 public:
  using input_type = cl_context;
  using return_type = input_type;
};

// Dependant class declarations.
class context;
}  // namespace detail

namespace info {

/**
  @brief Type of the value returned by calling
  context::get_info<info::context::gl_interop>
  @deprecated OpenGL interop no longer supported in SYCL 1.2.1
*/
using gl_context_interop = bool;

/**
  @brief Enum representing values that can be queried using context::get_info.
*/
enum class context : int { reference_count, platform, devices };

}  // namespace info

/** @cond COMPUTECPP_DEV */

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(context, cl_context_info, cl_context)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(context, devices, CL_CONTEXT_DEVICES,
                                      vector_class<cl::sycl::device>,
                                      cl_device_id)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(context, reference_count,
                                      CL_CONTEXT_REFERENCE_COUNT, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(context, platform, CL_CONTEXT_PROPERTIES,
                                      platform, cl_context_properties)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(context, devices,
                                 vector_class<cl::sycl::device>())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(context, reference_count, 0)

/* COMPUTECPP_DEV @endcond */

/** @brief Interface for abstracting and interacting with an underlying
 * cl_context object.
 */
class COMPUTECPP_EXPORT context {
 public:
  /** Constructs a host context
   * @param propList Additional properties
   */
  explicit context(const property_list& propList = {});

  /** @brief Constructs a context object in host mode.
   * @param asyncHandler Handler for asynchronous exceptions
   * @param propList Additional properties
   */
  explicit context(async_handler asyncHandler,
                   const property_list& propList = {});

  /** @brief Constructs a context object using a cl_context object.
   * @param clContext A cl_context object.
   * @param asyncHandler Handler for asynchronous exceptions
   */
  explicit context(cl_context clContext, async_handler asyncHandler = nullptr);

  /** @brief Constructs a context object using a device_selector object. The
   * context is constructed with a single device retrieved from the
   * device_selector object provided.
   * @param deviceSelector A reference to a device_selector object.
   * @param interopFlag Specify whether to use the context for OpenGL interop.
   * @param asyncHandler An optional parameter to specify the async_handler
   * associated with the context.
   * @deprecated OpenGL interop no longer supported in SYCL 1.2.1
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(201703,
                                    "OpenGL interop is no longer available.")
  context(const device_selector& deviceSelector,
          info::gl_context_interop interopFlag,
          async_handler asyncHandler = nullptr);

  /** Constructs a context using the provided device
   * @param dev Device to associate with the context
   * @param propList Additional properties
   */
  context(const device& dev, const property_list& propList = {});

  /** @brief Constructs a context object using a device object. The context is
   * constructed with the device object provided.
   * @param dev Device to associate with the context
   * @param asyncHandler Handler for asynchronous exceptions
   * @param propList Additional properties
   */
  context(const device& dev, async_handler asyncHandler,
          const property_list& propList = {});

  /** Constructs a context using devices available on the provided platform
   * @param plt Platform containing devices to associate with the context
   * @param propList Additional properties
   */
  context(const platform& plt, const property_list& propList = {});

  /** @brief Constructs a context object using a platform object. The context is
   * constructed with all the devices available under the platform object
   * provided.
   * @param plt Platform containing devices to associate with the context
   * @param asyncHandler Handler for asynchronous exceptions
   * @param propList Additional properties
   */
  context(const platform& plt, async_handler asyncHandler,
          const property_list& propList = {});

  /** Constructs a context using a list of devices
   * @param deviceList List of devices to associate with the context
   * @param propList Additional properties
   */
  context(const vector_class<device>& deviceList,
          const property_list& propList = {});

  /** @brief Constructs a context object using a vector_class of device objects.
   * The context is constructed with the devices provided.
   * @param deviceList A vector_class of device objects.
   * @param asyncHandler Handler for asynchronous exceptions
   * @param propList Additional properties
   */
  context(const vector_class<device>& deviceList, async_handler asyncHandler,
          const property_list& propList = {});

  /** @brief Constructs a context object from another device object and retains
   * the cl_context object if the context is not in host mode.
   */
  context(const context& rhs) = default;

  /** @brief Constructs a context object by moving another device object.
   */
  context(context&& rhs) = default;

  /** @brief Completely assigns the contents of the context to that of another
   * and retains the cl_context object if the context is not in host
   * mode.
   */
  context& operator=(const context& rhs) = default;

  /** @brief Completely moves the contents of the context to that of another.
   */
  context& operator=(context&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const context& lhs, const context& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const context& lhs, const context& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Destroys the implementation object.
   */
  COMPUTECPP_TEST_VIRTUAL ~context() = default;

  /** @brief Returns the underlying cl_context object.
   * @return The cl_context object.
   */
  COMPUTECPP_TEST_VIRTUAL cl_context get() const;

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Returns the backend associated with the context.
   * @return Backend associated with the context.
   */
  inline backend get_backend() const { return this->get_backend_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** @brief Specifies whether the context is in host mode.
   * @return A boolean specifying whether the context is in host mode.
   */
  COMPUTECPP_TEST_VIRTUAL bool is_host() const;

  /** @brief Gets OpenCL information for the underlying cl_context.
   * @tparam param A cl_int specifying the info parameter.
   * @return The retrieved information as the appropriate return type,
   * derived via the get_sycl_info function, defined in info.h.
   */
  template <info::context param>
  COMPUTECPP_EXPORT
      typename info::param_traits<info::context, param>::return_type
      get_info() const;

  /** @brief Retrieves the platform associated with this context
   * @return SYCL platform associated with this context
   */
  COMPUTECPP_TEST_VIRTUAL platform get_platform() const;

  /**  @brief Returns the list of devices from the current context
   */
  COMPUTECPP_TEST_VIRTUAL vector_class<cl::sycl::device> get_devices() const;

  /** @brief Returns an opaque pointer to the implementation object.
   * @return A pointer to the implementation object.
   **/
  dcontext_shptr get_impl() const;

  /* @brief Creates a new public context from an existing implementation */
  explicit context(cl::sycl::detail::context* detail);
  explicit context(dcontext_shptr detail);

 private:
  /** Returns the SYCL backend
   * @return Backend associated with the context
   */
  backend get_backend_impl() const;

 protected:
  /** @brief Implementation of the context.
   */
  dcontext_shptr m_impl;
};

/**
@brief This function converts a cl context object to a sycl context object
it is used to allow get_info<cl::sycl::typename::context> to return a
sycl context object
*/
template <>
struct info_convert<cl_context*, context> {
  static context cl_to_sycl(cl_context* clValue, size_t numElems,
                            cl_uint /*clParam*/) {
    for (size_t i = 0; i < numElems; i++) {
      return cl::sycl::context(clValue[0]);
    }
    return cl::sycl::context();
  }
};

}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the context class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <>
struct hash<cl::sycl::context> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::context& object) const {
    hash<cl::sycl::dcontext_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_CONTEXT_H_

////////////////////////////////////////////////////////////////////////////////
