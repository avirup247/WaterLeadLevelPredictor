/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file platform.h
 *
 * @brief This file implements the \ref cl::sycl::platform class as defined by
 * the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_PLATFORM_H_
#define RUNTIME_INCLUDE_SYCL_PLATFORM_H_

#include "SYCL/aspect.h"
#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/common.h"
#include "SYCL/device_info.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"

#include <cstddef>
#include <memory>
#include <system_error>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class device;
class device_selector;
class platform;

namespace info {
/** @brief Platform descriptor to query information about a platform object
 */
enum class platform : unsigned int {
  profile,   /**< Returns the profile name supported by the implementation */
  version,   /**< OpenCL software driver version string */
  name,      /**< Name of the platform */
  vendor,    /**< Vendor name */
  extensions /**< extension names supported by the platform (space-separated
                list) */
};

}  // namespace info

/// @cond COMPUTECPP_DEV

/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(platform, cl_platform_info, cl_platform_id)
/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(platform, name, CL_PLATFORM_NAME,
                                      string_class, char)
/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(platform, vendor, CL_PLATFORM_VENDOR,
                                      string_class, char)
/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(platform, profile, CL_PLATFORM_PROFILE,
                                      string_class, char)
/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(platform, version, CL_PLATFORM_VERSION,
                                      string_class, char)
/// Platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(platform, extensions,
                                      CL_PLATFORM_EXTENSIONS,
                                      vector_class<string_class>, char)

/// Host platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(platform, name, "Host Platform")
/// Host platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(platform, vendor, "Codeplay Software Ltd.")
/// Host platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(platform, profile, "NA")
/// Host platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(platform, version, "1.2.1")
/// Host platform info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(platform, extensions,
                                 vector_class<string_class>())

/// COMPUTECPP_DEV @endcond

namespace detail {

template <>
struct opencl_backend_traits<sycl::platform> {
 private:
 public:
  using input_type = cl_platform_id;
  using return_type = input_type;
};

}  // namespace detail

/** Interface for abstracting and interacting with an underlying cl_platform_id
 * object.
 */
class COMPUTECPP_EXPORT platform {
 public:
  /** Default Constructor.
   * Constructs a platform object in host mode.
   */
  platform();

  /** Constructs a platform object using a cl_platform_id object.
   * @param platformID The cl_platform_id object constructed using the OpenCL
   * API.
   */
  explicit platform(cl_platform_id platformID);

  /** Copy Constructor.
   * Constructs a platform object from another platform object.
   * \param rhs The platform object to copy
   */
  platform(const platform& rhs) = default;

  /** Constructs a platform from an existing device selector
   * \param deviceSelector User device selector
   */
  explicit platform(const cl::sycl::device_selector& deviceSelector);

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Constructs a SYCL platform object using a custom device selector callable
   * @tparam DeviceSelector Type of the callable used for device selection
   * @param deviceSelector Callable that can evaluate devices
   */
  template <typename DeviceSelector>
  explicit platform(const DeviceSelector& deviceSelector)
      : platform{detail::impl_constructor_tag{},
                 detail::device_selector_wrapper{deviceSelector}} {}

#endif  // SYCL_LANGUAGE_VERSION >= 202001

 protected:
  /** Constructs a SYCL platform object using a custom device selector callable
   * @param deviceSelector Callable that can evaluate devices
   */
  explicit platform(detail::impl_constructor_tag,
                    const detail::device_selector_wrapper& deviceSelector);

 public:
  /// @cond COMPUTECPP_DEV

  /** @brief Constructs a platform from a shared a pointer
   */
  explicit platform(const dplatform_shptr& impl);

  /// COMPUTECPP_DEV @endcond

  /** Assignment Operator.
   * Completely assigns the contents of the platform to that of another.
   * \param rhs The platform object to copy
   */
  platform& operator=(const platform& rhs) = default;

  /** Destroys the implementation object.
   */
  ~platform() = default;

  /**
  @brief Checks for equality with another platform instance.
  @return Boolean specifying whether the provided platform if this platform is
  equal to the the provided platform.
  */
  friend bool operator==(const platform& lhs, const platform& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  friend inline bool operator!=(const platform& lhs, const platform& rhs) {
    return !(lhs == rhs);
  }

  /** Returns the underlying cl_platform_id object.
   * @return The cl_platform_id object usable by the OpenCL API.
   */
  cl_platform_id get() const;

  /** Returns the underlying cl_platform_id object without checking if the
   * system is host or device.
   * @return The cl_platform_id object usable by the OpenCL API.
   */
  cl_platform_id get_no_retain() const noexcept;

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Returns the backend associated with the platform.
   * @return Backend associated with the platform.
   */
  inline backend get_backend() const { return this->get_backend_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** Specifies whether the platform is a host platform.
   * @return True if the platform is a host platform.
   */
  bool is_host() const;

  /** Get OpenCL information for the underlying cl_platform_id.
   * @tparam param The \ref info::platform descriptor parameter.
   * @return The retrieved information as the appropriate return type,
   * derived via the param_traits struct, defined in param_traits.h.
   */
  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type get_info()
      const {
    // Call the get method of the info struct, with the cl_device_info
    // template argument.
    cl_platform_id platform_id = nullptr;
    if (!this->is_host()) {
      platform_id = get();
    }
    return get_sycl_info<info::platform,
                         typename opencl_platform_info<param>::sycl_type,
                         typename opencl_platform_info<param>::cl_type,
                         opencl_platform_info<param>::cl_param>(
        platform_id, this->is_host());
  }

  /** This function avoids using strings across the ABI.
   * @copydoc platform::has_extension(const string_class& extension)
   */
  bool has_extension(const char* extension) const;

  /** Check whether a specific extension is supported on the platform.
   * @param extension A string specifying the extension to check for.
   * @return A boolean specifying whether the extension is supported by the
   * platform.
   */
  inline bool has_extension(const string_class& extension) const {
    return has_extension(extension.c_str());
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  /** @brief Returns true if all of the devices associated with the platform
   * support the specified aspect.
   * @asp The aspect to be queried for.
   */
  inline bool has(aspect_impl asp) const { return this->has_impl(asp); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** Get a list of devices associated with the platform.
   * @param deviceType The type of devices to search for, set to
   * info::device_type::all
   * by default
   * @return A vector of device objects.
   */
  vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all) const;

  /** Get a list of all available platforms.
   * @return A vector of platform objects. The returned vector will always
   * contain a host platform.
   */
  static vector_class<platform> get_platforms();

  /** @internal
   * Returns a shared pointer to the implementation object.
   * @return A shared pointer to the implementation object.
   */
  dplatform_shptr get_impl() const;

 private:
  /** Get a cached null terminated string of the platform vendor. */
  const char* get_vendor_cstr() const noexcept;
  /** Get a cached null terminated string of the platform name. */
  const char* get_name_cstr() const noexcept;

  /** Returns the SYCL backend
   * @return Backend associated with the platform
   */
  backend get_backend_impl() const;

 protected:
  /** @brief Returns true if all of the devices associated with the platform
   * support the specified aspect.
   * @asp The aspect to be queried for.
   */
  bool has_impl(aspect_impl asp) const;

  /** Implementation object
   */
  dplatform_shptr m_impl;
};

// The cached get_info calls returning a string must go through a *_cstr()
// proxy function to ensure that the string type is not included in the library
// ABI.

/** @copydoc platform::get_info() */
template <>
inline typename info::param_traits<info::platform,
                                   info::platform::name>::return_type
platform::get_info<info::platform::name>() const {
  return this->get_name_cstr();
}

/** @copydoc platform::get_info() */
template <>
inline typename info::param_traits<info::platform,
                                   info::platform::vendor>::return_type
platform::get_info<info::platform::vendor>() const {
  return this->get_vendor_cstr();
}

/**
@brief This function converts a cl platform object to a sycl platform object
it is used to allow get_info<cl::sycl::device::platform> to return a
sycl platform object
*/
template <>
struct info_convert<cl_platform_id*, platform> {
  static platform cl_to_sycl(cl_platform_id* clValue, size_t /*numElems*/,
                             cl_uint /*clParam*/) {
    return platform(*clValue);
  }
};
}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the platform class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <>
struct hash<cl::sycl::platform> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::platform& object) const {
    hash<cl::sycl::dplatform_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std

#endif  // RUNTIME_INCLUDE_SYCL_PLATFORM_H_

////////////////////////////////////////////////////////////////////////////////
