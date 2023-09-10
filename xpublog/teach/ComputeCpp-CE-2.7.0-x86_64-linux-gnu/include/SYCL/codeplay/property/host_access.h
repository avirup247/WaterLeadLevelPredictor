/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file host_access.h
 *
 * @brief Defines the Codeplay property extension host_access.
 */

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_HOST_ACCESS_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_HOST_ACCESS_H_

#include "SYCL/property.h"

namespace cl {
namespace sycl {
template <typename T, int dimensions, typename AllocatorT>
class buffer;
namespace codeplay {
/** @brief Access modes that can limit host access to device data
 * @note This only applies to host accessors,
 *       the host device is treated as any other device
 */
enum class host_access_mode {
  none,        ///< No host access to device data allowed
  read,        ///< Host can only read from device data
  read_write,  ///< Host can read and write device data
  write        ///< Host can only write to device data
};

namespace property {
namespace buffer {
/** @brief The host_access property determines
 *        if and how device data can be accessed from the host
 */
class COMPUTECPP_EXPORT host_access : public detail::property_base {
 public:
  /** @brief Constructs a property object using the specified access mode
   * @param hostAccessMode Mode specifying
   *        if and how device data can be accessed from the host
   */
  host_access(host_access_mode hostAccessMode) noexcept
      : detail::property_base(detail::property_enum::host_access),
        m_hostAccessMode(hostAccessMode) {}

  /** @brief Retrieves the access mode as provided on construction
   * @return Host access mode
   */
  host_access_mode get_host_access_mode() const { return m_hostAccessMode; }

 private:
  /** @brief Store the host access mode
   */
  host_access_mode m_hostAccessMode;
};
}  // namespace buffer
}  // namespace property
}  // namespace codeplay
#if SYCL_LANGUAGE_VERSION >= 202001
/** Property trait specializations
 */
template <>
struct is_property<codeplay::property::buffer::host_access>
    : public std::true_type {};

template <typename T, int dimensions, typename AllocatorT>
struct is_property_of<codeplay::property::buffer::host_access,
                      buffer<T, dimensions, AllocatorT>>
    : public std::true_type {};

#endif  // SYCL_LANGUAGE_VERSION >= 202001
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_HOST_ACCESS_H_
