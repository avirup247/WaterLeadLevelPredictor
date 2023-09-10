/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file use_onchip_memory.h
 *
 * @brief Defines the Codeplay property extension use_onchip_memory.
 */

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_USE_ONCHIP_MEMORY_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_USE_ONCHIP_MEMORY_H_

#include "SYCL/codeplay/property/property_tags.h"
#include "SYCL/property.h"

namespace cl {
namespace sycl {
namespace codeplay {
namespace property {
namespace buffer {
/** @brief Determines if specialised on-chip memory is to be used or not.
 */
class COMPUTECPP_EXPORT use_onchip_memory : public detail::property_base {
 private:
  enum class state : bool;

 public:
  /** @brief Constructs an object of type use_onchip_memory, indicating that the
   * property is required for functionality.
   */
  explicit use_onchip_memory(::cl::sycl::detail::require_tag) noexcept
      : use_onchip_memory(state::required) {}

  /** @brief Constructs an object of type use_onchip_memory, indicating that the
   * property is preferred (but not required) for functionality.
   */
  explicit use_onchip_memory(::cl::sycl::detail::prefer_tag) noexcept
      : use_onchip_memory(state::preferred) {}

  /** @brief Checks if the property is required.
   * @returns `true` if the property is required, `false` otherwise.
   */
  bool is_required() const noexcept { return m_state == state::required; }

  /** @brief Checks if the property is preferred.
   * @returns `true` if the property is preferred, `false` otherwise.
   */
  bool is_preferred() const noexcept { return !is_required(); }

  /** @brief Checks that two use_onchip_memory properties are both required or
   * are both preferred.
   * @param x A property to check.
   * @param y A property to check.
   * @returns `true` if `x` and `y` are both required, `true` if `x` and `y` are
   * both preferred, `false` otherwise.
   */
  friend bool operator==(const use_onchip_memory x,
                         const use_onchip_memory y) noexcept {
    return x.m_state == y.m_state;
  }

  /** @brief Checks that two use_onchip_memory properties are not both required
   * and are not both preferred.
   * @param x A property to check.
   * @param y A property to check.
   * @returns !(x == y)
   */
  friend bool operator!=(const use_onchip_memory x,
                         const use_onchip_memory y) noexcept {
    return !(x == y);
  }

 private:
  enum class state : bool { required, preferred };
  state m_state;

  explicit use_onchip_memory(const state s) noexcept
      : detail::property_base(detail::property_enum::use_onchip_memory),
        m_state(s) {}
};
}  // namespace buffer
}  // namespace property
}  // namespace codeplay
#if SYCL_LANGUAGE_VERSION >= 202001
/** Property trait specializations
 */
template <>
struct is_property<codeplay::property::buffer::use_onchip_memory>
    : public std::true_type {};

template <typename T, int dimensions, typename AllocatorT>
struct is_property_of<codeplay::property::buffer::use_onchip_memory,
                      buffer<T, dimensions, AllocatorT>>
    : public std::true_type {};

#endif  // SYCL_LANGUAGE_VERSION >= 202001
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_USE_ONCHIP_MEMORY_H_
