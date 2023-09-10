/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file property_tags.h
 *
 * @brief Defines tags for Codeplay property extensions.
 */

#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_PROPERTY_TAGS_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_PROPERTY_TAGS_H_

#include "computecpp/gsl/gsl"

namespace cl {
namespace sycl {
namespace detail {
/** @brief Base class for all property tags.
 *
 * This class should permanently remain empty and be _privately_ inherited from
 * any property tags that are created. This enables the empty base class
 * optimisation (EBO), which is explained on C++ Reference.
 *
 * @see https://en.cppreference.com/w/cpp/language/ebo
 */
struct basic_property_tag {};

/** @brief A tag to indicate that a property is required.
 */
struct require_tag : private basic_property_tag {};

/** @brief A tag to indicate that a property is preferred.
 */
struct prefer_tag : private basic_property_tag {};
}  // namespace detail

namespace codeplay {
namespace property {
// This mechanism allows for a pseudo-C++11 inline constexpr variable: uses EBO
// and inline namespace for internal linkage.
// NOLINTNEXTLINE(cert-dcl59-cpp)
inline namespace {
/** An object to pass to a property to say that it is required.
 */
// NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr auto& require = computecpp::gsl::pre_cpp17_constexpr_global<
    ::cl::sycl::detail::require_tag>::value;

/** An object to pass to a property to say that it is preferred.
 */
// NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr auto& prefer = computecpp::gsl::pre_cpp17_constexpr_global<
    ::cl::sycl::detail::prefer_tag>::value;
}  // namespace
}  // namespace property
}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_PROPERTY_PROPERTY_TAGS_H_
