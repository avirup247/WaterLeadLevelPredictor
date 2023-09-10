/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file property.h

  \brief This file
 */

#ifndef RUNTIME_INCLUDE_SYCL_PROPERTY_H_
#define RUNTIME_INCLUDE_SYCL_PROPERTY_H_

#include "SYCL/base.h"
#include "SYCL/predefines.h"
#include "SYCL/type_traits.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

namespace detail {
namespace profiling {
class json_profiler;
}  // namespace profiling

class queue;

/** @brief Check whether type T is contained within the provided list of types.
 *        This is the escape condition, T is not duplicated in an empty list.
 * @tparam T Type to check
 * @tparam Types List of types to check whether it contains T
 */
template <typename T, typename... Types>
struct is_duplicated {
  static constexpr bool value = false;
};

/** @brief Check whether type T is contained withing the provided list of types.
 *        This specialization performs the actual check.
 * @tparam T Type to check
 * @tparam Head First type of the list of types
 * @tparam Tail Remaining types of the list
 */
template <typename T, typename Head, typename... Tail>
struct is_duplicated<T, Head, Tail...> {
  static constexpr bool value = (detail::is_same_basic_type<T, Head>::value ||
                                 is_duplicated<T, Tail...>::value);
};

/** @brief Internal enum of all the available properties
 */
enum class property_enum {
  use_host_ptr,
  use_mutex,
  use_onchip_memory,
  context_bound,
  enable_profiling,
  host_access,
  in_order,
  initialize_to_identity,
};

/** @brief Retrieves the enum value associated with the property type
 * @tparam propertyT Type of the property
 * @return Enum value associated with the property
 */
template <typename propertyT>
COMPUTECPP_EXPORT property_enum get_property_enum();

/** @brief Base class for all the different properties
 */
class COMPUTECPP_EXPORT property_base {
 public:
  /** @brief Constructs the property
   * @param property Enum value associated with this property type
   */
  property_base(property_enum property) : m_property(property) {}

  /** @brief Retrieves the enum value associated with the property
   * @return Enum value of the property
   */
  inline property_enum get_property_enum() const { return m_property; }

 private:
  /** @brief Enum value representing the property type
   */
  property_enum m_property;
};

/** @brief Checks whether the type parameter pack contains only properties.
 *        This is the escape condition, an empty pack is valid.
 * @tparam Tail List of types
 */
template <typename... Tail>
struct contains_properties {
  static constexpr bool value = true;
};
/** @brief Checks whether the type parameter pack contains only properties.
 *        This is the general case, the first type and all the others need to
 *        derive from property_base.
 * @tparam Head First type
 * @tparam Tail List of remaining types
 */
template <typename Head, typename... Tail>
struct contains_properties<Head, Tail...> {
  using property_decayed_t = detail::decay_t<Head>;
  static constexpr bool value =
      (std::is_base_of<property_base, property_decayed_t>::value &&
       contains_properties<Tail...>::value);
};

/** @brief Adds properties to a list. This is the escape condition, when there
 *        are no more properties to add.
 * @tparam propertyTN List of property types
 */
template <typename... propertyTN>
struct add_properties {
  /** @brief Doesn't do anything, just an escape condition
   */
  static void apply(vector_class<dproperty_shptr>&) {}
};

/** @brief Adds properties to a list. This specialization performs the actual
 *        step of adding properties.
 * @tparam propertyT Type of first property in the list
 * @tparam propertyTN Types of the remaining properties in the list
 */
template <typename propertyT, typename... propertyTN>
struct add_properties<propertyT, propertyTN...> {
  /** @brief Adds properties to the property list
   * @param properties List of properties to be expanded
   * @param first The first property that needs to be added
   * @param newProperties The remaining properties that need to be added
   */
  static void apply(vector_class<dproperty_shptr>& properties,
                    propertyT&& first, propertyTN&&... newProperties) {
    using property_decayed_t = detail::decay_t<propertyT>;
    static_assert(
        !is_duplicated<propertyT, propertyTN...>::value,
        "ComputeCpp: Cannot specify more than one property of the same type.");
    // Copy/move and store the property
    properties.emplace_back(
        new property_decayed_t(std::forward<propertyT>(first)),
        std::default_delete<property_decayed_t>());
    add_properties<propertyTN...>::apply(
        properties, std::forward<propertyTN>(newProperties)...);
  }
};

}  // namespace detail

#if SYCL_LANGUAGE_VERSION >= 202001
/** @brief Trait used to check if a type is a property.
 * SYCL properties and vendor implementation specific properties specialize and
 * inherit from std::true_type.
 *
 */
template <typename propertyT>
struct is_property : public std::false_type {};

/** @brief Trait used to check if propertyT is a property useable in the
 * construction of syclObjectT. Specializations that inherit from std::true_type
 * exist for both SYCL and vendor implementation specific properties/objects.
 *
 */
template <typename propertyT, typename syclObjectT>
struct is_property_of : public std::false_type {};

#endif  // SYCL_LANGUAGE_VERSION >= 202001

#if SYCL_LANGUAGE_VERSION >= 202002
/** @brief Helper variable containg the value of is_property<propertyT>
 */
template <typename propertyT>
inline constexpr bool is_property_v = is_property<propertyT>::value;

/** @brief Helper variable containg the value of
 * is_property_of<propertyT, syclObjectT>
 */
template <typename propertyT, typename syclObjectT>
inline constexpr bool is_property_of_v =
    is_property_of<propertyT, syclObjectT>::value;
#endif  // SYCL_LANGUAGE_VERSION >= 202002

/** @brief Storage class for different properties
 */
class COMPUTECPP_EXPORT property_list {
 public:
  /** @brief Construct a SYCL property_list with zero or more properties
   * @tparam propertyTN Types of properties
   * @tparam COMPUTECPP_ENABLE_IF This ensures the constructor only accepts
   *         valid property types.
   * @param props Values of properties
   */
  template <typename... propertyTN,
            COMPUTECPP_ENABLE_IF(
                void, (detail::contains_properties<propertyTN...>::value))>
  property_list(propertyTN&&... props) {
    m_base.reserve(sizeof...(props));
    detail::add_properties<propertyTN...>::apply(
        m_base, std::forward<propertyTN>(props)...);
  }

  /** @internal
   * @brief Constructs a SYCL property_list from a pre-computed sequence of
   * properties.
   * @param properties The sequence of properties to construct from.
   */
  explicit property_list(vector_class<dproperty_shptr> properties)
      : m_base{std::move(properties)} {}

  /// @cond COMPUTECPP_DEV

  /** @internal
   * @brief Returns whether the list of properties contains the property
   *        specified by propertyT
   * @tparam propertyT Property to check for
   * @return True if property list contains the requested property
   */
  template <typename propertyT>
  inline bool has_property() const noexcept {
    return has_property(detail::get_property_enum<propertyT>());
  }

  /** @internal
   * @brief Retrieves the property specified by propertyT from the list. Throws
   *        an error if the list does not contain the property.
   * @tparam propertyT Property to retrieve
   * @return Property from the list
   */
  template <typename propertyT>
  inline propertyT get_property() const {
    return *static_cast<propertyT*>(
        get_property(detail::get_property_enum<propertyT>()));
  }

  /** @internal
   * @brief Returns whether the list of properties contains the property
   *        specified by \ref{requested}
   * @param requested Property to check for
   * @return True if property list contains the requested property
   */
  bool has_property(detail::property_enum requested) const;

  /** @internal
   * @brief Retrieves the property specified by \ref{requested} from the list.
   *        Throws an error if the list does not contain the property.
   * @param requested Property to retrieve
   * @return Property from the list
   */
  detail::property_base* get_property(detail::property_enum requested) const;

 private:
  /** @brief Underlying type for storing a collection of properties.
   */
  vector_class<dproperty_shptr> m_base;

  /** @internal
   * @brief Forces emission of ~propety_list() on Clang family of
   *        compilers. No call to function necessary.
   */
  inline void detail_emit_property_list_dtor_clang() { property_list pl; }

  friend class detail::profiling::json_profiler;
  friend class detail::queue;

  /** @brief Provides access to the underlying storage.
   * @returns An reference to the underlying storage.
   */
  vector_class<dproperty_shptr>& base() { return m_base; }
  const vector_class<dproperty_shptr>& base() const { return m_base; }
  /// COMPUTECPP_DEV @endcond
};

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_PROPERTY_H_

////////////////////////////////////////////////////////////////////////////////
