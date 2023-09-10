/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file usm_wrapper.h
 *
 * @brief Helpers for wrapping a USM pointer
 */

#ifndef RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_WRAPPER_H_
#define RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_WRAPPER_H_

#include "SYCL/base.h"
#include "SYCL/device.h"
#include "SYCL/experimental/usm_definitions.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/predefines.h"

#include <cstddef>

/// @cond COMPUTECPP_DEV

namespace cl {
namespace sycl {

class context;
class queue;

namespace detail {
/** @brief Base type for wrapping USM pointers to be used in a kernel
 */
class usm_wrapper_base {
 public:
  /// Generic global pointer type without the original type
#ifdef __COMPUTECPP_NO_ASP__
  using void_pointer_t = void*;
#else   // __COMPUTECPP_NO_ASP__
  using void_pointer_t = COMPUTECPP_CL_ASP_GLOBAL void*;
#endif  // __COMPUTECPP_NO_ASP__

  /** @brief Constructs a null pointer
   */
  constexpr usm_wrapper_base(std::nullptr_t = nullptr) noexcept
      : m_pointer{nullptr} {}

  /** @brief Constructs a USM pointer from the user provided pointer
   * @param pointer Raw pointer without an address space
   */
  usm_wrapper_base(void* pointer)
      : m_pointer{global_ptr<void>{pointer}.get()} {}

  /** @brief Retrieves the stored pointer
   * @return Stored USM pointer
   */
  void_pointer_t get_void_ptr() const noexcept { return m_pointer; }

  /** Implicit conversion to a void pointer
   * @return Stored void USM pointer
   */
  operator void_pointer_t() const noexcept { return this->get_void_ptr(); }

#if defined(__SYCL_DEVICE_ONLY__) && !__COMPUTECPP_NO_ASP__
  // These implicit conversions are only used to allow host code conversions,
  // where the address space can be ignored.

  operator void*() const noexcept;
  operator const void*() const noexcept;
#endif  // defined(__SYCL_DEVICE_ONLY__) && !__COMPUTECPP_NO_ASP__

 protected:
  /** @brief Sets the pointer to a new value
   * @param pointer New pointer value
   */
  void set_void_ptr(void_pointer_t pointer) noexcept { m_pointer = pointer; }

 private:
  /// Underlying global pointer that provides access
  void_pointer_t m_pointer;
};
}  // namespace detail

COMPUTECPP_INLINE_EXPERIMENTAL
namespace experimental {

/** @brief Helper class for wrapping USM pointers to be used in a kernel
 *
 * Provides mostly the same interface as if using a raw pointer.
 *
 * @tparam T Underlying type
 */
template <class T>
class usm_wrapper : public detail::usm_wrapper_base {
 public:
  using base_t = usm_wrapper_base;
#ifdef __COMPUTECPP_NO_ASP__
  using element_t = T;
#else   // __COMPUTECPP_NO_ASP__
  using element_t = COMPUTECPP_CL_ASP_GLOBAL T;
#endif  // __COMPUTECPP_NO_ASP__

  using pointer_t = element_t*;

 public:
  /** @brief Constructs a null pointer
   */
  constexpr usm_wrapper(std::nullptr_t = nullptr) noexcept : base_t{nullptr} {}

  /** @brief Constructs a USM pointer from the user provided pointer
   * @param pointer Raw pointer without an address space
   */
  usm_wrapper(T* pointer) : base_t{pointer} {}

  /** @brief Constructs a USM pointer from the user provided void pointer
   * @param pointer Raw void pointer without an address space
   */
  usm_wrapper(void* pointer) : base_t{pointer} {}

  /** @brief Retrieves the underlying raw pointer
   * @return Stored USM pointer
   */
  pointer_t get() const noexcept {
    return static_cast<pointer_t>(this->get_void_ptr());
  }

  /** Implicit conversion to the underlying raw pointer
   * @return Stored USM pointer
   */
  operator pointer_t() const noexcept { return this->get(); }

  auto operator*() noexcept -> decltype(*this->get()) { return *this->get(); }

  auto operator*() const noexcept -> decltype(*this->get()) {
    return *this->get();
  }

  auto operator-> () const noexcept -> decltype(*this->get()) {
    return *this->get();
  }

  auto operator[](std::ptrdiff_t idx) const -> decltype(*(this->get() + idx)) {
    return *(this->get() + idx);
  }

  explicit operator bool() const noexcept {
    return static_cast<bool>(this->get());
  }

  usm_wrapper& operator+=(std::ptrdiff_t index) noexcept {
    this->set_void_ptr(this->get() + index);
    return *this;
  }
  friend usm_wrapper operator+(usm_wrapper lhs, std::ptrdiff_t index) noexcept {
    lhs += index;
    return lhs;
  }

/** @brief Defines logical operators for the USM wrapper class
 * @param op Operator symbol
 */
#define COMPUTECPP_USM_WRAPPER_LOGICAL_OP(op)                                  \
  friend inline bool operator op(const usm_wrapper& lhs,                       \
                                 const usm_wrapper& rhs) noexcept {            \
    return (lhs.get() op rhs.get());                                           \
  }                                                                            \
  friend inline bool operator op(const usm_wrapper& lhs,                       \
                                 std::nullptr_t) noexcept {                    \
    return (lhs.get() op nullptr);                                             \
  }                                                                            \
  friend inline bool operator op(std::nullptr_t,                               \
                                 const usm_wrapper& rhs) noexcept {            \
    return (nullptr op rhs.get());                                             \
  }

  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(==)
  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(!=)
  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(<)
  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(<=)
  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(>)
  COMPUTECPP_USM_WRAPPER_LOGICAL_OP(>=)

#undef COMPUTECPP_USM_WRAPPER_LOGICAL_OP

} COMPUTECPP_CONVERT_ATTR_USM_WRAPPER;

#ifdef __SYCL_DEVICE_ONLY__

// Templated declarations of USM functions allow the device compiler
// to see different address spaces
// and prevent it from failing to parse host code.
// The functions are not valid inside kernels anyway.

COMPUTECPP_EXPORT void free(void* ptr, const context& ctx);
template <class T>
void free(usm_wrapper<T> ptr, const context& ctx);

COMPUTECPP_EXPORT void free(void* ptr, const queue& q);
template <class T>
void free(usm_wrapper<T> ptr, const queue& q);

COMPUTECPP_EXPORT usm::alloc get_pointer_type(const void* ptr,
                                              const context& ctx);
template <class T>
usm::alloc get_pointer_type(usm_wrapper<T> ptr, const context& ctx);

COMPUTECPP_EXPORT device get_pointer_device(const void* ptr,
                                            const context& ctx);
template <class T>
device get_pointer_device(usm_wrapper<T> ptr, const context& ctx);

#endif  //__SYCL_DEVICE_ONLY__

}  // namespace experimental
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_EXPERIMENTAL_USM_WRAPPER_H_
