//
// Copyright (C) 2002-2018 Codeplay Software Limited
// All Rights Reserved.
//
#ifndef RUNTIME_INCLUDE_GSL_DETAIL_NOT_NULL_H_
#define RUNTIME_INCLUDE_GSL_DETAIL_NOT_NULL_H_

#include "computecpp/gsl/impl/type_traits.h"
#include <cassert>
#include <cstddef>
#include <iterator>

namespace computecpp {
namespace gsl {
template <typename T>
class not_null;

/** @brief A fancy pointer designed to ensure that the pointer it encapsulates
 * is never null.
 *
 * not_null prohibits pointer arithmetic, and cannot be implicitly converted
 * to T*. If a conversion to T* is necessary, use not_null<T*>::get. As
 * not_null<T*> is not Nullable, it is not possible to perform comparisons with
 * std::nullptr_t. not_null<T*> is convertible to not_null<const T*>; the
 * reverse is not possible.
 * @tparam The type of the pointer. not_null<T> is ill-formed, except when
 * std::is_pointer<T>::value is true.
 * @note const not_null<T*> is not the same as not_null<const T*>. Should a
 * non-Nullable pointer-to-const be required, use the latter. If a constant
 * pointer is required, use the former. Excluding assignment, all member
 * functions are specified as const, as it is possible for a constant pointer to
 * have its addressee modified.
 * @note not_null<T*> is trivially copyable.
 */
template <typename T>
class not_null<T*> {
 public:
  using value_type = typename std::iterator_traits<T*>::value_type;
  using difference_type = typename std::iterator_traits<T*>::difference_type;
  using pointer = typename std::iterator_traits<T*>::pointer;
  using reference = typename std::iterator_traits<T*>::reference;

  not_null(std::nullptr_t) = delete;

  /** @brief Constructs a not_null<T*> from a pointer.
   * @param p A pointer to an object of type T.
   * @pre p != nullptr
   */
  explicit not_null(const pointer p) noexcept : m_pointer{p} {
    assert(m_pointer != nullptr);
  }

  /** @brief Reseats the pointer
   * @param other The reseat target.
   * @pre p != nullptr
   * @ensures *this != nullptr
   * @returns *this
   */
  not_null& operator=(const pointer other) noexcept {
    *this = not_null{other};
    return *this;
  }

  /** @brief Dereferences the pointer. Guaranteed to be valid, provided the
   * pointer is pointing to valid memory.
   * @return A reference to the pointed-to-object.
   * @note As a const not_null<T> is equivalent to T* const, obtaining a
   * reference-to-const is only
   *       possible if T is const.
   */
  reference operator*() const noexcept { return *m_pointer; }

  /** @brief Equivalent to dereferencing and then using the dot operator.
   *
   * Guaranteed to be valid provided the pointer is pointing to valid memory.
   * @tparam T2 Used for internal purposes, to ensure that SFINAE happens on T.
   * @requires value_type is a composite type.
   * @note As a const not_null<T> is equivalent to T* const, obtaining a
   * pointer-to-const is only possible if T is const.
   */
  template <typename T2 = value_type>
  auto operator-> () const noexcept
      -> enable_if_t<!std::is_fundamental<T2>::value, pointer> {
    return m_pointer;
  }

  /** @brief Returns the value of the pointer.
   */
  pointer get() const noexcept { return m_pointer; }

  /** @brief Converts not_null<T*> to not_null<const T*>.
   */
  operator not_null<const T*>() const noexcept {
    return not_null<const T*>{m_pointer};
  }

  /** @brief Checks that a is equivalent to b.
   * @param a A pointer that is not null.
   * @param b A pointer that might be null.
   */
  friend bool operator==(const not_null a, const T* b) noexcept {
    return a.m_pointer == b;
  }

  /** @brief Checks that a is equivalent to b.
   * @param a A pointer that might be null.
   * @param b A pointer that is not null.
   */
  friend bool operator==(const T* a, const not_null b) noexcept {
    return b == a;
  }

  /** @brief Checks that a is equivalent to b.
   * @param a A pointer that is not null.
   * @param b A pointer that is not null.
   */
  friend bool operator==(const not_null a, const not_null b) noexcept {
    return a.m_pointer == b;
  }

  /** @brief Checks that a is equivalent to b.
   * @param a A pointer that is not null.
   * @param b A pointer that might be null.
   */
  friend bool operator!=(const not_null a, const T* b) noexcept {
    return !(a == b);
  }

  /** @brief Checks that a is equivalent to b.
   * @param a A pointer that might be null.
   * @param b A pointer that is not null.
   */
  friend bool operator!=(const T* a, const not_null b) noexcept {
    return !(a == b);
  }

  /** @brief Checks that a is not equivalent to b.
   * @param a A pointer that is not null.
   * @param b A pointer that is not null.
   */
  friend bool operator!=(const not_null a, const not_null b) noexcept {
    return !(a == b);
  }

 private:
  pointer m_pointer;
};

template <typename T>
bool operator==(const not_null<T>&, std::nullptr_t) = delete;

template <typename T>
bool operator==(std::nullptr_t, const not_null<T>&) = delete;

template <typename T>
bool operator!=(const not_null<T>&, std::nullptr_t) = delete;

template <typename T>
bool operator!=(std::nullptr_t, const not_null<T>&) = delete;
}  // namespace gsl
}  // namespace computecpp

#endif  // RUNTIME_INCLUDE_GSL_DETAIL_NOT_NULL_H_
