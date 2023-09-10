/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file exception_list.h
 *
 * @brief Provides SYCL exception list types.
 */
#ifndef RUNTIME_INCLUDE_SYCL_EXCEPTION_LIST_H_
#define RUNTIME_INCLUDE_SYCL_EXCEPTION_LIST_H_

#include "SYCL/base.h"

#include "computecpp_export.h"

namespace cl {
namespace sycl {
/** @brief Class used to store exception objects and transfer them across
 * thread, equivalent to std::exception_ptr.
 */
using exception_ptr_class = std::exception_ptr;

/** @brief List of exceptions thrown asynchronously,
 * contains objects of type exception_ptr_class.
 *
 * The method add_exception has to be called from a derived or
 * friend class, it cannot be accessed directly by the user.
 */
class COMPUTECPP_EXPORT exception_list {
  friend COMPUTECPP_EXPORT exception_list* make_exception_list();

  friend COMPUTECPP_EXPORT void add_exception_to_list(
      exception_list* el, exception_ptr_class asyncExcp);

 private:
  using _exception_list = vector_class<exception_ptr_class>;

  _exception_list m_exceptionList;

 protected:
  /// @cond COMPUTECPP_DEV
  /** @brief Default constructor, not available to users
   */
  exception_list() = default;

  /** @brief Adds an exception to the list
   * @param asyncExcp exception to add.
   */
  void add_exception(exception_ptr_class asyncExcp);

  /// COMPUTECPP_DEV @endcond
 public:
  /** @brief Type of the list elements
   */
  using value_type = exception_ptr_class;
  /** @brief Reference type to a list element
   */
  using reference = value_type&;
  /** @brief Constant reference type to a list element
   */
  using const_reference = const value_type&;
  /** @brief Type of the size of the list
   */
  using size_type = std::size_t;
  /** @brief iterator definition
   */
  using iterator = _exception_list::iterator;
  /** @brief Constant iterator definition
   */
  using const_iterator = _exception_list::const_iterator;

  /** @brief Number of reported errors.
   * @return the number of errors
   */
  size_type size() const;
  /** @return The head of the error list
   */
  const_iterator begin() const;
  /** @return The sentinel value representing the end of the error list
   */
  const_iterator end() const;
};

/** @brief async_handler type definition. This is the type expected by a
 * \ref device to report asynchronous errors.
 */
using async_handler = cl::sycl::function_class<void(exception_list)>;
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_EXCEPTION_LIST_H_
