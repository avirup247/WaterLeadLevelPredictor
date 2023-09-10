/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file storage_mem.h

  \brief This file implements an internal base class for memory objects.
*/

#ifndef RUNTIME_INCLUDE_SYCL_STORAGE_MEM_H_
#define RUNTIME_INCLUDE_SYCL_STORAGE_MEM_H_

#include "SYCL/base.h"
#include "SYCL/common.h"
#include "SYCL/final_data.h"
#include "SYCL/index_array.h"
#include "SYCL/kernel.h"
#include "SYCL/predefines.h"

#include "computecpp/gsl/impl/type_traits.h"

#include <cstddef>
#include <iterator>
#include <memory>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
namespace detail {
class base_allocator;
}  // namespace detail

enum class write_back : bool {
  disable_write_back = false,
  enable_write_back = true
};

class property_list;

/** @cond COMPUTECPP_DEV */
/** @brief A common base class for memory objects
 */
class COMPUTECPP_EXPORT storage_mem {
 public:
  storage_mem() = default;

  explicit storage_mem(dmem_shptr impl);

  virtual ~storage_mem();

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use byte_size() instead.")
  COMPUTECPP_TEST_VIRTUAL size_t get_size() const;

  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use size() instead.")
  COMPUTECPP_TEST_VIRTUAL size_t get_count() const;

#if SYCL_LANGUAGE_VERSION >= 202001
  inline size_t byte_size() const noexcept { return byte_size_impl(); }

  inline size_t size() const noexcept { return size_impl(); }
#endif  // SYCL_LANGUAGE_VERSION >= 202001

  COMPUTECPP_TEST_VIRTUAL detail::index_array get_range_impl() const;

  COMPUTECPP_TEST_VIRTUAL dmem_shptr get_impl() const;

  COMPUTECPP_TEST_VIRTUAL void set_as_kernel_arg(
      const cl::sycl::kernel& syclKernel, unsigned int index);

  /** @brief Sets where data should be written to on destruction of the buffer
   * @tparam Destination Underlying type of the final data.
   *
   * Can be the same pointer type (T*) as the buffer/image,
   * a void pointer,
   * a weak_ptr to T or void,
   * or an output iterator that values of type T can be copied into.
   * If the destination is null, the final copy will be omitted.
   *
   */
  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = nullptr) {
    this->set_final_data_internal(std::move(finalData));
  }

  COMPUTECPP_TEST_VIRTUAL void set_write_back(bool flag = true);

  /**
  @brief Copy constructor. Copies the reference of the implementation of the rhs
  object
  @param rhs the object whose implementation will be copied
  */
  storage_mem(const storage_mem& rhs);

  /**
  @brief Move constructor. Moves the reference of the implementation of the rhs
  object
  @param rhs the object whose implementation will be moved
  */
  storage_mem(storage_mem&& rhs) noexcept;

  /**
  @brief Copy Assignment Operator. Copies the reference of the implementation of
  the rhs object
  @param rhs the object whose implementation will be copied
  */
  storage_mem& operator=(const storage_mem& rhs) = default;

  /**
  @brief Move Assignment Operator. Moves the reference of the implementation of
  the rhs object
  @param rhs the object whose implementation will be moved
  */
  storage_mem& operator=(storage_mem&& rhs) noexcept;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const storage_mem& lhs,
                                const storage_mem& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const storage_mem& lhs,
                                const storage_mem& rhs) {
    return !(lhs == rhs);
  }

 protected:
  /** @brief Sets the implementation pointer
   */
  void set_impl(dmem_shptr impl);

  /** @brief Retrieves the allocator that was used on construction
   * @return Pointer to the type-erased allocator object
   **/
  detail::base_allocator* get_base_allocator() const;

  /** @brief Retrieves the properties associated with this object
   * @return List of properties
   */
  const property_list& get_properties() const;

 private:
  size_t byte_size_impl() const;

  size_t size_impl() const;

  /** @brief Prepare the final data object
   * @return Reference to the final data object
   */
  detail::final_data& prepare_final_data();

  /** @brief Set a raw pointer as the final data destination
   *        of this storage object.
   * @tparam T The value type that is \p destination points to.
   * @param destination The pointer that will be used as the final data
   *        destination of this storage object.
   */
  template <typename T>
  void set_final_data_internal(T* destination) {
    this->set_final_data_internal(static_cast<void*>(destination));
  }

  /** @brief Set a raw void pointer as the final data destination
   *        of this storage object.
   * @tparam T The value type that is \p destination points to.
   * @param destination The void pointer that will be used as the final data
   *        destination of this storage object.
   */
  void set_final_data_internal(void* destination);

  /** @brief Set an output iterator object as the final data destination of this
   *        storage object.
   * @tparam T The iterator type.
   * @param destination The iterator object that will be used as the final
   *        data destination of this storage object.
   */
  template <typename T>
  void set_final_data_internal(T destination);

  /** @brief Disable copying to a final data destination
   *        for this storage object.
   * @param destination A null pointer.
   *        This signifies that there will be no copy performed
   *        On destruction of the object.
   */
  void set_final_data_internal(std::nullptr_t destination);

  /** @brief Set a weak pointer as the final data destination of this
   *        storage object.
   * @tparam T The underlying type of the weak pointer.
   * @param destination The weak pointer that will be used as the final
   *        data destination of this storage object.
   */
  template <typename T>
  void set_final_data_internal(const weak_ptr_class<T>& destination) {
    this->set_final_data_internal(weak_ptr_class<void>{destination});
  }

  /** @brief Set a weak pointer to void as the final data destination of this
   *        storage object.
   * @param destination The weak pointer that will be used as the final
   *        data destination of this storage object.
   */
  void set_final_data_internal(const weak_ptr_class<void>& destination);

  /** @brief Set a shared pointer as the final data destination of this
   *        storage object.
   * @tparam T The underlying type of the shared pointer.
   * @param destination The shared pointer that will be used as the
   *        final data destination of this storage object.
   */
  template <typename T>
  void set_final_data_internal(const shared_ptr_class<T>& destination) {
    this->set_final_data_internal(weak_ptr_class<T>{destination});
  }

  dmem_shptr m_impl;
};

template <typename T>
void storage_mem::set_final_data_internal(T destination) {
  using value_type = typename std::iterator_traits<T>::value_type;

  static_assert(computecpp::gsl::is_writable<T, value_type>::value,
                "Invalid final data destination: iterator must be writable.");

  this->prepare_final_data()
      .on_copy_back([=](const detail::final_data_handler& handler) {
        // 1/2: copy to internal host buffer
        handler.copy_to_internal();

        // 2/2: copy to range specified by iterator
        if (const auto data = handler.get_host_pointer()) {
          const auto size = handler.get_size() / sizeof(value_type);
          const auto begin = static_cast<const value_type*>(data);
          const auto end = begin + size;
          std::copy(begin, end, destination);
        }
      })
      .on_null_check([=]() { return false; });
}

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_STORAGE_MEM_H_
