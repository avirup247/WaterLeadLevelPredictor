#ifndef RUNTIME_INCLUDE_SYCL_FINAL_DATA_H_
#define RUNTIME_INCLUDE_SYCL_FINAL_DATA_H_

#include "SYCL/common.h"  // IWYU pragma: keep

#include <cstddef>
#include <functional>
#include <memory>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
namespace detail {
class storage_mem;

/** @brief A shallow abstraction of \p detail::storage_mem, this class exposes
 * copying functionality to enable copying from a storage_mem object to
 * user-defined final data destinations.
 */
class COMPUTECPP_EXPORT final_data_handler {
 private:
  /** @brief Construct a final_data_handler object.
   * @param impl The \p detail::storage_mem object that this object will expose
   * access to.
   */
  explicit final_data_handler(const detail::storage_mem* impl);

 public:
  friend class detail::storage_mem;

  /** @brief Get the size of the buffer that will be copied into the final data
   * destination in bytes.
   */
  size_t get_size() const;

  /** @brief Get a pointer to the \p detail::storage_mem object's internal
   * memory.
   */
  void* get_host_pointer() const;

  /** @brief Copy from the storage_mem object into the internal host buffer.
   */
  void copy_to_internal() const;

  /** @brief Copy from the storage_mem object into a new destination.
   * @param data A type-erased pointer to the final data destination.
   */
  void copy_back(void* data) const;

  /** @brief Copy from the storage_mem object into a new destination.
   * @param data A type-erased weak_pointer_class object that contains a pointer
   * to the final data destination.
   */
  void copy_back(weak_ptr_class<void> data) const;

 private:
  const detail::storage_mem* m_impl;
};

/** @brief A function wrapper that checks to see if the final data destination
 * of a storage_mem object is null.
 */
using is_final_data_null_t = std::function<bool()>;

/** @brief A function wrapper that enables a storage_mem object to be copied
 * into a final data destination.
 */
using final_data_copy_t = std::function<void(const final_data_handler& writer)>;

/** @brief A wrapper around function objects for interacting with the final data
 * destination of a storage_mem object. Uses type erasure to decouple
 * storage_mem from any specific type of destination.
 */
class COMPUTECPP_EXPORT final_data {
 public:
  /** @brief Set the function callback that will be responsible for copying the
   * buffer contents to the final data destination.
   * @param func The function callback.
   * @return A reference to this to enable function chaining.
   */
  final_data& on_copy_back(final_data_copy_t&& func);

  /** @brief Set the function callback that will be responsible for null
   * checking the final data destination. The function must return true if the
   * data location is null.
   * @param func The function callback.
   * @return A reference to this to enable function chaining.
   */
  final_data& on_null_check(is_final_data_null_t&& func);

  /** @brief Call the copy function.
   * @param writer The final_data_handler object that exposes internal copying
   * functionality.
   */
  void invoke(const final_data_handler& writer) const;

  /** @brief Check if the final data destination is null. Internally, this
   * invokes the function registered with \p on_null_check.
   * @return True if the final data destination is null, otherwise false.
   */
  bool is_final_data_null() const;

 private:
  /** @brief A function wrapper that enables a storage_mem object to be copied
   * into a final data destination. Registered using the \p on_copy_back member
   * function.
   */
  final_data_copy_t m_finalDataCallback;
  /** @brief A function wrapper that checks to see if the final data destination
   * of a storage_mem object is null. Registered using the \p on_null_check
   * member function.
   */
  is_final_data_null_t m_isFinalDataNullCallback;
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_FINAL_DATA_H_
