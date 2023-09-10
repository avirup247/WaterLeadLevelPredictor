/**************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

****************************************************************/

/**
  @file queue.h
  @brief This file contains the queue class as define in the SYCL 1.2
  specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_QUEUE_H_
#define RUNTIME_INCLUDE_SYCL_QUEUE_H_

#include "SYCL/apis.h"
#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/command_group.h"
#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/event.h"
#include "SYCL/exception_list.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"
#include "SYCL/predefines.h"
#include "SYCL/property.h"  // IWYU pragma: keep

#include <cstddef>
#include <functional>
#include <memory>
#include <system_error>
#include <type_traits>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

/// COMPUTECPP_DEV @endcond
class device_selector;
/// @cond COMPUTECPP_DEV

namespace info {

/** @brief Queue information descriptors
 */
enum class queue : int {
  reference_count, /**< Query the reference count of the queue */
  device,          /**< Query the device associate to the queue */
  context,         /**< Query the context associate to the queue */
  queue_profiling  /**< Query the if the queue profiling is enabled */
};

}  // namespace info

/// @cond COMPUTECPP_DEV

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(queue, cl_command_queue_info,
                                    cl_command_queue)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(queue, reference_count,
                                      CL_QUEUE_REFERENCE_COUNT, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(queue, context, CL_QUEUE_CONTEXT,
                                      cl::sycl::context, cl_context)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(queue, device, CL_QUEUE_DEVICE,
                                      cl::sycl::device, cl_device_id)

/** Defines get_info<info::queue::queue_profiling>
 * @deprecated Use info::device::queue_profiling instead
 */
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(queue, queue_profiling,
                                                  CL_QUEUE_PROPERTIES, bool,
                                                  cl_command_queue_properties,
                                                  CL_QUEUE_PROFILING_ENABLE)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(queue, reference_count, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(queue, context, cl::sycl::context())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(queue, device, cl::sycl::device())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(queue, queue_profiling, true)

using command_group_functor_t = function_class<void(handler&)>;

/// COMPUTECPP_DEV @endcond

namespace property {
namespace queue {

/** @brief The enable_profiling property adds the requirement that the SYCL
 *        runtime must capture profiling information for the command groups that
 *        are submitted from this SYCL queue and provide said information via
 *        the SYCL event class get_profiling_info member function, if the
 *        associated SYCL device supports queue profiling
 */
class COMPUTECPP_EXPORT enable_profiling : public detail::property_base {
 public:
  enable_profiling()
      : detail::property_base(detail::property_enum::enable_profiling) {}
};

/**
 * @brief Property which enabled in-order scheduling for any command groups
 submitted to the queue.
 */
class in_order_impl : public detail::property_base {
 public:
  COMPUTECPP_EXPORT in_order_impl()
      : detail::property_base(detail::property_enum::in_order) {}
};

#if SYCL_LANGUAGE_VERSION >= 202001

/**
 * @brief Property which enabled in-order scheduling for any command groups
 submitted to the queue.
 */
using in_order = in_order_impl;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace queue
}  // namespace property

class queue;

namespace detail {
template <>
struct opencl_backend_traits<sycl::queue> {
 private:
 public:
  using input_type = cl_command_queue;
  using return_type = input_type;
};

#if SYCL_LANGUAGE_VERSION >= 202001

/** In some cases there are ambiguities
 *  as to what the template argument should convert to.
 *  This function checks what the argument can be converted to
 *  and performs that conversion.
 * @tparam T Type of the argument
 * @param arg Argument passed on queue construction
 * @return Argument converted to a suitable type
 */
template <class T>
auto wrap_queue_template_arg(T&& arg) {
  if constexpr (std::is_convertible_v<T, async_handler>) {
    return async_handler{std::forward<T>(arg)};
  } else {
    return detail::device_selector_wrapper{std::forward<T>(arg)};
  }
}

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace detail

/** The cl::sycl::queue object is the SYCL abstraction of the OpenCL object
 * cl_command_queue.
 *
 * It is responsible for constructing the OpenCL cl_command_queue object and all
 * OpenCL API functions that involve enqueuing. As the cl::sycl::queue object
 * can be constructed using different methods, it maintains the ownership over
 * objects that it can potentially be responsible for constructing and
 * destructing.
 */
class COMPUTECPP_EXPORT queue {
 public:
  /** @brief Constructs a queue using a default device selector.
   * @param propList List of queue properties
   */
  explicit queue(const property_list& propList = {})
      : queue(async_handler{}, propList) {}

  /** @brief Constructs a queue using a default device selector.
   * @param asyncHandler User defined \ref async_handler
   * @param propList List of queue properties
   */
  explicit queue(const async_handler& asyncHandler,
                 const property_list& propList = {});

  /** @brief Constructs a queue using a user defined device selector. The device
   *        selector \ref device_selector::select_device() is called by the
   *        constructor to get the device to construct the queue.
   * @param deviceSelector User defined \ref device_selector
   * @param propList List of queue properties
   */
  explicit queue(const device_selector& deviceSelector,
                 const property_list& propList = {})
      : queue(deviceSelector, async_handler{}, propList) {}

  /** @brief Constructs a queue using a user defined device selector. The device
   *        selector \ref device_selector::select_device() is called by the
   *        constructor to get the device to construct the queue.
   * @param deviceSelector User defined \ref device_selector
   * @param asyncHandler User defined \ref async_handler
   * @param propList List of queue properties
   */
  explicit queue(const device_selector& deviceSelector,
                 const async_handler& asyncHandler,
                 const property_list& propList = {});

  /** @brief Construct a queue from a given device, creating an implicit context
   *        in the process.
   * @param dev The device to use to create the queue
   * @param propList List of queue properties
   */
  explicit queue(const device& dev, const property_list& propList = {})
      : queue(dev, async_handler{}, propList) {}

  /** @brief Construct a queue from a given device, creating an implicit context
   *        in the process.
   * @param dev The device to use to create the queue
   * @param asyncHandler User defined \ref async_handler
   * @param propList List of queue properties
   */
  explicit queue(const device& dev, const async_handler& asyncHandler,
                 const property_list& propList = {});

  /** @brief Constructs a queue using a user defined device selector against a
   *        specific context. The device selector
   *        \ref device_selector::select_device() is called by the constructor
   *        to get the device to construct the queue.
   * @param syclContext Context in which the queue will be created.
   * @param selector Used to get a device from the context
   * @param propList List of queue properties
   */
  explicit queue(const context& syclContext, const device_selector& selector,
                 const property_list& propList = {})
      : queue(syclContext, selector, async_handler{}, propList) {}

  /** @brief Constructs a queue using a user defined device selector against a
   *        specific context. The device selector
   *        \ref device_selector::select_device() is called by the constructor
   *        to get the device to construct the queue.
   * @param syclContext Context in which the queue will be created.
   * @param selector Used to get a device from the context
   * @param propList List of queue properties
   */
  explicit queue(const context& syclContext, const device_selector& selector,
                 const async_handler& asyncHandler,
                 const property_list& propList = {});

  /** @brief Constructs a queue with properties using an existing device and
   * context.
   * @param syclContext The context the queue will belong to.
   * @param dev The device that work will be executed on.
   * @param asyncHandler Handler for asynchronous exceptions
   * @param propList List of properties for the queue to have.
   * @throw sycl_exception If dev is not in syclContext
   */
  explicit queue(const context& syclContext, const device& dev,
                 const async_handler& asyncHandler,
                 const property_list& propList = {});

  /** @brief Constructs a queue using an existing device and context.
   * @param syclContext The context the queue will belong to.
   * @param dev The device work will be queued on.
   * @param propList List of properties for the queue to have.
   * @throw sycl_exception If dev is not in syclContext
   */
  explicit queue(const context& syclContext, const device& dev,
                 const property_list& propList = {})
      : queue(syclContext, dev, async_handler{}, propList) {}

  /** @brief Construct a queue object from a given OpenCL object
   * @param clQueue a valid OpenCL object for a command queue
   * @param s_context a valid OpenCL context
   * @param asyncHandler User defined \ref async_handler
   */
  explicit queue(cl_command_queue clqueue, const context& s_context,
                 const async_handler& asyncHandler = nullptr);

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Constructs a SYCL queue object using a custom device selector callable
   * @tparam DeviceSelector Type of the callable used for device selection.
   * Note that since this is templated, it could be async_handler as well.
   * @param deviceSelector Callable that can evaluate devices
   * @param propList List of queue properties
   */
  template <typename DeviceSelector>
  explicit queue(const DeviceSelector& deviceSelector,
                 const property_list& propList = {})
      : queue{detail::wrap_queue_template_arg(deviceSelector), propList} {}

  /** Constructs a SYCL queue object using a custom device selector callable
   * @param deviceSelector Callable that can evaluate devices
   * @param propList List of queue properties
   */
  explicit queue(const detail::device_selector_wrapper& deviceSelector,
                 const property_list& propList = {})
      : queue{detail::impl_constructor_tag{}, deviceSelector, async_handler{},
              propList} {}

  /** Constructs a SYCL queue object using a custom device selector callable
   * @tparam DeviceSelector Type of the callable used for device selection
   * @param deviceSelector Callable that can evaluate devices
   * @param asyncHandler Handler for asynchronous errors
   * @param propList List of queue properties
   */
  template <typename DeviceSelector>
  explicit queue(const DeviceSelector& deviceSelector,
                 const async_handler& asyncHandler,
                 const property_list& propList = {})
      : queue{detail::impl_constructor_tag{},
              detail::device_selector_wrapper{deviceSelector}, asyncHandler,
              propList} {}

  /** Constructs a SYCL queue object using a custom device selector callable
   * @tparam DeviceSelector Type of the callable used for device selection.
   * Note that since this is templated, it could be async_handler as well.
   * @param syclContext SYCL context to associate with the queue
   * @param deviceSelector Callable that can evaluate devices
   * @param propList List of queue properties
   */
  template <typename DeviceSelector>
  explicit queue(const context& syclContext,
                 const DeviceSelector& deviceSelector,
                 const property_list& propList = {})
      : queue{syclContext, detail::wrap_queue_template_arg(deviceSelector),
              propList} {}

  /** Constructs a SYCL queue object using a custom device selector callable
   * @param syclContext SYCL context to associate with the queue
   * @param deviceSelector Callable that can evaluate devices
   * @param propList List of queue properties
   */
  explicit queue(const context& syclContext,
                 const detail::device_selector_wrapper& deviceSelector,
                 const property_list& propList = {})
      : queue{detail::impl_constructor_tag{}, syclContext, deviceSelector,
              async_handler{}, propList} {}

  /** Constructs a SYCL queue object using a custom device selector callable
   * @tparam DeviceSelector Type of the callable used for device selection
   * @param syclContext SYCL context to associate with the queue
   * @param deviceSelector Callable that can evaluate devices
   * @param asyncHandler Handler for asynchronous errors
   * @param propList List of queue properties
   */
  template <typename DeviceSelector>
  explicit queue(const context& syclContext,
                 const DeviceSelector& deviceSelector,
                 const async_handler& asyncHandler,
                 const property_list& propList = {})
      : queue{detail::impl_constructor_tag{}, syclContext,
              detail::device_selector_wrapper{deviceSelector}, asyncHandler,
              propList} {}

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** @brief Construct a queue object by copying the contents of a
   * given queue object.
   * @param rhs a queue to be copied over to the returned object
   */
  queue(const queue& rhs) = default;

  /** @brief Completely assigns the contents of the queue to that of another
   * queue.
   */
  queue& operator=(const queue& rhs) = default;

  /** @brief Completely moves the contents of a queue to that of another queue.
   */
  queue(queue&& rhs) = default;

  /** @brief Completely moves the contents of a queue to that of another queue.
   */
  queue& operator=(queue&& rhs) = default;

 private:
  /** Constructs a SYCL queue object using a custom device selector callable
   * @param deviceSelector Wrapper around a user callable
   * @param asyncHandler Handler for asynchronous errors
   * @param propList List of queue properties
   */
  explicit queue(detail::impl_constructor_tag,
                 const detail::device_selector_wrapper& deviceSelector,
                 const async_handler& asyncHandler,
                 const property_list& propList);

  /** Constructs a SYCL queue object using a custom device selector callable
   * @param syclContext SYCL context to associate with the queue
   * @param deviceSelector Wrapper around a user callable
   * @param asyncHandler Handler for asynchronous errors
   * @param propList List of queue properties
   */
  explicit queue(detail::impl_constructor_tag, const context& syclContext,
                 const detail::device_selector_wrapper& deviceSelector,
                 const async_handler& asyncHandler,
                 const property_list& propList);

 public:
  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const queue& lhs, const queue& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const queue& lhs, const queue& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Returns whether this SYCL queue was constructed with the property
   *        specified by propertyT
   * @tparam propertyT Property to check for
   * @return True if queue constructed with the property
   */
  template <typename propertyT>
  bool has_property() const noexcept {
    return this->get_properties().template has_property<propertyT>();
  }

  /** @brief Returns a copy of the property of type propertyT that this SYCL
   *        queue was constructed with. Throws an error if the queue was not
   *        constructed with the property.
   * @tparam propertyT Property to retrieve
   * @return Copy of the property
   */
  template <typename propertyT>
  propertyT get_property() const {
    return this->get_properties().template get_property<propertyT>();
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Returns the backend associated with the queue.
   * @return Backend associated with the queue.
   */
  inline backend get_backend() const { return this->get_backend_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** Determine if the queue is executing kernels on the host.
   * @return true if the queue is executing kernels on the host.
   */
  COMPUTECPP_TEST_VIRTUAL bool is_host() const;

  /** Gets OpenCL information for the queue.
   * @tparam param A \ref info::queue descriptor specifying the information to
   * retrieve.
   * @return The retrieved information as the appropriate return type
   */
  template <info::queue param>
  COMPUTECPP_EXPORT typename info::param_traits<info::queue, param>::return_type
  get_info() const;

  /** Returns the context associate to the queue.
   * @return The context object.
   */
  COMPUTECPP_TEST_VIRTUAL context get_context() const;

  /**  Returns the underlying OpenCL cl_command_queue object.
   *  @return an OpenCL cl_command_queue object.
   */
  cl_command_queue get() const;

  /** Performs a blocking wait for the completion of all
   * enqueued tasks in the queue. Synchronous errors
   * are reported through C++ exceptions.
   */
  void wait();

  /** Returns the Device associated with the queue.
   * @return The device object used with the queue.
   */
  device get_device() const;

  /**
   * @brief Enqueues an USM fill operation. Fills the memory pointed by @p ptr.
   * @tparam T The type of the element.
   * @param ptr Pointer object to fill.
   * @param pattern The pattern to fill each element of @p ptr
   * @param count The number of elements of type T to fill.
   * @return A runtime event representing this operation
   */
  template <typename T>
  event fill(void* ptr, const T& pattern, size_t count) {
#if (defined(COMPUTECPP_WINDOWS) || !defined(COMPUTECPP_GCC_PRE_5)) &&         \
    !defined(__SYCL_DEVICE_ONLY__)
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type T needs to be trivially copyable");
#endif
    return fill(ptr, static_cast<const void*>(&pattern), sizeof(T),
                count * sizeof(T));
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  /**
   * @brief Enqueues an USM fill operation. Fills the memory pointed by @p ptr.
   * @tparam T The type of the element.
   * @param ptr Pointer object to fill.
   * @param pattern The pattern to fill each element of @p ptr
   * @param count The number of elements of type T to fill.
   * @return A runtime event representing this operation
   */
  template <typename T>
  event fill(void* ptr, const T& pattern, size_t count, event dependency) {
#if defined(COMPUTECPP_WINDOWS) && !defined(__SYCL_DEVICE_ONLY__)
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type T needs to be trivially copyable");
#endif
    return fill(ptr, static_cast<const void*>(&pattern), sizeof(T),
                count * sizeof(T), {dependency});
  }

  /**
   * @brief Enqueues an USM fill operation. Fills the memory pointed by @p ptr.
   * @tparam T The type of the element.
   * @param ptr Pointer object to fill.
   * @param pattern The pattern to fill each element of @p ptr
   * @param count The number of elements of type T to fill.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @return A runtime event representing this operation
   */
  template <typename T>
  event fill(void* ptr, const T& pattern, size_t count,
             const std::vector<event>& dependencies) {
#if defined(COMPUTECPP_WINDOWS) && !defined(__SYCL_DEVICE_ONLY__)
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type T needs to be trivially copyable");
#endif
    return fill(ptr, static_cast<const void*>(&pattern), sizeof(T),
                count * sizeof(T), dependencies);
  }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

#if SYCL_LANGUAGE_VERSION >= 202002
  /** Enqueues a USM memset operation. Sets the numBytes from
   * memory pointed at by @p ptr to given value interpreted as an unsigned char.
   * This is a utility function that will submit an operation in this queue.
   * @param ptr Pointer to the memory location to write to.
   * @param value The value to set memory to. static_cast to unsigned char.
   * @param numBytes The number of bytes to set from ptr.
   * @return A runtime event representing this operation.
   */
  inline event memset(void* ptr, int value, size_t numBytes) {
    return submit([&](handler& cgh) { cgh.memset(ptr, value, numBytes); });
  }

  /** Enqueues a USM memset operation. Sets the numBytes from
   * memory pointed at by @p ptr to given value interpreted as an unsigned char.
   * This is a utility function that will submit an operation in this queue.
   * @param ptr Pointer to the memory location to write to.
   * @param value The value to set memory to. static_cast to unsigned char.
   * @param numBytes The number of bytes to set from ptr.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @return A runtime event representing this operation.
   */
  inline event memset(void* ptr, int value, size_t numBytes, event depEvent) {
    return submit([&](handler& cgh) {
      cgh.depends_on(depEvent);
      cgh.memset(ptr, value, numBytes);
    });
  }

  /** Enqueues a USM memset operation. Sets the numBytes from
   * memory pointed at by @p ptr to given value interpreted as an unsigned char.
   * This is a utility function that will submit an operation in this queue.
   * @param ptr Pointer to the memory location to write to.
   * @param value The value to set memory to. static_cast to unsigned char.
   * @param numBytes The number of bytes to set from ptr.
   * @param depEvents Event objects which must be complete before the
   * command can be executed.
   * @return A runtime event representing this operation.
   */
  inline event memset(void* ptr, int value, size_t numBytes,
                      const std::vector<event>& depEvents) {
    return submit([&](handler& cgh) {
      cgh.depends_on(depEvents);
      cgh.memset(ptr, value, numBytes);
    });
  }
#endif  // SYCL_LANGUAGE_VERSION >= 202002

  /**
   * @brief Enqueues an USM memcpy operation. Copies @p size bytes from @p src
   * to @p dest. This is a utility function that will submit an operation in
   * this queue.
   * @param dest Pointer to the memory location to copy to.
   * @param src Pointer to the memory location to copy from.
   * @param size The number of bytes to copy.
   * @return A runtime event representing this operation.
   */
  event memcpy(void* dest, const void* src, size_t size);

  /**
   * @brief Enqueues an USM memcpy operation. Copies @p size bytes from @p src
   * to @p dest. This is a utility function that will submit an operation in
   * this queue.
   * @param dest Pointer to the memory location to copy to.
   * @param src Pointer to the memory location to copy from.
   * @param size The number of bytes to copy.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @return A runtime event representing this operation.
   */
  event memcpy(void* dest, const void* src, size_t size, event dependency);

  /**
   * @brief Enqueues an USM memcpy operation. Copies @p size bytes from @p src
   * to @p dest. This is a utility function that will submit an operation in
   * this queue.
   * @param dest Pointer to the memory location to copy to.
   * @param src Pointer to the memory location to copy from.
   * @param size The number of bytes to copy.
   * @param dependencies Event objects which must be complete before the
   * command can be executed.
   * @return A runtime event representing this operation.
   */
  event memcpy(void* dest, const void* src, size_t size,
               const std::vector<event>& dependencies);

  /**
   * @brief Hints to the SYCL runtime that the data is available earlier
   *        than when the USM model would require it.
   *        Can only be overlapped with kernel execution
   *        when Concurrent or System USM is available.
   * @param ptr Pointer to the memory to be prefetched to the device
   * @param size Number of bytes requested to be prefetched
   * @return Event associated with the memory prefetch
   */
  event experimental_prefetch(const void* ptr, size_t size);

  /// @copydoc experimental_prefetch(const void*, size_t)
  inline event prefetch(const void* ptr, size_t size) {
    return this->experimental_prefetch(ptr, size);
  }

  /** Provides the SYCL runtime with information about how the allocation
   *  is used.
   * @param ptr Address of allocation
   * @param size Number of bytes in the allocation
   * @param advice Device-defined advice for the specified allocation
   * @return Event associated with the advise operation
   */
  event experimental_mem_advise(const void* ptr, size_t size, int advice);

  /// @copydoc experimental_mem_advise(const void*, size_t, int)
  inline event mem_advise(const void* ptr, size_t size, int advice) {
    return this->experimental_mem_advise(ptr, size, advice);
  }

  /// @cond COMPUTECPP_DEV
  /** Returns the implementation of the class.
   * @internal
   */
  COMPUTECPP_TEST_VIRTUAL dqueue_shptr get_impl() const;

  /** Creates a queue using a specific implementation object.
   * @internal
   * @param detail::queue impl Implementation to use for the queue.
   */
  explicit queue(dqueue_shptr impl);

  /// COMPUTECPP_DEV @endcond

  /** Performs a blocking wait for the completion of all
   * enqueued tasks in the queue. Synchronous errors
   * are reported through C++ exceptions. Asynchronous errors are reported to
   * the async_handler if provided (lost otherwise)
   */
  void wait_and_throw();

  /** Report any unreported asynchronous errors via the
   * async_handler if provided (lost otherwise)
   */
  void throw_asynchronous();

#if SYCL_LANGUAGE_VERSION >= 202001

  /**
   * @brief Returns true if the queue was constructed with the
   * property::queue::in_order property.
   */
  inline bool is_in_order() const { return this->is_in_order_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** Submits a command group functor to execution.
   * @tparam T The command group type
   * @param cgf The command group functor
   */
  template <typename T>
  inline event submit(T cgf) {
    auto cg = detail::command_group(this->get_impl());
    return cg.submit_handler(
        cgf, nullptr,
        detail::get_handler_tag(cgf, detail::select_handler_overload{}));
  }

  /** Submits a command group functor to execution with
   * a fallback queue. If an error occur during the execution of the kernel on
   * the current queue, the runtime will try to run the kernel on the fallback
   * queue.
   *
   * @tparam T The command group type
   * @param cgf The command group functor
   * @param fallbackQ The fallback queue to use in case of error.
   */
  template <typename T>
  inline event submit(T cgf, const cl::sycl::queue& fallbackQ) {
    auto cg = detail::command_group(this->get_impl());
    return cg.submit_handler(
        cgf, fallbackQ.get_impl(),
        detail::get_handler_tag(cgf, detail::select_handler_overload{}));
  }

#if SYCL_LANGUAGE_VERSION >= 202001

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a single_task command.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT>
  inline event single_task(const functorT& functor) {
    return this->submit(
        [&](handler& cgh) { cgh.single_task<nameT, functorT>(functor); });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a single_task command with a single event pre-requisite.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT>
  inline event single_task(event dependency, const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependency);
      cgh.single_task<nameT, functorT>(functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a single_task command with multiple event pre-requisites.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT>
  inline event single_task(const std::vector<event>& dependencies,
                           const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.single_task<nameT, functorT>(functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with a global range parameter.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.parallel_for<nameT, functorT>(globalRange, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with a global range parameter and a
   * single event pre-requisite.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange, event dependency,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<nameT, functorT>(globalRange, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with a global range paramerter and
   * multiple event pre-requisites.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange,
                            const std::vector<event>& dependencies,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<nameT, functorT>(globalRange, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with global range and global offset
   * parameters.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param globalOffset The global offset that the index of each invocation of
   * the kernel function object should be offset by.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange,
                            id<dimensions> globalOffset,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.parallel_for<nameT, functorT>(globalRange, globalOffset, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with global range and global offset
   * parameters and a single event pre-requisite.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param globalOffset The global offset that the index of each invocation of
   * the kernel function object should be offset by.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange,
                            id<dimensions> globalOffset, event dependency,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<nameT, functorT>(globalRange, globalOffset, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with global range and global offset
   * paramerters and multiple event pre-requisites.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param globalRange The global range that the kernel function object should
   * be invoked over.
   * @param globalOffset The global offset that the index of each invocation of
   * the kernel function object should be offset by.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(range<dimensions> globalRange,
                            id<dimensions> globalOffset,
                            const std::vector<event>& dependencies,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<nameT, functorT>(globalRange, globalOffset, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with an nd_range parameter.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param ndRange The nd range that the kernel function object should be
   * invoked over.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(nd_range<dimensions> ndRange,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.parallel_for<nameT, functorT>(ndRange, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command with an nd_range parameter and a
   * single event pre-requisite.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param ndRange The nd range that the kernel function object should be
   * invoked over.
   * @param dependency The event object which must be complete before the
   * command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(nd_range<dimensions> ndRange, event dependency,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependency);
      cgh.parallel_for<nameT, functorT>(ndRange, functor);
    });
  }

  /**
   * @brief Shortcut member function which submits a command group to the queue
   * which enqueues a parallel_for command withan nd_range paramerter and
   * multiple event pre-requisites.
   * @tparam nameT The type used to name the kernel function.
   * @tparam functorT The type of the kernel function callable object.
   * @param ndRange The nd range that the kernel function object should be
   * invoked over.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @param functor The kernel function callable object.
   * @return An event object which can be used to synchronize with the enqueued
   * command.
   */
  template <typename nameT = std::nullptr_t, typename functorT, int dimensions>
  inline event parallel_for(nd_range<dimensions> ndRange,
                            const std::vector<event>& dependencies,
                            const functorT& functor) {
    return this->submit([&](handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.parallel_for<nameT, functorT>(ndRange, functor);
    });
  }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

 protected:
  /// @cond COMPUTECPP_DEV

  /**
   Assigns the cl::sycl::context object provided and assigns the
   cl::sycl::enum_execution_mode value
   from the cl::sycl::context object provided. Constructs the OpenCL
   cl_command_queue object using the
   cl::sycl::context object.
   @param syclContext a pointer to a valid context object.
   */
  explicit queue(dcontext_shptr syclContext);

  /**
   * @brief Fills the memory pointed by @p ptr.
   * @param ptr Pointer object to fill.
   * @param patternData Pointer to the memory that contains the pattern to use
   * when filling @p ptr.
   * @param patternSize The size in bytes of the pattern.
   * @param size The number of bytes of @p ptr to fill with @p patternData.
   * @return A runtime event that corresponds to this operation.
   */
  event fill(void* ptr, const void* pattern, size_t patternSize, size_t size);

  /**
   * @brief Returns true if the queue was constructed with the
   * property::queue::in_order property.
   */
  bool is_in_order_impl() const;

  /**
   * @brief Fills the memory pointed by @p ptr.
   * @param ptr Pointer object to fill.
   * @param patternData Pointer to the memory that contains the pattern to use
   * when filling @p ptr.
   * @param patternSize The size in bytes of the pattern.
   * @param size The number of bytes of @p ptr to fill with @p patternData.
   * @param dependencies A list of event objects which must be complete before
   * the command can be executed.
   * @return A runtime event that corresponds to this operation.
   */
  event fill(void* ptr, const void* pattern, size_t patternSize, size_t size,
             const std::vector<event>& dependencies);

  /**
   * Pointer to the class implementation.
   */
  dqueue_shptr m_impl;

  /** @brief Obtain the properties
   */
  property_list get_properties() const;

 private:
  /** Returns the SYCL backend
   * @return Backend associated with the queue
   */
  backend get_backend_impl() const;

  /// COMPUTECPP_DEV @endcond
};

#if SYCL_LANGUAGE_VERSION >= 202001
/** Property trait specializations
 */
template <>
struct is_property<property::queue::enable_profiling> : public std::true_type {
};

template <>
struct is_property<property::queue::in_order_impl> : public std::true_type {};

template <>
struct is_property_of<property::queue::enable_profiling, queue>
    : public std::true_type {};

template <>
struct is_property_of<property::queue::in_order_impl, queue>
    : public std::true_type {};

#endif  // SYCL_LANGUAGE_VERSION >= 202001

}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the buffer class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <>
struct hash<cl::sycl::queue> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::queue& object) const {
    hash<cl::sycl::dqueue_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_QUEUE_H_

////////////////////////////////////////////////////////////////////////////////
