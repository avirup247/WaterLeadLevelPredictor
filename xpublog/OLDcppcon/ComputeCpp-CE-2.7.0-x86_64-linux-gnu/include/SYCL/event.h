/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 * @file event.h
 * @brief This file contains the @ref cl::sycl::event class interface.
 */
#ifndef RUNTIME_INCLUDE_SYCL_EVENT_H_
#define RUNTIME_INCLUDE_SYCL_EVENT_H_

#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"
#include "SYCL/predefines.h"

#include <iosfwd>
#include <memory>
#include <system_error>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class context;
class event;

// Info classes
namespace info {
/** @brief Event class descriptor.
 * SYCL 1.2.1 specification Appendix A.7
 */
enum class event : int {
  command_execution_status,
  reference_count
};  // enum class event

enum class event_command_status : int { submitted, running, complete };

enum class event_profiling : int { command_submit, command_start, command_end };
}  // namespace info

/** @cond COMPUTECPP_DEV */

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(event, cl_event_info, cl_event)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(event, reference_count,
                                      CL_EVENT_REFERENCE_COUNT, cl_uint,
                                      cl_uint)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(event, command_execution_status,
                                      CL_EVENT_COMMAND_EXECUTION_STATUS,
                                      info::event_command_status, cl_int)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(event, reference_count, 0)

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(event_profiling, cl_profiling_info,
                                    cl_event)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(event_profiling, command_submit,
                                      CL_PROFILING_COMMAND_SUBMIT, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(event_profiling, command_start,
                                      CL_PROFILING_COMMAND_START, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(event_profiling, command_end,
                                      CL_PROFILING_COMMAND_END, cl_ulong,
                                      cl_ulong)

// Dummy host definitions
COMPUTECPP_DEFINE_SYCL_INFO_HOST(event_profiling, command_start, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(event_profiling, command_submit, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(event_profiling, command_end, 0)

namespace detail {

template <>
struct opencl_backend_traits<sycl::event> {
 public:
  using input_type = std::vector<cl_event>;
  using return_type = input_type;
};

}  // namespace detail

/** COMPUTECPP_DEV @endcond */

/** @brief Abstraction of a cl_event object.
 * See Section 4.4.6 of the SYCL Specification 1.2.1
 */
class COMPUTECPP_EXPORT event {
 public:
  /** @brief Constructs a ready SYCL event.
   *
   * If the constructed SYCL event is waited on, it will complete immediately.
   */
  event();

  /** @brief Creates a SYCL event from an OpenCL event
   * @param The OpenCL event we are constructing the SYCL object from
   * @deprecated Need to provide a context as well
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703,
      "Use the OpenCL interop constructor which takes a SYCL context instead.")
  event(cl_event);

  /** @brief Creates a SYCL event from an OpenCL event
   * @param The OpenCL event we are constructing the SYCL object from
   * @param Context associated with the OpenCL event
   */
  event(cl_event clEvent, const context& syclContext);

  /// @cond COMPUTECPP_DEV

  /** @brief Internal constructor to create events from
   * internal implementation objects
   */
  event(const devent_shptr& impl);

  /// COMPUTECPP_DEV @endcond

  /** @brief Default copy constructor.
   */
  event(const event& rhs) = default;

  /** @brief Default copy assignment.
   */
  event& operator=(const event& rhs) = default;

  /** @brief Default move constructor.
   * @param rhs will have its contents moved. after the operation rhs will be
   * invalid.
   */
  event(event&& rhs) = default;

  /** @brief Default move assignment operator
   * @param rhs will have its contents moved. after the operation rhs will be
   * invalid.
   */
  event& operator=(event&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const event& lhs, const event& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const event& lhs, const event& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Returns the underlying cl_event
   * @return The associated OpenCL event
   */
  cl_event get() const;

  /** @brief Returns the list of events that depend on the current one
   * @return A vector of SYCL events that this event depends on.
   */
  vector_class<event> get_wait_list();

  /** @brief Waits for the event to complete.
   */
  void wait();

  /** @brief Waits for the event to complete.
   * Throws any exception that can be associated with the execution of the
   * event
   * @throw cl::sycl::exception
   */
  void wait_and_throw();

  /** @brief Waits for all the events in the list
   * @param eventList List of events to wait
   */
  static void wait(const vector_class<event>& eventList);

  /** @brief Waits for all the events in the list. Exceptions may be thrown.
   * @param eventList List of events to wait
   */
  static void wait_and_throw(const vector_class<event>& eventList);

  /** @brief Returns a pointer to the implementation of the event
   * @return A pointer to an event implementation
   */
  devent_shptr get_impl() const;

  /** @brief Returns true if the event is a host event
   * @return True if the event is a host event, false otherwise
   */
  bool is_host() const;

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Returns the SYCL backend
   * @return Backend associated with the event
   */
  inline backend get_backend() const { return this->get_backend_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /** @brief Gets the OpenCL event information from the SYCL event
   * @tparam info::event param  The OpenCL parameter requested
   * @return opencl_event_info<param> The equivalent OpenCL type
   */
  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info()
      const {
    cl_event e = this->get_no_retain();
    return get_sycl_info<info::event,
                         typename opencl_event_info<param>::sycl_type,
                         typename opencl_event_info<param>::cl_type,
                         opencl_event_info<param>::cl_param>(e, is_host());
  }

  /** @brief Queries the SYCL event for profiling information
   * @tparam param The profiling parameter requested
   * @return An implementation defined 64-bit value describing the time in
   *         nanoseconds when the requested profiling event occurred
   * @throw invalid_object_error If the queue associated with the event was not
   *        constructed with the property::queue::enable_profiling property
   */
  template <info::event_profiling param>
  COMPUTECPP_EXPORT
      typename info::param_traits<info::event_profiling, param>::return_type
      get_profiling_info() const;

 protected:
  /** @brief Retrieves the OpenCL event without retaining it
   * @param OpenCL event associated with this event object
   */
  cl_event get_no_retain() const;

 private:
  /** Returns the SYCL backend
   * @return Backend associated with the event
   */
  backend get_backend_impl() const;

  /** @brief A managed pointer to the event implementation.
   */
  devent_shptr m_impl;
};

COMPUTECPP_GET_INFO_SPECIALIZATION_DECL(event, command_execution_status)

}  // namespace sycl
}  // namespace cl

namespace std {
/**
 * @brief provides a specialization for std::hash for the buffer class. An
 * std::hash<std::shared_ptr<...>> object is created and its function call
 * operator is used to hash the contents of the shared_ptr. The returned hash is
 * actually the result of (size_t) object.get_impl().get()
 */
template <>
struct hash<cl::sycl::event> {
 public:
  /**
   * @brief enables calling an std::hash object as a function with the object to
   * be hashed as a parameter
   * @param object the object to be hashed
   * @tparam std the std namespace where this specialization resides
   */
  size_t operator()(const cl::sycl::event& object) const {
    hash<cl::sycl::devent_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_EVENT_H_

////////////////////////////////////////////////////////////////////////////////
