/**************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

****************************************************************/

/**
  @file command_group.h
  @brief This files contains the detail command group class for dealing with
         queue submission and the command group handler
 */

#ifndef RUNTIME_INCLUDE_SYCL_COMMAND_GROUP_H_
#define RUNTIME_INCLUDE_SYCL_COMMAND_GROUP_H_

#include "SYCL/base.h"
#include "SYCL/event.h"

#include <memory>
#include <type_traits>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

/// COMPUTECPP_DEV @endcond
class handler;

namespace codeplay {
class host_handler;
class handler;
}  // namespace codeplay
/// @cond COMPUTECPP_DEV

namespace detail {

// The handler tag implementation was provided by Simon Brand
// This allows up to 2 different handlers

/**
  @brief Helper struct that allows overloads for the standard handler
*/
struct standard_handler_tag {};

/**
  @brief Helper struct that allows overloads for the Codeplay host handler
*/
struct codeplay_host_handler_tag {};

/**
  @brief Helper struct that allows overloads for the Codeplay handler
*/
struct codeplay_handler_tag {};

/**
  @brief Helper struct used when selecting a handler

  Inheritance here ensures that if there isn't a function that accepts
  handler_choice<handler_level>, it can still be converted to
  handler_choice<handler_level+1>
  Handlers with a lower handler_choice number are tested first
*/
template <unsigned handler_level>
struct handler_choice : handler_choice<handler_level + 1> {};

/**
  @brief The upper limit of the handler_choice inheritance chain
         a.k.a. max number of handlers supported
 */
template <>
struct handler_choice<3> {};

/**
  @brief This just provides a clearer interface for selecting the proper
  overload
*/
using select_handler_overload = handler_choice<0>;

/**
  @brief Helper function that tries to instantiate a function call using the
         standard handler
*/
template <typename T>
static auto get_handler_tag(T t, handler_choice<0>)
    -> decltype(t(std::declval<handler&>()), standard_handler_tag{}) {
  return {};
}

/**
  @brief Helper function that tries to instantiate a function call using the
         Codeplay host handler
*/
template <typename T>
static auto get_handler_tag(T t, handler_choice<1>)
    -> decltype(t(std::declval<codeplay::host_handler&>()),
                codeplay_host_handler_tag{}) {
  return {};
}

/**
  @brief Helper function that tries to instantiate a function call using the
         Codeplay handler
*/
template <typename T>
static auto get_handler_tag(T t, handler_choice<2>)
    -> decltype(t(std::declval<codeplay::handler&>()), codeplay_handler_tag{}) {
  return {};
}

class COMPUTECPP_EXPORT command_group {
 public:
  command_group(const dqueue_shptr& queueImpl) : m_queue(queueImpl) {}

  /** @brief Creates a handler with the given queue as a fallback queue.
   *
   * @param fallbackQueue
   *
   * @return A pointer to a new instance of a handler.
   */
  unique_ptr_class<handler> create_handler(
      const dqueue_shptr& fallbackQueue) const;

  /** @brief Creates a Codeplay host handler with the given queue as a fallback
   * queue.
   *
   * @param fallbackQueue
   *
   * @return A pointer to a new instance of a Codeplay host handler.
   */
  unique_ptr_class<codeplay::host_handler> create_codeplay_host_handler(
      const dqueue_shptr& fallbackQueue) const;

  /** @brief Creates a Codeplay handler with the given queue
   *        and a fallback queue
   * @param fallbackQueue Queue to use in case of an error
   * @return A pointer to a new instance of a Codeplay handler
   */
  unique_ptr_class<codeplay::handler> create_codeplay_handler(
      const dqueue_shptr& fallbackQueue) const;
  /** @brief Extracts the transaction from the handler
   *
   * @param cgh The handler to work on
   *
   * @return Handler events: Events associated with the handler.
   */
  devent_shptr run_handler(handler& cgh) const;

  /** @brief Extracts the transaction from the handler and destroys the handler
   *
   * @param cgh The handler to finish.
   *
   * @return Handler events: Events associated with the handler.
   */
  devent_shptr finish_handler(handler* cgh) const;

  /** @brief Extracts the transaction from the handler and destroys the handler
   *
   * @param cgh The handler to finish.
   *
   * @return Handler events: Events associated with the handler.
   */
  devent_shptr finish_handler(codeplay::host_handler* cgh) const;

  /** @brief Extracts the transaction from the handler and destroys the handler
   * @param cgh The handler to finish
   * @return Handler events: Events associated with the handler.
   */
  devent_shptr finish_handler(codeplay::handler* cgh) const;

  template <typename T>
  inline cl::sycl::event submit_handler(T cgf,
                                        const dqueue_shptr& fallbackQueue,
                                        detail::standard_handler_tag) {
    auto cgh = create_handler(fallbackQueue);
    cgf(*cgh);
    return cl::sycl::event(finish_handler(cgh.get()));
  }

  template <typename T>
  inline cl::sycl::event submit_handler(T cgf,
                                        const dqueue_shptr& fallbackQueue,
                                        detail::codeplay_host_handler_tag) {
    auto cgh = create_codeplay_host_handler(fallbackQueue);
    cgf(*cgh);
    return cl::sycl::event(finish_handler(cgh.get()));
  }

  /** @brief Creates a Codeplay handler, passes it to the user function,
   *        and finalizes the submission.
   * @tparam FunctorT Type of the user function
   * @param cgf User function
   * @param fallbackQueue Option fallback queue
   * @return Event corresponding to the command group submission
   */
  template <typename FunctorT>
  inline cl::sycl::event submit_handler(FunctorT cgf,
                                        const dqueue_shptr& fallbackQueue,
                                        detail::codeplay_handler_tag) {
    auto cgh = create_codeplay_handler(fallbackQueue);
    cgf(*cgh);
    return cl::sycl::event(finish_handler(cgh.get()));
  }

 private:
  dqueue_shptr m_queue;
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_COMMAND_GROUP_H_

////////////////////////////////////////////////////////////////////////////////
