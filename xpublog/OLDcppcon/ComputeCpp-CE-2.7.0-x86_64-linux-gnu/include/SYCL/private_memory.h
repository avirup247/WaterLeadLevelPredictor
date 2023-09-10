/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file private_memory.h

  @brief This file implements the \ref private_memory class for host and device.
 */

#ifndef RUNTIME_INCLUDE_SYCL_PRIVATE_MEMORY_H_
#define RUNTIME_INCLUDE_SYCL_PRIVATE_MEMORY_H_

#include "SYCL/common.h"

namespace cl {
namespace sycl {
namespace detail {

/*******************************************************************************
    private_memory
*******************************************************************************/
#ifdef __SYCL_DEVICE_ONLY__

/// @brief Base implementation of private_memory
template <typename elementT>
class private_memory_base {
 public:
  template <int kDimensions>
  private_memory_base(const group<kDimensions>&) {}

  template <int kDimensions, bool with_offset>
  elementT& get(const item<kDimensions, with_offset>&) {
    return m_privateMemVariable;
  }

 private:
  elementT m_privateMemVariable;
};

#else

/// @brief Base implementation of private_memory
template <typename elementT>
class private_memory_base {
 protected:
  /** Allocate space in private memory based on the group range.
   * \tparam kDimensions The group dimension
   * \param group The group as provided by \ref handler::parallel_for_work_group
   */
  template <int kDimensions>
  private_memory_base(const group<kDimensions>& group)
      : m_privateMemVariable(group.get_global_range().size() /
                             group.get_group_range().size()) {}

  /** Get the element allocated in private memory for the work item.
   * \tparam kDimensions The item dimension
   * \param item The item as provided by \ref parallel_for_work_item
   * \warning This method cannot be used if the local range was redefined when
   * calling parallel_for_work_item.
   */
  template <int kDimensions, bool with_offset>
  elementT& get(const item<kDimensions, with_offset>& index) {
    unsigned offset = index.get_linear_id();
    return m_privateMemVariable[offset];
  }

 private:
  std::vector<elementT> m_privateMemVariable;
};

#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail

/** @brief This class allows private memory allocation inside a \ref
 * handler::parallel_for_work_group. By default, named variables declared in a
 * \ref handler::parallel_for_work_group are allocated in the OpenCL local
 * address space and shared across work items inside a work group. Instances of
 * this class are private to work items, and allow sharing of private data
 * across different \ref parallel_for_work_item calls.
 *
 * Use COMPUTECPP_PRIVATE_MEMORY_ATTR attribute to ensure that the
 * private memory class is allocated in the private address space.
 * By default, a variable declared in a parallel_for_work_group
 * will be allocated in the local address space
 * - this attribute cancels this effect.
 *
 * @tparam elementT Underlying type of the stored data
 * @tparam kDimensions Data dimensions
 */
template <typename elementT, int kDimensions = 1>
class private_memory : public detail::private_memory_base<elementT> {
 private:
  using base_t = detail::private_memory_base<elementT>;

 public:
  /** @brief Allocate private memory based on the group range.
   * @param group The group instance provided by the \ref
   * handler::parallel_for_work_group.
   */
  private_memory(const group<kDimensions>& group) : base_t(group) {}

  /** @brief Return the allocated private memory for the work item.
   * @param index The item instance representing the work-item.
   * @return A reference to the work-item private instance.
   * \warning This method cannot be used if the local range was redefined when
   * calling parallel_for_work_item.
   * @deprecated Use operator()(h_item)
   */
  COMPUTECPP_DEPRECATED_API("operator()(item) deprecated in SYCL 1.2.1, "
                            "use operator()(h_item) instead")
  elementT& operator()(const item<kDimensions>& index) {
    return base_t::get(index);
  }

  /** @brief Return the allocated private memory for the work item.
   * @param index The h_item instance representing the work-item.
   * @return A reference to the work-item private instance.
   * @warning This method cannot be used if the local range was redefined when
   *          calling parallel_for_work_item.
   */
  elementT& operator()(const h_item<kDimensions>& index) {
    return base_t::get(index.get_local());
  }

} COMPUTECPP_PRIVATE_MEMORY_ATTR;

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_PRIVATE_MEMORY_H_

////////////////////////////////////////////////////////////////////////////////
