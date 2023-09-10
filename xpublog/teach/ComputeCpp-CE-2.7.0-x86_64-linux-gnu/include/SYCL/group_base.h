/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file group_base.h

  @brief Internal file for the @ref cl::sycl::group class
*/
#ifndef RUNTIME_INCLUDE_SYCL_GROUP_BASE_H_
#define RUNTIME_INCLUDE_SYCL_GROUP_BASE_H_

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/builtins/device_builtins.h"
#endif  // __SYCL_DEVICE_ONLY__

#include "SYCL/common.h"
#include "SYCL/id.h"
#include "SYCL/index_array.h"
#include "SYCL/range.h"

/* @cond COMPUTECPP_DEV */
namespace cl {
namespace sycl {
namespace detail {

class group_base {
  /* Currently all internal methods are called using the friend keyword. */
  friend class item_base;

 public:
  /** Default constructor.
   */
  group_base();

  /** \brief Constructor: Assigns the local range, global range, number of
   * groups and group id provided, then calculates the linear local range,
   * linear global range and the number of work groups.
   * @tparam dimensions Number of dimensions for the passed in ids and ranges
   * @param groupID the cl::sycl::id object specifying the work group id.
   * @param groupRange Number of work groups.
   * @param globalRange the cl::sycl::range object specifying the global range.
   * @param localRange the cl::sycl::range object specifying the local range.
   */
  template <int dimensions>
  group_base(id<dimensions> groupID, range<dimensions> groupRange,
             range<dimensions> globalRange, range<dimensions> localRange);

  template <int dimensions>
  group_base(const group<dimensions>& g)
      : m_globalRange(g.m_globalRange),
        m_localRange(g.m_localRange),
        m_groupRange(g.m_groupRange),
        m_groupID(g.m_groupID),
        m_linearGroupID(g.m_linearGroupID) {}

  /** \brief Get Group ID
   * @param the dimension of the nd_range we need the group id for
   * @return the group id for that dimension
   * @deprecated Use get_id instead
   */
  COMPUTECPP_DEPRECATED_API("get(int) was deprecated in SYCL 1.2.1")
  size_t get(int dimension) const { return this->get_id(dimension); }

  /** \brief Get Group ID
   * @param the dimension of the nd_range we need the group id for
   * @return the group id for that dimension
   */
  size_t get_id(int dimension) const { return m_groupID[dimension]; }

  /**! \brief Returns the global range in a specified dimension.
   * @param dimension the dimension of the global range to be returned.
   * @return the value of the global range in the specified dimension.
   */
  size_t get_global_range(int dimension) const {
    return m_globalRange[dimension];
  }

  /**! \brief Returns the local range in a specified dimension.
   * @param dimension the dimension of the local range to be returned.
   * @return the value of the local range in the specified dimension.
   */
  size_t get_local_range(int dimension) const {
    return m_localRange[dimension];
  }

  /**! \brief Returns the group range in a specified dimension.
   * @param dimension the dimension of the group range to be returned.
   * @return the value of the group range in the specified dimension.
   */
  size_t get_group_range(int dimension) const {
    return m_groupRange[dimension];
  }

  /** \brief Returns the linearized group id
   * @return the linearized group id
   */
  COMPUTECPP_DEPRECATED_API(
      "SYCL 1.2.1 revision 3 replaces group::get_linear with "
      "group::get_linear_id.")
  size_t get_linear() const { return this->get_linear_id(); }

  /** \brief Returns the linearized group id
   * @return the linearized group id
   */
  size_t get_linear_id() const { return m_linearGroupID; }

  /// @cond COMPUTECPP_DEV

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const group_base& rhs) const {
    return m_globalRange.is_equal<dimensions>(rhs.m_globalRange) &&
           m_localRange.is_equal<dimensions>(rhs.m_localRange) &&
           m_groupRange.is_equal<dimensions>(rhs.m_groupRange) &&
           m_groupID.is_equal<dimensions>(rhs.m_groupID) &&
           (m_linearGroupID == rhs.m_linearGroupID);
  }

  /// COMPUTECPP_DEV @endcond

  /** @brief mem_fence operation
   *
   *        Executes a work-group memory fence with memory ordering
   *        on the local or global fence space (or both).
   * @tparam accessMode Specifies whether all load (access::mode::read),
   *         store (access::mode::write) or both load and store memory accesses
   *         (access::mode::read_write) in the specified address space issued
   *         before the mem-fence should complete before those issued
   *         after the mem-fence.
   * @param accessSpace Address space where the memory fence will be enforced
   */
  template <access::mode accessMode = access::mode::read_write,
            COMPUTECPP_ENABLE_IF(access::mode,
                                 ((accessMode == access::mode::read_write) ||
                                  (accessMode == access::mode::read) ||
                                  (accessMode == access::mode::write)))>
  void mem_fence(access::fence_space accessSpace =
                     access::fence_space::global_and_local) const;

 protected:
  detail::index_array m_globalRange;  // global range struct
  detail::index_array m_localRange;   // local range struct
  detail::index_array m_groupRange;   // group range struct
  detail::index_array m_groupID;      // group id struct
  size_t m_linearGroupID;             // linear computation of group id
};
/********************class group_base ********************/

inline group_base::group_base()
    : m_globalRange(1, 1, 1),
      m_localRange(1, 1, 1),
      m_groupRange(1, 1, 1),
      m_groupID(0, 0, 0),
      m_linearGroupID(0) {}

template <int dimensions>
group_base::group_base(id<dimensions> groupID, range<dimensions> groupRange,
                       range<dimensions> globalRange,
                       range<dimensions> localRange)
    : m_globalRange(globalRange),
      m_localRange(localRange),
      m_groupRange(groupRange),
      m_groupID(groupID),
      m_linearGroupID{
          detail::construct_linear_row_major_index(groupID, groupRange)} {}

#ifdef __SYCL_DEVICE_ONLY__
template <>
inline void group_base::mem_fence<access::mode::read_write>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
template <>
inline void group_base::mem_fence<access::mode::read>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::read_mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
template <>
inline void group_base::mem_fence<access::mode::write>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::write_mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
#else
template <>
inline void group_base::mem_fence<access::mode::read_write>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::read_write);
}
template <>
inline void group_base::mem_fence<access::mode::read>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::read);
}
template <>
inline void group_base::mem_fence<access::mode::write>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::write);
}
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail
}  // namespace sycl
}  // namespace cl

/** COMPUTECPP_DEV @endcond */

#endif  // RUNTIME_INCLUDE_SYCL_GROUP_BASE_H_
