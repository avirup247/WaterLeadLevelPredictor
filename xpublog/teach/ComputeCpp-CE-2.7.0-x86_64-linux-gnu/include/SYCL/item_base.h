/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file item_base.h

  @brief This file implements the internal base classes for \ref cl::sycl::item
  and \ref cl::sycl::nd_item.
*/

#ifndef RUNTIME_INCLUDE_SYCL_ITEM_BASE_H_
#define RUNTIME_INCLUDE_SYCL_ITEM_BASE_H_

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/builtins/device_builtins.h"
#endif  // __SYCL_DEVICE_ONLY__

#include "SYCL/common.h"
#include "SYCL/index_array.h"
#include "SYCL/vec_types_defines.h"

namespace cl {
namespace sycl {
namespace detail {

class nd_item_base;
class workgroup_barrier;

/** \brief It is the non-templated base class of item which contains all the
 * operators
 * and methods of an item struct.
 */
class item_base {
 public:
  /** @internal
   * Default constructor.
   */
  item_base() : m_id(0, 0, 0), m_range(1, 1, 1), m_linearID(0) {}

  /** @internal
   * Default copy constructor.
   */
  item_base(const item_base& rhs) = default;

  /** @internal
   * Assigns the id and range provided, then calculates the linear ids and
   * ranges.
   * @param id the detail::index_array object specifying the id.
   * @param range the detail::index_array object specifying the range in which
   * the item_base is being instantiated.
   */
  item_base(detail::index_array id, detail::index_array range);

  /** @internal
   * Assigns the id, range and offset provided, then calculates the linear ids
   * and ranges.
   * @param id the detail::index_array object specifying the id.
   * @param range the detail::index_array object specifying the range in which
   * the item_base is being instantiated.
   * @param offset the detail::index_array object specifying the instantiated
   * offset.
   */
  item_base(detail::index_array id, detail::index_array range,
            detail::index_array offset);

  /** \brief Returns the id for a specific dimension.
   * @param dimension of the id
   * @return the id for the specified dimension.
   * @deprecated Use get_id instead
   */
  COMPUTECPP_DEPRECATED_API("get(int) was deprecated in SYCL 1.2.1")
  size_t get(int dimension) const { return this->get_id(dimension); }

  /** \brief Returns the id for a specific dimension.
   * @param dimension of the id
   * @return the id for the specified dimension.
   */
  size_t get_id(int dimension) const { return this->m_id[dimension]; }

  /** \brief Returns the id for a specific dimension.
   * @param dimension of the id
   * @return the id for the specified dimension.
   */
  size_t operator[](int dimension) const { return this->get_id(dimension); }

  /** \brief Calculates the linear local id.
   * @return linear local id
   */
  size_t get_linear_id() const { return this->m_linearID; }

  /** @internal
   * \brief Get the range of the associated with this item.
   * @return the values of the range for all dimensions.
   */
  detail::index_array get_range() const { return this->m_range; }

  /** @internal
   * Return the current offset
   */
  detail::index_array get_offset() const { return this->m_offset; }

  /// @cond COMPUTECPP_DEV

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const item_base& rhs) const {
    return m_id.is_equal<dimensions>(rhs.m_id) &&
           m_range.is_equal<dimensions>(rhs.m_range) &&
           m_offset.is_equal<dimensions>(rhs.m_offset) &&
           (m_linearID == rhs.m_linearID);
  }

  /// COMPUTECPP_DEV @endcond

 protected:
  detail::index_array m_id;
  detail::index_array m_range;
  detail::index_array m_offset;
  size_t m_linearID;
};

/** \brief It is the non-templated base class of nd_item which contains all the
 * operators and methods of an nd_item struct.
 * In comparison to item class, nd_item class contains all the low-level OpenCL
 * functionality provided by SYCL.
 * This class offers the ability to the user to use mem_fence synchronization
 * using the barrier method.
 */
class nd_item_base {
 public:
  /** @internal
   * Default constructor.
   * Assigns default values for all fields.
   */
  nd_item_base() = default;

  /** @internal
   * Default constructor.
   * Assigns default values for all fields.
   */
  nd_item_base(const nd_item_base& rhs) = default;

  /** @internal
   * Constructor.
   * Assigns the local size, global size, local id and global id provided, then
   * calculates the linear local and global ids and ranges.
   * The pointer to the barrier is set to null by default.
   * @param localID the detail::index_array object specifying the local id.
   * @param globalID the detail::index_array object specifying the global id.
   * @param localRange the detail::index_array object specifying the local
   * size.
   * @param globalRange the detail::index_array object specifying the global
   * size.
   * @param globalOffset the detail::index_array object indicating the global
   * offset
   * @param groupId the detail::index_array object inidicating the groupID
   */
#ifdef __SYCL_DEVICE_ONLY__
  nd_item_base(detail::index_array localID, detail::index_array globalID,
               detail::index_array localRange, detail::index_array globalRange,
               detail::index_array globalOffset, detail::index_array groupID,
               detail::index_array groupRange)
      : m_globalItem(globalID, globalRange, globalOffset),
        m_localItem(localID, localRange),
        m_groupRange(groupRange),
        m_groupId(groupID) {}
#else
  nd_item_base(detail::index_array localID, detail::index_array globalID,
               detail::index_array localRange, detail::index_array globalRange,
               detail::index_array globalOffset, detail::index_array groupID,
               detail::index_array /*groupRange*/)
      : m_globalItem(globalID, globalRange, globalOffset),
        m_localItem(localID, localRange),
        m_groupRange{[this]() {
          auto global = get_global_item().get_range();
          auto local = get_local_item().get_range();
          return detail::index_array{global / local};
        }()},
        m_groupId(groupID) {}
#endif  // __SYCL_DEVICE_ONLY__
  /** Barrier operation.
   * @param access::fence_space enum class object that specifies the
   * memory fence associated with the barrier.
   */
  void barrier(
      access::fence_space flag = access::fence_space::global_and_local) const;

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

  /** \brief Returns the global id for a specific dimension.
   * @param dimension of the nd_range provided
   * @return the global id for the specified dimension.
   */
  size_t get_global_id(int dimension) const {
    return this->m_globalItem.get_id(dimension);
  }

  /** \brief Returns the local id for a specific dimension.
   * @param dimension of the nd_range provided
   * @return the local id for the specified dimension.
   */
  size_t get_local_id(int dimension) const {
    return this->m_localItem.get_id(dimension);
  }

  /** \brief Get the global range in a specified dimension of the range
   * @param dimension the dimension of the global range to be returned.
   * @return the value of the global range in the specified dimension.
   */
  size_t get_global_range(int dimension) const {
    return this->m_globalItem.get_range()[dimension];
  }

  /** Returns the local range in a specified dimension.
   * @param dimension the dimension of the local range to be returned.
   * @return the value of the local range in the specified dimension.
   */
  size_t get_local_range(int dimension) const {
    return this->m_localItem.get_range()[dimension];
  }

  /** Returns the linearized global id
   */
  size_t get_global_linear_id() const {
    return this->m_globalItem.get_linear_id();
  }

  /** Returns the linearized global id
   */
  size_t get_local_linear_id() const {
    return this->m_localItem.get_linear_id();
  }

  /** Returns the current group id in a given dimension.
   * @param dimension the dimension of the id to be returned.
   * @return the value of the group range in the specified dimension.
   */
  size_t get_group(int dim) const { return m_groupId[dim]; }

  /// @cond COMPUTECPP_DEV

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const nd_item_base& rhs) const {
    return m_globalItem.is_equal<dimensions>(rhs.m_globalItem) &&
           m_localItem.is_equal<dimensions>(rhs.m_localItem) &&
           m_groupId.is_equal<dimensions>(rhs.m_groupId);
  }

  /// COMPUTECPP_DEV @endcond

 protected:
  /** @brief Retrieves the global item
   * @return Global item associated with this nd_item
   */
  detail::item_base get_global_item() const noexcept { return m_globalItem; }

  /** @brief Retrieves the local item
   * @return Local item associated with this nd_item
   */
  detail::item_base get_local_item() const noexcept { return m_localItem; }

  /** @brief Retrieves the group range
   * @return Group range associated with this nd_item
   */
  detail::index_array get_group_range() const noexcept { return m_groupRange; }

  /** @brief Retrieves the group ID
   * @return ID of the group associated with this nd_item
   */
  detail::index_array get_group_id() const noexcept { return m_groupId; }

 private:
  // Expose members to item_base, don't create a public interface
  friend class detail::item_base;

  detail::item_base m_globalItem;
  detail::item_base m_localItem;
  detail::index_array m_groupRange;
  detail::index_array m_groupId;
};

/** @brief Base class of the h_item class.
 *
 *        Stores the global item, the logical local item,
 *        and the physical local range.
 *        h_item has basically the same data layout as nd_item,
 *        that's why this class inherits from nd_item_base,
 *        but it performs protected inheritance because there are some nd_item
 *        member functions that should not be public in h_item.
 */
class h_item_base {
 public:
  /** @brief Initializes all IDs to zeros and ranges to ones
   */
  h_item_base() = default;

  /** @brief Constructor from a logical local item, physical local item and
   *  global item
   * @param logicalLocalItem Work-item logical local item
   * @param physicalLocalItem Work-item physical local item
   * @param globalItem Work-item global item
   */
  h_item_base(detail::item_base logicalLocalItem,
              detail::item_base physicalLocalItem, detail::item_base globalItem)
      : m_localItem(logicalLocalItem),
        m_localPhysicalItem(physicalLocalItem),
        m_globalItem(globalItem) {}

 protected:
  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const h_item_base& rhs) const {
    return m_localItem.is_equal<dimensions>(rhs.m_localItem) &&
           m_localPhysicalItem.is_equal<dimensions>(rhs.m_localPhysicalItem) &&
           m_globalItem.is_equal<dimensions>(rhs.m_globalItem);
  }

  /** @brief Retrieves the constituent global item representing the work-item's
   *        position in the global iteration space
   * @return Item representing the global ID and range
   */
  inline detail::item_base get_global_item_base() const { return m_globalItem; }

  /** @brief Retrieves the constituent logical local item representing the
   *        work-item's position in the local iteration space as provided upon
   *        the invocation of parallel_for_work_item
   * @return Item representing the logical local ID and range
   */
  inline detail::item_base get_logical_local_item_base() const {
    return m_localItem;
  }

  /** @brief Retrieves the constituent physical local item representing the
   *        work-item's position in the local iteration space as provided upon
   *        the invocation of parallel_for_work_group
   * @return Item representing the physical local ID and range
   */
  inline detail::item_base get_physical_local_item_base() const {
    return m_localPhysicalItem;
  }

 private:
  // Expose members to item_base, don't create a public interface
  friend class detail::item_base;

  detail::item_base m_localItem;
  detail::item_base m_localPhysicalItem;
  detail::item_base m_globalItem;
};

/*********************
 * class item_base *
 *********************/

inline item_base::item_base(detail::index_array id, detail::index_array range,
                            detail::index_array offset)
    : m_id(id),
      m_range(range),
      m_offset(offset),
      m_linearID{detail::construct_linear_row_major_index(m_id, m_range)} {}

inline item_base::item_base(detail::index_array id, detail::index_array range)
    : item_base(id, range, detail::index_array(0, 0, 0)) {}

/*********************
 * class nd_item_base *
 *********************/

COMPUTECPP_DEPRECATED_BY_SYCL_VER(202001, "Use group_barrier(group) instead.")
inline void nd_item_base::barrier(access::fence_space flag) const {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::barrier(::cl::sycl::detail::get_cl_mem_fence_flag(flag));
#else
  ::cl::sycl::detail::host_barrier(*this);
  (void)flag;
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
template <>
inline void nd_item_base::mem_fence<access::mode::read_write>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
template <>
inline void nd_item_base::mem_fence<access::mode::read>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::read_mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
template <>
inline void nd_item_base::mem_fence<access::mode::write>(
    access::fence_space accessSpace) const {
  ::cl::sycl::detail::write_mem_fence(
      ::cl::sycl::detail::get_cl_mem_fence_flag(accessSpace));
}
#else
template <>
inline void nd_item_base::mem_fence<access::mode::read_write>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::read_write);
}
template <>
inline void nd_item_base::mem_fence<access::mode::read>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::read);
}
template <>
inline void nd_item_base::mem_fence<access::mode::write>(
    access::fence_space) const {
  cl::sycl::detail::host_mem_fence(access::mode::write);
}
#endif  // __SYCL_DEVICE_ONLY__

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ITEM_BASE_H_
