/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file nd_range_base.h

  @brief This file defined the base definition of the \ref cl::sycl::nd_range.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ND_RANGE_BASE_H_
#define RUNTIME_INCLUDE_SYCL_ND_RANGE_BASE_H_

#include "SYCL/common.h"
#include "SYCL/index_array.h"

namespace cl {
namespace sycl {
namespace detail {

/** @internal
  @brief The cl::sycl::nd_range object is a container for that is constructed to
  specify global and local range and an optional offset to used when enqueuing
  a kernel. The cl::sycl::nd_range object can return the global and local
  sizes.
*/
class nd_range_base {
 public:
  /** @internal
   * @brief Default constructor.
   */
  nd_range_base();
  /** @internal
   * \brief Constructor: It assigns the global range and offset and calculates
   * the linear global range.
   * @param globalRange the cl::sycl::range object specifying the global range
   * size.
   * @param globalOffset the cl::sycl::id object specifying the global offset
   * size.
   */
  nd_range_base(detail::index_array globalRange,
                detail::index_array globalOffset);

  /** @internal
   * \brief Constructor. Set the global and local ranges and global offset
   * provided, then calculates the linear global and local range.
   * @param globalRange the cl::sycl::range object specifying the global range
   * size.
   * @param localRange the cl::sycl::range object specifying the local range
   * @param globalOffset the cl::sycl::id object specifying the global offset
   * size.
   */
  nd_range_base(detail::index_array globalRange, detail::index_array localRange,
                detail::index_array globalOffset);

#ifndef __APPLE__

  /** @internal
   * \brief Constructor: Assigns the global and local ranges provided and sets a
   * boolean which specifies whether or not the local range was provided
   * depending on the values provided in the initializer list.
   * Also generates errors if the values provided in the initializer list are
   * invalid. It then calculates the linear global and local range values.
   * Note: this constructor is not supported on Mac OSX.
   * @param xss an initializer list of two sub initializer lists that together
   * specify the global local range respectively.
   */
  nd_range_base(  // NOLINT This constructor is used for conversions
      std::initializer_list<std::initializer_list<size_t>> xss);

#endif

  /** \brief Returns the linear global range size.
   * @return the linear global range size.
   */
  size_t get_global_linear_range() const;

  /** \brief Returns the global range in a specified dimension.
   * @param dimension the dimension of the global range to be returned.
   * @return the value of the global range in the specified dimension.
   */
  size_t get_global_range(unsigned int dimension) const;

  /** @internal
   * \brief It returns the global offset of the nd_range
   * @return The global offset of the nd_range
   */
  detail::index_array get_offset() const;

  /** @internal
   * \brief It returns the global range of the nd_range
   * @return The global range of the nd_range
   */
  detail::index_array get_global_range() const;

  /** @internal
   * \brief It returns the local range of the nd_range
   * @return The local range of the nd_range
   */
  detail::index_array get_local_range() const;

  /** \brief Returns the linear local range size.
   * @return the linear local range size.
   */
  size_t get_local_linear_range() const;

  /** @brief Returns the local range in a specified dimension.
   * @param dimension is the dimension of the local range to be returned.
   * @return the value of the local range in the specified dimension.
   */
  size_t get_local_range(unsigned int dimension) const;

  /** @internal
   * \brief Returns whether or not the local range was provided.
   * @return a boolean specifying whether or not the local range was provided on
   * construction.
   */
  bool is_local_size_specified() const;

  /** \brief Returns whether or not all the global range elements are divisible
   * by the correspondent local size
   * @return True if divisible, false if not.
   */
  bool is_divisible() const;

  /** @internal
   * \brief Returns the group range
   * \return The group range computed from the local and global range
   */
  detail::index_array get_group_range() const;

  /** @brief Helper function for calling operator==() in the parent
   * @tparam dimensions Number of dimensions of the parent object
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  template <int dimensions>
  inline bool is_equal(const nd_range_base& rhs) const {
    return m_globalRange.is_equal<dimensions>(rhs.m_globalRange) &&
           m_localRange.is_equal<dimensions>(rhs.m_localRange) &&
           m_globalOffset.is_equal<dimensions>(rhs.m_globalOffset) &&
           (m_LinearGlobalRange == rhs.m_LinearGlobalRange) &&
           (m_LinearLocalRange == rhs.m_LinearLocalRange) &&
           (m_localRangeSpecified == rhs.m_localRangeSpecified);
  }

 private:
  detail::index_array m_globalRange;
  detail::index_array m_localRange;
  detail::index_array m_globalOffset;
  size_t m_LinearGlobalRange;
  size_t m_LinearLocalRange;
  bool m_localRangeSpecified;
};

/**********************
 * class nd_range_base *
 **********************/

inline nd_range_base::nd_range_base(detail::index_array globalRange,
                                    detail::index_array globalOffset)
    : m_globalRange(globalRange),
      m_localRange(1, 1, 1),
      m_globalOffset(globalOffset),
      m_localRangeSpecified(false) {
  m_LinearLocalRange = 1;
  m_LinearGlobalRange = m_globalRange[0] * m_globalRange[1] * m_globalRange[2];
}

inline nd_range_base::nd_range_base()
    : m_globalRange(1, 1, 1),
      m_localRange(1, 1, 1),
      m_globalOffset(0, 0, 0),
      m_localRangeSpecified(false) {
  m_LinearLocalRange = 1;
  m_LinearGlobalRange = m_globalRange[0] * m_globalRange[1] * m_globalRange[2];
}

inline nd_range_base::nd_range_base(detail::index_array globalRange,
                                    detail::index_array localRange,
                                    detail::index_array globalOffset)
    : m_globalRange(globalRange),
      m_localRange(localRange),
      m_globalOffset(globalOffset),
      m_localRangeSpecified(true) {
  m_LinearLocalRange = m_localRange[0] * m_localRange[1] * m_localRange[2];
  m_LinearGlobalRange = m_globalRange[0] * m_globalRange[1] * m_globalRange[2];
}

inline nd_range_base::nd_range_base(  // NOLINT This constructor is used for
                                      // conversions
    std::initializer_list<std::initializer_list<size_t>> xss)
    : m_globalRange(1, 1, 1),
      m_localRange(1, 1, 1),
      m_globalOffset(0, 0, 0),
      m_localRangeSpecified(false) {
  if ((xss.size() < 1) || (xss.size() > 2)) {
    COMPUTECPP_CL_ERROR_CODE_MSG(
        CL_SUCCESS, detail::cpp_error_code::CREATE_NDRANGE_ERROR, nullptr,
        "Invalid number of work sizes provided in initializer list")
    return;
  }

  int idIndex = 0;
  unsigned int sizeIndex = 0;

  for (std::initializer_list<size_t> const& xs : xss) {
    size_t idVals[3] = {1, 1, 1};

    for (size_t const& val : (*(xss.begin() + sizeIndex))) {
      if (idIndex >= 3) {
        COMPUTECPP_CL_ERROR_CODE_MSG(
            CL_SUCCESS, detail::cpp_error_code::CREATE_NDRANGE_ERROR, nullptr,
            "Invalid number of arguments for ND Range provided in initializer "
            "list")

        break;
      }
      idVals[idIndex++] = val;
    }

    if (sizeIndex == 0)
      m_globalRange = detail::index_array(idVals[0], idVals[1], idVals[2]);

    if (sizeIndex == 1) {
      m_localRange = detail::index_array(idVals[0], idVals[1], idVals[2]);
      m_localRangeSpecified = true;
    }

    idIndex = 0;
    sizeIndex++;

    // XS is only used for counting the number of arguments, but compiler
    // complains it is not used. We introduce a "fake usage" here to
    // avoid the compiler warning.
    (void)xs;
  }

  m_LinearLocalRange = m_localRange[0] * m_localRange[1] * m_localRange[2];
  m_LinearGlobalRange = m_globalRange[0] * m_globalRange[1] * m_globalRange[2];
}

inline size_t nd_range_base::get_local_linear_range() const {
  return m_LinearLocalRange;
}

inline detail::index_array nd_range_base::get_offset() const {
  return m_globalOffset;
}

inline detail::index_array nd_range_base::get_global_range() const {
  return m_globalRange;
}

inline size_t nd_range_base::get_global_range(unsigned int dimension) const {
  return m_globalRange[dimension];
}

inline size_t nd_range_base::get_global_linear_range() const {
  return m_LinearGlobalRange;
}

inline detail::index_array nd_range_base::get_local_range() const {
  return m_localRange;
}

inline size_t nd_range_base::get_local_range(unsigned int dimension) const {
  return m_localRange[dimension];
}

inline bool nd_range_base::is_local_size_specified() const {
  return m_localRangeSpecified;
}

/** Checks if the global range is divisible by the local one
 * @param globalRange Global range
 * @param localRange Local range
 * @param dimensions Number of range dimensions
 * @return True if entire global range equally divisible by local range,
 *         false otherwise
 */
inline bool is_divisible(const index_array& globalRange,
                         const index_array& localRange,
                         const int dimensions = 3) {
  for (int i = 0; i < dimensions; ++i) {
    if ((globalRange[i] % localRange[i]) != 0) {
      return false;
    }
  }
  return true;
}

inline bool nd_range_base::is_divisible() const {
  return detail::is_divisible(m_globalRange, m_localRange);
}

/** \brief Returns the global range
 */
inline detail::index_array nd_range_base::get_group_range() const {
  return m_globalRange / m_localRange;
}

}  // namespace detail
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ND_RANGE_BASE_H_
