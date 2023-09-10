/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file stream_args.h

  @brief Internal file used by the @ref cl::sycl::stream class.
*/

#ifndef RUNTIME_INCLUDE_SYCL_STREAM_ARGS_H_
#define RUNTIME_INCLUDE_SYCL_STREAM_ARGS_H_

#include "SYCL/common.h"

namespace cl {
namespace sycl {

/// @cond COMPUTECPP_DEV

/**
@brief Enum class that enumerates the different modes a stream class can be in.
*/
enum class stream_mode {
  standard = 0,
  scientific = 1,
  hex = 2,
  oct = 4,
  showbase = 4,
  showpos = 5,
  dec = 6,
  noshowbase = 7,
  noshowpos = 8,
  fixed = 9,
  hexfloat = 10,
  defaultfloat = 11
};

/// COMPUTECPP_DEV @endcond

namespace detail {

/**
@brief Struct that encapsulates the host arguments for a stream class on the
host.
*/
struct host_stream_container {
  /**
  @brief A shared_ptr to the detail buffer object.
  */
  std::shared_ptr<cl::sycl::storage_mem> m_buffer;

  /**
  @brief The current index into the buffer.
  */
  mutable int m_currentIndex;

  /**
  @brief The maximum statement size.
  */
  int m_maxStatementSize;

  /**
  @brief The stream mode.
  */
  stream_mode m_streamMode;

  /**
  @brief The precision value.
  */
  int m_precision;

  /** @brief Width of a single stream element
   */
  int m_width;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const host_stream_container& lhs,
                                const host_stream_container& rhs) {
    return (lhs.m_buffer == rhs.m_buffer) &&
           (lhs.m_currentIndex == rhs.m_currentIndex) &&
           (lhs.m_maxStatementSize == rhs.m_maxStatementSize) &&
           (lhs.m_streamMode == rhs.m_streamMode) &&
           (lhs.m_precision == rhs.m_precision) && (lhs.m_width == rhs.m_width);
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const host_stream_container& lhs,
                                const host_stream_container& rhs) {
    return !(lhs == rhs);
  }
};

/**
@brief Struct that encapsulates stream meta data on the device. Size of fields
is different depending on device pointer width in order to work around an issue
where kernel arguments have to be exactly 8 bytes on some CPU devices,
sizeof(device_stream_metadata) will always be 8 bytes.
*/
struct device_stream_metadata {
#ifdef COMPUTECPP_ENV_64
  using field_type = int16_t;
  int16_t m_bufferSize;
  int16_t m_maxStatementSize;
  mutable int16_t m_currentIndex;
  int16_t m_streamMode;
#else
  using field_type = int8_t;
  int8_t m_bufferSize;
  int8_t m_maxStatementSize;
  mutable int8_t m_currentIndex;
  int8_t m_streamMode;
#endif

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if all member variables are the same
   */
  friend inline bool operator==(const device_stream_metadata& lhs,
                                const device_stream_metadata& rhs) {
    return ((lhs.m_bufferSize == rhs.m_bufferSize) &&
            (lhs.m_maxStatementSize == rhs.m_maxStatementSize) &&
            (lhs.m_currentIndex == rhs.m_currentIndex) &&
            (lhs.m_streamMode == rhs.m_streamMode));
  }
};

#ifdef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DEVICE_STREAM_MIRROR_CONVERT                                \
  [[computecpp::opencl_mirror_convert(host_stream_container)]]
#else  // !__SYCL_DEVICE_ONLY__
#define COMPUTECPP_DEVICE_STREAM_MIRROR_CONVERT
#endif  // __SYCL_DEVICE_ONLY__

/**
@brief Struct that encapsulates the device arguments for a stream class on the
device.
*/
struct COMPUTECPP_DEVICE_STREAM_MIRROR_CONVERT device_stream_container {
  COMPUTECPP_CL_ASP_GLOBAL char* m_ptr;
  device_stream_metadata m_metadata;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if all member variables are the same
   */
  friend inline bool operator==(const device_stream_container& lhs,
                                const device_stream_container& rhs) {
    return ((lhs.m_ptr == rhs.m_ptr) && (lhs.m_metadata == rhs.m_metadata));
  }

} COMPUTECPP_CONVERT_ATTR_STREAM;

#undef COMPUTECPP_DEVICE_STREAM_MIRROR_CONVERT

}  // namespace detail

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_STREAM_ARGS_H_

////////////////////////////////////////////////////////////////////////////////
