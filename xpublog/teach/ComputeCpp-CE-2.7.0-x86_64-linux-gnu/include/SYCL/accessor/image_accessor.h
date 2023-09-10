/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file image_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/image.h"

namespace cl {
namespace sycl {

namespace detail {

/** Template trait which provides a boolean value which specifies whether
 * a particular type is a value type for coordinates of a particular
 * dimensionality.
 * @tparam dimensions The required dimensionality for the type.
 * @tparam coordT The type which is checked for being a valid coordinates type.
 */
template <int dimensions, typename coordT>
struct is_coords {
  /** The value result of the type trait @ref is_coords */
  static constexpr auto value =
      (detail::is_same_basic_type<typename coordT::element_type,
                                  cl::sycl::cl_int>::value ||
       detail::is_same_basic_type<typename coordT::element_type,
                                  cl::sycl::cl_float>::value) &&
      ((dimensions == coordT::width) ||
       (dimensions == 3 && coordT::width == 4));
};

/** Specialization of is_coords because for 1 dimension we have a scalar
 *        instead of a vec
 * @tparam coordT The type which is checked for being a valid coordinates type
 */
template <typename coordT>
struct is_coords<1, coordT> {
  /** The value result of the type trait @ref is_coords */
  static constexpr auto value =
      (detail::is_same_basic_type<coordT, cl::sycl::cl_int>::value ||
       detail::is_same_basic_type<coordT, cl::sycl::cl_float>::value);
};

////////////////////////////////////////////////////////////////////////////////
// accessor_image_interface

/** Common interface for image-based accessors
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions (> 0)
 * @tparam kMode Access mode
 * @tparam kTarget Access target
 */
template <typename elemT, int kDims, access::mode kMode, access::target kTarget>
class accessor_image_interface
    : public detail::accessor_common<elemT, kDims, kMode, kTarget,
                                     access::placeholder::false_t> {
 private:
  static_assert(kDims > 0, "Image accessors don't allow zero dimensions");

 protected:
  static constexpr bool is_image_read_syntax = (kMode == access::mode::read);
  static constexpr bool is_image_write_syntax =
      (kMode == access::mode::write) || (kMode == access::mode::discard_write);
  static constexpr bool is_cl_float4 =
      std::is_same<elemT, cl::sycl::cl_float4>::value;
  static constexpr bool is_cl_half4 =
      std::is_same<elemT, cl::sycl::cl_half4>::value;
  static constexpr bool is_cl_int4 =
      std::is_same<elemT, cl::sycl::cl_int4>::value;
  static constexpr bool is_cl_uint4 =
      std::is_same<elemT, cl::sycl::cl_uint4>::value;

  using base_t = detail::accessor_common<elemT, kDims, kMode, kTarget,
                                         access::placeholder::false_t>;

  // Inherit constructors
  using base_t::base_t;

 public:
  /** Member function for reading an element of a read image accessor with
   * dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to read
   * from.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_float4 read(const coordT& coords) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readf(coords);
#else
    return cl::sycl::cl_float4(
        detail::read_imagef(this->get_device_ptr(), coords));
#endif
  }

  /** Member function for reading an element of a read image accessor with
   * dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to read
   * from.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_half4 read(const coordT& coords) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readh(coords);
#else
    return cl::sycl::cl_half4(
        detail::read_imageh(this->get_device_ptr(), coords));
#endif
  }

  /** Member function for reading an element of a read image accessor with
   * dataT cl_int4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to read
   * from.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_int4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_int4 read(const coordT& coords) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readi(coords);
#else
    return cl::sycl::cl_int4(
               detail::read_imagei(this->get_device_ptr(), coords))
        .convert<cl::sycl::cl_int, rounding_mode::automatic>();
#endif
  }

  /** Member function for reading an element of a read image accessor with
   * dataT cl_uint4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to read
   * from.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_uint4 read(const coordT& coords) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readui(coords);
#else
    return cl::sycl::cl_uint4(
        detail::read_imageui(this->get_device_ptr(), coords));
#endif
  }

  /** Member function for sampling a point in a read image accessor with
   * dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image to
   * sample from.
   * @param smpl The sampler to use when sampling from the image.
   * @return An element calculated from sampling the image using the coordinates
   * specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_float4 read(const coordT& coords,
                           const cl::sycl::sampler& smpl) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readf(coords, smpl);
#else
    return cl::sycl::cl_float4(
        detail::read_imagef(this->get_device_ptr(), smpl.m_sampler, coords));
#endif
  }

  /** Member function for sampling a point in a read image accessor with
   * dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image to
   * sample from.
   * @param smpl The sampler to use when sampling from the image.
   * @return An element calculated from sampling the image using the coordinates
   * specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_half4 read(const coordT& coords,
                          const cl::sycl::sampler& smpl) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readh(coords, smpl);
#else
    return cl::sycl::cl_half4(
        detail::read_imageh(this->get_device_ptr(), smpl.m_sampler, coords));
#endif
  }

  /** Member function for sampling a point in a read image accessor with
   * dataT cl_int4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image to
   * sample from.
   * @param smpl The sampler to use when sampling from the image.
   * @return An element calculated from sampling the image using the coordinates
   * specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_int4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_int4 read(const coordT& coords,
                         const cl::sycl::sampler& smpl) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readi(coords, smpl);
#else
    return cl::sycl::cl_int4(detail::read_imagei(this->get_device_ptr(),
                                                 smpl.m_sampler, coords))
        .convert<cl::sycl::cl_int, rounding_mode::automatic>();
#endif
  }

  /** Member function for sampling a point in a read image accessor with
   * dataT cl_uint4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image to
   * sample from.
   * @param smpl The sampler to use when sampling from the image.
   * @return An element calculated from sampling the image using the coordinates
   * specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_read_syntax && is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_uint4 read(const coordT& coords,
                          const cl::sycl::sampler& smpl) const {
#ifndef __SYCL_DEVICE_ONLY__
    return this->readui(coords, smpl);
#else
    return cl::sycl::cl_uint4(
        detail::read_imageui(this->get_device_ptr(), smpl.m_sampler, coords));
#endif
  }

  /** Member function for writing to an element of a write image accessor
   * with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to
   * write to.
   * @param color The value that is to be assigned to the element of the image.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_write_syntax && is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_float4& color) const {
#ifndef __SYCL_DEVICE_ONLY__
    this->writef(coords, color);
#else
    detail::write_imagef(this->get_device_ptr(), coords, color);
#endif
  }

  /** Member function for writing to an element of a write image accessor
   * with dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to
   * write to.
   * @param color The value that is to be assigned to the element of the image.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_write_syntax && is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_half4& color) const {
#ifndef __SYCL_DEVICE_ONLY__
    this->writeh(coords, color);
#else
    detail::write_imageh(this->get_device_ptr(), coords, color);
#endif
  }

  /** Member function for writing to an element of a write image accessor
   * with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to
   * write to.
   * @param color The value that is to be assigned to the element of the image.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_write_syntax && is_cl_int4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_int4& color) const {
#ifndef __SYCL_DEVICE_ONLY__
    this->writei(coords, color);
#else
    detail::write_imagei(this->get_device_ptr(), coords, color);
#endif
  }

  /** Member function for writing to an element of a write image accessor
   * with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image to
   * write to.
   * @param color The value that is to be assigned to the element of the image.
   * @return An element read from the image using the coordinates specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_image_write_syntax && is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_uint4& color) const {
#ifndef __SYCL_DEVICE_ONLY__
    this->writeui(coords, color);
#else
    detail::write_imageui(this->get_device_ptr(), coords, color);
#endif
  }

  COMPUTECPP_DEPRECATED_API("This is an internal function")
  inline int convert_coords(id<1> index) const { return int(index[0]); }

  COMPUTECPP_DEPRECATED_API("This is an internal function")
  inline int2 convert_coords(id<2> index) const {
    return int2(index[0], index[1]);
  }

  COMPUTECPP_DEPRECATED_API("This is an internal function")
  inline int4 convert_coords(id<3> index) const {
    return int4(index[0], index[1], index[2], 0);
  }
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// image accessor

/** Specialization for an image accessor
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions (> 0)
 * @tparam kMode Access mode
 */
template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT accessor<
    elemT, kDims, kMode, access::target::image, access::placeholder::false_t>
    : public detail::accessor_image_interface<elemT, kDims, kMode,
                                              access::target::image> {
 private:
  using base_t = detail::accessor_image_interface<elemT, kDims, kMode,
                                                  access::target::image>;

 public:
  /** Constructs an image accessor
   * @tparam AllocatorT Type of the image allocator
   * @param imageRef Image object where access is being requested
   * @param commandHandler Command group handler
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(image<kDims, AllocatorT>& imageRef, handler& commandHandler,
           const property_list& propList = {})
      : base_t{imageRef, commandHandler} {
    (void)propList;
  }

  /// @cond COMPUTECPP_DEV

  /** Constructs an image from a storage object
   * @internal
   * @param store Storage object
   * @param commandHandler Command group handler
   * @param accessRange Data access range
   */
  accessor(storage_mem&& store, handler& commandHandler,
           detail::access_range accessRange)
      : base_t{store, commandHandler, accessRange} {}

  /// COMPUTECPP_DEV @endcond

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ACCESSOR_H_
