/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
 * @file image_array_accessor.h
 */

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ARRAY_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ARRAY_ACCESSOR_H_

#include "SYCL/accessor.h"
#include "SYCL/accessor/image_accessor.h"
#include "SYCL/image.h"

namespace cl {
namespace sycl {

namespace detail {

////////////////////////////////////////////////////////////////////////////////
// image_array_slice

/** @brief Utility function which constructs coordinates for an image array
 * slice by combining the image coordinates and the image array index. Overload
 * for constructing 2-element coordinates.
 * @tparam elementT The element type of the coordinates.
 * @param coords The image coordinates.
 * @param arrayIndex The image array index.
 * @return The coordinates for the image array slice.
 */
template <typename elementT>
sycl::vec<elementT, 2> make_image_array_slice_coords(elementT coords,
                                                     size_t arrayIndex) {
  return sycl::vec<elementT, 2>(coords, static_cast<elementT>(arrayIndex));
}

/** @brief Utility function which constructs coordinates for an image array
 * slice by combining the image coordinates and the image array index. Overload
 * for constructing 4-element coordinates.
 * @tparam elementT The element type of the coordinates.
 * @param coords The image coordinates.
 * @param arrayIndex The image array index.
 * @return The coordinates for the image array slice.
 */
template <typename elementT>
sycl::vec<elementT, 4> make_image_array_slice_coords(
    sycl::vec<elementT, 2> coords, size_t arrayIndex) {
  return sycl::vec<elementT, 4>(coords, static_cast<elementT>(arrayIndex),
                                static_cast<elementT>(0));
}

/** @brief Intermediate class template which contains a reference to an image
 * array accessors and an image array index. Used to provide acc[i].read(..)
 * syntax.
 * @tparam elemT The element type of the image array accessor.
 * @tparam kDims The dimensionality of the image array accessor.
 * @tparam kMode The access mode of the image array accessor.
 */
template <typename elemT, int kDims, access::mode kMode>
class image_array_slice {
  /* Friend declaration to the associated specialisation of the accessor class.
   */
  friend class cl::sycl::accessor<elemT, kDims, kMode,
                                  access::target::image_array,
                                  access::placeholder::false_t>;

  static constexpr bool is_cl_float4 =
      (std::is_same<elemT, cl::sycl::cl_float4>::value);
  static constexpr bool is_cl_half4 =
      (std::is_same<elemT, cl::sycl::cl_half4>::value);
  static constexpr bool is_cl_int4 =
      (std::is_same<elemT, cl::sycl::cl_int4>::value);
  static constexpr bool is_cl_uint4 =
      (std::is_same<elemT, cl::sycl::cl_uint4>::value);

 public:
  /**
   * @brief Member function for reading an element of a read image array
   * accessor with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to read from.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_float4 read(const coordT& coords) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readf(imageArraySliceCoords);
#else
    return detail::read_imagef(m_imageArrayAccRef.get_device_ptr(),
                               imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for reading an element of a read image array
   * accessor with dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to read from.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_half4 read(const coordT& coords) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readh(imageArraySliceCoords);
#else
    return detail::read_imageh(m_imageArrayAccRef.get_device_ptr(),
                               imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for reading an element of a read image array
   * accessor with dataT cl_int4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to read from.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(
                elemT, (is_cl_int4 && detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_int4 read(const coordT& coords) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readi(imageArraySliceCoords);
#else
    return detail::read_imagei(m_imageArrayAccRef.get_device_ptr(),
                               imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for reading an element of a read image array
   * accessor with dataT cl_uint4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to read from.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_uint4 read(const coordT& coords) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readui(imageArraySliceCoords);
#else
    return detail::read_imageui(m_imageArrayAccRef.get_device_ptr(),
                                imageArraySliceCoords);

#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for sampling a point in a read image array accessor
   * with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image
   * array to sample from.
   * @param smpl The sampler to use when sampling from the image array.
   * @return An element calculated from sampling the image array using the
   * coordinates specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_float4 read(const coordT& coords,
                           const cl::sycl::sampler& smpl) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readf(imageArraySliceCoords, smpl);
#else
    return detail::read_imagef(m_imageArrayAccRef.get_device_ptr(),
                               smpl.m_sampler, imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for sampling a point in a read image array accessor
   * with dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image
   * array to sample from.
   * @param smpl The sampler to use when sampling from the image array.
   * @return An element calculated from sampling the image array using the
   * coordinates specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_half4 read(const coordT& coords,
                          const cl::sycl::sampler& smpl) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readh(imageArraySliceCoords, smpl);
#else
    return detail::read_imageh(m_imageArrayAccRef.get_device_ptr(),
                               smpl.m_sampler, imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for sampling a point in a read image array accessor
   * with dataT cl_int4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image
   * array to sample from.
   * @param smpl The sampler to use when sampling from the image array.
   * @return An element calculated from sampling the image array using the
   * coordinates specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(
                elemT, (is_cl_int4 && detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_int4 read(const coordT& coords,
                         const cl::sycl::sampler& smpl) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readi(imageArraySliceCoords, smpl);
#else
    return detail::read_imagei(m_imageArrayAccRef.get_device_ptr(),
                               smpl.m_sampler, imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for sampling a point in a read image array accessor
   * with dataT cl_uint4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates which specify the point within the image
   * array to sample from.
   * @param smpl The sampler to use when sampling from the image array.
   * @return An element calculated from sampling the image array using the
   * coordinates specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  cl::sycl::cl_uint4 read(const coordT& coords,
                          const cl::sycl::sampler& smpl) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    return m_imageArrayAccRef.readui(imageArraySliceCoords, smpl);
#else
    return detail::read_imageui(m_imageArrayAccRef.get_device_ptr(),
                                smpl.m_sampler, imageArraySliceCoords);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for writing to an element of a write image array
   * accessor with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to write to.
   * @param color The value that is to be assigned to the element of the image
   * array.
   * @return An element read from the image using the coordinates specified and
   * the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_float4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_float4& color) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    m_imageArrayAccRef.writef(imageArraySliceCoords, color);
#else
    detail::write_imagef(m_imageArrayAccRef.get_device_ptr(),
                         imageArraySliceCoords, color);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for writing to an element of a write image array
   * accessor with dataT cl_half4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to write to.
   * @param color The value that is to be assigned to the element of the image
   * array.
   * @return An element read from the image array using the coordinates
   * specified.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_half4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_half4& color) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    m_imageArrayAccRef.writeh(imageArraySliceCoords, color);
#else
    detail::write_imageh(m_imageArrayAccRef.get_device_ptr(),
                         imageArraySliceCoords, color);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for writing to an element of a write image array
   * accessor with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to write to.
   * @param color The value that is to be assigned to the element of the image
   * array.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(
                elemT, (is_cl_int4 && detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_int4& color) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    m_imageArrayAccRef.writei(imageArraySliceCoords, color);
#else
    detail::write_imagei(m_imageArrayAccRef.get_device_ptr(),
                         imageArraySliceCoords, color);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /**
   * @brief Member function for writing to an element of a write image array
   * accessor with dataT cl_float4.
   * @tparam coordT The type of the coordinates parameter.
   * @tparam COMPUTECPP_ENABLE_IF condition.
   * @param coords The coordinates to specify which element of the image array
   * to write to.
   * @param color The value that is to be assigned to the element of the image
   * array.
   * @return An element read from the image array using the coordinates
   * specified and the image array index.
   */
  template <typename coordT,
            COMPUTECPP_ENABLE_IF(elemT,
                                 (is_cl_uint4 &&
                                  detail::is_coords<kDims, coordT>::value))>
  void write(const coordT& coords, const cl::sycl::cl_uint4& color) const {
    auto imageArraySliceCoords =
        make_image_array_slice_coords(coords, m_arrayIndex);
#ifndef __SYCL_DEVICE_ONLY__
    m_imageArrayAccRef.writeui(imageArraySliceCoords, color);
#else
    detail::write_imageui(m_imageArrayAccRef.get_device_ptr(),
                          imageArraySliceCoords, color);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Constructor which takes a reference to an image array accessor and
   * image array index and initialises the respective members.
   * @param imageArrayAccRef The reference to an image array accessor.
   * @param arrayIndex The image array index.
   */
  image_array_slice(
      const accessor_common<
          elemT, kDims, kMode, cl::sycl::access::target::image_array,
          cl::sycl::access::placeholder::false_t>& imageArrayAccRef,
      size_t arrayIndex)
      : m_imageArrayAccRef(imageArrayAccRef), m_arrayIndex(arrayIndex) {
    m_arrayIndex = arrayIndex;
  }

  const accessor_common<elemT, kDims, kMode, access::target::image_array,
                        access::placeholder::false_t>& m_imageArrayAccRef;
  size_t m_arrayIndex;
};

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
// image_array accessor

template <typename elemT, int kDims, access::mode kMode>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT
    accessor<elemT, kDims, kMode, access::target::image_array,
             access::placeholder::false_t>
    : public detail::accessor_common<elemT, kDims, kMode,
                                     access::target::image_array,
                                     access::placeholder::false_t> {
 private:
  static_assert((kDims == 1) || (kDims == 2),
                "Image array accessors are limited to 1 or 2 dimensions");

  using accessor_common =
      detail::accessor_common<elemT, kDims, kMode, access::target::image_array,
                              access::placeholder::false_t>;

 public:
  /** Constructs an image_array accessor
   * by taking an image object of dimensionality one greater than this accessor
   * and initialises the base class with the image.
   * @tparam AllocatorT Specifies the allocator type.
   * @param imageRef Reference to the image object being accessed.
   * @param commandHandler Reference to the handler of the command group the
   * accessor is being constructed within.
   * @param propList Additional properties
   */
  template <typename AllocatorT>
  accessor(image<(kDims + 1), AllocatorT>& imageRef, handler& commandHandler,
           const property_list& propList = {})
      : accessor_common{imageRef, commandHandler} {
    (void)propList;
  }

  detail::image_array_slice<elemT, kDims, kMode> operator[](
      size_t index) const {
    return {*this, index};
  }

} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR;

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_IMAGE_ARRAY_ACCESSOR_H_
