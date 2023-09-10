/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file image.h
 *
 * @brief This file implements the @ref cl::sycl::image class as defined by the
 * SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_IMAGE_H_
#define RUNTIME_INCLUDE_SYCL_IMAGE_H_

#include "SYCL/allocator.h"
#include "SYCL/base.h"
#include "SYCL/buffer.h"
#include "SYCL/common.h"
#include "SYCL/error_log.h"
#include "SYCL/event.h"
#include "SYCL/include_opencl.h"
#include "SYCL/index_array.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/predefines.h"
#include "SYCL/range.h"
#include "SYCL/storage_mem.h"

#include <cstddef>
#include <memory>
#include <system_error>
#include <type_traits>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
class context;
class handler;
class property_list;

namespace property {
namespace image {

/** @brief The use_host_ptr property adds the requirement that the SYCL runtime
 *        must not allocate any memory for the image and instead uses the
 *        provided host pointer directly.
 */
using use_host_ptr = buffer::use_host_ptr;

/** @brief The use_mutex property adds the requirement that the memory which is
 *        owned by the SYCL image can be shared with the application via a
 *        mutex_class provided to the property. The mutex is locked by the
 *        runtime whenever the data is in use and unlocked otherwise. Data is
 *        synchronized with host data when the mutex is unlocked by the runtime.
 */
using use_mutex = buffer::use_mutex;

/** @brief The context_bound property adds the requirement that the SYCL image
 *        can only be associated with a single SYCL context that is provided to
 *        the property.
 */
using context_bound = buffer::context_bound;

}  // namespace image
}  // namespace property

/** @brief Specify the number of channels and the channel layout in which
 * channels are stored in the image.
 */
enum class image_channel_order : unsigned int {
  r = CL_R,
  a = CL_A,
  rg = CL_RG,
  ra = CL_RA,
  rgb = CL_RGB,
  rgba = CL_RGBA,
  bgra = CL_BGRA,
  argb = CL_ARGB,
  intensity = CL_INTENSITY,
  luminance = CL_LUMINANCE,
  rx = CL_Rx,
  rgx = CL_RGx,
  rgbx = CL_RGBx,
  abgr = 0x10C3,  // CL_ABGR
};

/** @brief Specify the size of the channels data type.
 */
enum class image_channel_type : unsigned int {
  snorm_int8 = CL_SNORM_INT8,
  snorm_int16 = CL_SNORM_INT16,
  unorm_int8 = CL_UNORM_INT8,
  unorm_int16 = CL_UNORM_INT16,
  unorm_short_565 = CL_UNORM_SHORT_565,
  unorm_short_555 = CL_UNORM_SHORT_555,
  unorm_int_101010 = CL_UNORM_INT_101010,
  signed_int8 = CL_SIGNED_INT8,
  signed_int16 = CL_SIGNED_INT16,
  signed_int32 = CL_SIGNED_INT32,
  unsigned_int8 = CL_UNSIGNED_INT8,
  unsigned_int16 = CL_UNSIGNED_INT16,
  unsigned_int32 = CL_UNSIGNED_INT32,
  fp16 = CL_HALF_FLOAT,
  fp32 = CL_FLOAT
};

/// @cond COMPUTECPP_DEV

/** @brief Public interface for the image class, which defines a shared image
 * that can be used by kernels in queues.
 *
 */
class COMPUTECPP_EXPORT image_mem : public storage_mem {
 protected:
  image_mem(string_class errorMessage) : storage_mem() {
    COMPUTECPP_CL_ERROR_CODE_MSG(CL_SUCCESS,
                                 detail::cpp_error_code::NOT_SUPPORTED_ERROR,
                                 nullptr, errorMessage);
    (void)errorMessage;
  }

  image_mem(shared_ptr_class<void> hostPointer, dim_t numDims,
            detail::index_array range, detail::index_array pitch,
            image_channel_order order, image_channel_type type,
            detail::pointer_origin pointerOrigin,
            unique_ptr_class<detail::base_allocator>&& bA,
            write_back enableWriteBack, const property_list& propertyList);

  /** @brief Interop constructor
   * @param memObject the user provided OpenCL image
   * @param numDims the dimensionality of the image
   * @param context the SYCL context where the image was created (can be an
   * interop context)
   * @param bA the allocator used on the host to allocate the data if needed
   */
  image_mem(cl_mem memObject, dim_t numDims, const context& context,
            unique_ptr_class<detail::base_allocator>&& bA);

  detail::index_array get_pitch_impl() const;

 public:
  ~image_mem() override = default;

  /** @brief Calculates the size of an image element based on the image channel
   *        order and type
   * @param order Image channel order
   * @param type Image channel type
   * @return size_t Size of the image element
   */
  static size_t calculate_element_size(image_channel_order order,
                                       image_channel_type type);
};

/// COMPUTECPP_DEV @endcond

namespace detail {

using byte_t = unsigned char;

/** @brief Helper struct to construct a range used for an image pitch,
 *        which uses one dimension less than the image it's used in
 * @tparam kDimensions Number of image dimensions
 */
template <int kDimensions>
struct pitch_range {
  /** @brief In the general case, just use a range of one dimension one less
   */
  using type = range<kDimensions - 1>;
};

/** @brief Specialization of the pitch range for 1-dimensional images
 */
template <>
struct pitch_range<1> {
  /** @brief This can be any valid type, as long as it's not a range<0>,
   *        and pitch should be SFINAE-d out anyway for 1D images
   */
  using type = std::integral_constant<int, 1>;
};

}  // namespace detail

template <int kDimensions = 1, typename AllocatorT = image_allocator>
class image : public image_mem {
  image() = delete;

  /** @brief Changes kDimensions into a dependant type
   */
  using kdims_t = typename std::integral_constant<int, kDimensions>::type;

  /** @brief Retrieve the type of the range used for the pitch
   */
  using pitch_range_t = typename detail::pitch_range<kDimensions>::type;

 public:
  /// @cond COMPUTECPP_DEV

  /// @brief Default destructor
  /// @internal
  ~image() override = default;

  /// COMPUTECPP_DEV @endcond

  /** @brief Copy constructor. Copy the image descriptor of the original image.
   * After the copy, both image object will point to the same underlying memory.
   */
  image(const image<kDimensions, AllocatorT>&) = default;

  /** @brief Move Constructor. Moves the image descriptor of the original image.
   * After the move, rhs will be invalid.
   */
  // NOLINTNEXTLINE(performance-noexcept-move-constructor)
  image(image<kDimensions, AllocatorT>&& rhs) = default;

  /** @brief Copy assignment.Copies the image descriptor of the original image.
   * After the copy, both image object will point to the same underlying memory.
   */
  image<kDimensions, AllocatorT>& operator=(
      const image<kDimensions, AllocatorT>& rhs) = default;

  /** @brief Move Assignment. Moves the image descriptor of the original image.
   * After the move, rhs will be invalid.
   */
  // NOLINTNEXTLINE(performance-noexcept-move-constructor)
  image<kDimensions, AllocatorT>& operator=(
      image<kDimensions, AllocatorT>&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const image& lhs, const image& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const image& lhs, const image& rhs) {
    return !(lhs == rhs);
  }

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with no host pointer, performing device side allocation of the buffer,
  this means that on destruction the data will not be copied back unless a final
  pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using
  the allocator specified by AllocatorT.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param propList List of image properties
  */
  image(cl::sycl::image_channel_order order, cl::sycl::image_channel_type type,
        const range<kDimensions>& rng, const property_list& propList = {})
      : image_mem(
            nullptr, kDimensions, rng, detail::index_array(0, 0, 0), order,
            type, detail::pointer_origin::none,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with no host pointer, performing device side allocation of the buffer,
  this means that on destruction the data will not be copied back unless a final
  pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  image(cl::sycl::image_channel_order order, cl::sycl::image_channel_type type,
        const range<kDimensions>& rng, AllocatorT allocator,
        const property_list& propList = {})
      : image_mem(
            nullptr, kDimensions, rng, detail::index_array(0, 0, 0), order,
            type, detail::pointer_origin::none,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with no host pointer, performing device side allocation of the buffer,
  this means that on destruction the data will not be copied back unless a final
  pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using
  the allocator specified by AllocatorT.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(cl::sycl::image_channel_order order, cl::sycl::image_channel_type type,
        const range<kDimensions>& rng, const pitch_range_t& pit,
        const property_list& propList = {})
      : image_mem(
            nullptr, kDimensions, rng, pit, order, type,
            detail::pointer_origin::none,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with no host pointer, performing device side allocation of the buffer,
  this means that on destruction the data will not be copied back unless a final
  pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(cl::sycl::image_channel_order order, cl::sycl::image_channel_type type,
        const range<kDimensions>& rng, const pitch_range_t& pit,
        AllocatorT allocator, const property_list& propList = {})
      : image_mem(
            nullptr, kDimensions, rng, pit, order, type,
            detail::pointer_origin::none,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a raw host pointer to the image data. On object destruction, the
  data will be copied to the specified host pointer unless a final pointer is
  specified using set_final_data() in which case that specified pointer will be
  used.

  Any host side allocation of data will be performed using the allocator
  specified by AllocatorT.
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param propList List of image properties
  */
  image(void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(hostPtr, detail::NullDeleter()), kDimensions,
            rng, detail::index_array(0, 0, 0), order, type,
            detail::pointer_origin::raw,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a raw host pointer to the image data. On object destruction, the
  data will be copied to the specified host pointer unless a final pointer is
  specified using set_final_data() in which case that specified pointer will be
  used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  image(void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        AllocatorT allocator, const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(hostPtr, detail::NullDeleter()), kDimensions,
            rng, detail::index_array(0, 0, 0), order, type,
            detail::pointer_origin::raw,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a constant raw host pointer to the image data. On object
  destruction, the data will not be copied back, unless a final pointer is
  specified using set_final_data() in which case that specified pointer will be
  used.

  Any host side allocation of data will be performed using the allocator
  specified by AllocatorT.
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param propList List of image properties
  */
  image(const void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(const_cast<void*>(hostPtr),
                                   detail::NullDeleter()),
            kDimensions, rng, detail::index_array(0, 0, 0), order, type,
            detail::pointer_origin::raw_const,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::disable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a raw host pointer to the image data. On object destruction, the
  data will not be copied back unless a final pointer is
  specified using set_final_data() in which case that specified pointer will be
  used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  image(const void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        AllocatorT allocator, const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(const_cast<void*>(hostPtr),
                                   detail::NullDeleter()),
            kDimensions, rng, detail::index_array(0, 0, 0), order, type,
            detail::pointer_origin::raw_const,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::disable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type,
  range and pitch, with a raw host pointer to the image data. On object
  destruction, the data will be copied to the specified host pointer unless a
  final pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using
  the allocator specified by AllocatorT.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const pitch_range_t& pit, const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(hostPtr, detail::NullDeleter()), kDimensions,
            rng, pit, order, type, detail::pointer_origin::raw,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type,
  range and pitch, with a raw host pointer to the image data. On object
  destruction, the data will be copied to the specified host pointer unless a
  final pointer is specified using set_final_data() in which case that specified
  pointer will be used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param hostPtr Raw pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(void* hostPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const pitch_range_t& pit, AllocatorT allocator,
        const property_list& propList = {})
      : image_mem(
            shared_ptr_class<void>(hostPtr, detail::NullDeleter()), kDimensions,
            rng, pit, order, type, detail::pointer_origin::raw,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a shared host pointer to the image data.

  The host pointer's ownership is shared and on destruction the data will be
  copied to the specified host pointer unless the runtime maintains the last
  reference to the shared_ptr or a final pointer is specified using
  set_final_data() in which case that specified pointer will be used. Any host
  side allocation of data will be performed using the allocator specified by
  AllocatorT.
  @param sharedPtr shared pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  image(shared_ptr_class<void> sharedPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const property_list& propList = {})
      : image_mem(
            sharedPtr, kDimensions, rng, detail::index_array(0, 0, 0), order,
            type, detail::pointer_origin::shared,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type and
  range, with a shared host pointer to the image data.

  The host pointer's ownership is shared and on destruction the data will be
  copied to the specified host pointer unless the runtime maintains the last
  reference to the shared_ptr or a final pointer is specified using
  set_final_data() in which case that specified pointer will be used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @param sharedPtr shared pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  image(shared_ptr_class<void> sharedPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        AllocatorT allocator, const property_list& propList = {})
      : image_mem(
            sharedPtr, kDimensions, rng, detail::index_array(0, 0, 0), order,
            type, detail::pointer_origin::shared,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type,
  range and pitch, with a shared host pointer to the image data.

  The host pointer's ownership is shared and on destruction the data will be
  copied to the specified host pointer unless the runtime maintains the last
  reference to the shared_ptr or a final pointer is specified using
  set_final_data() in which case that specified pointer will be used.
  Any host side allocation of data will be performed using the allocator
  specified by AllocatorT.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param sharedPtr Shared pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(shared_ptr_class<void> sharedPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const pitch_range_t& pit, const property_list& propList = {})
      : image_mem(
            sharedPtr, kDimensions, rng, pit, order, type,
            detail::pointer_origin::shared,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type)),
            write_back::enable_write_back, propList) {}

  /**
  @brief Construct an image of the specified channel_order and channel_type,
  range and pitch, with a shared host pointer to the image data.

  The host pointer's ownership is shared and on destruction the data will be
  copied to the specified host pointer unless the runtime maintains the last
  reference to the shared_ptr or a final pointer is specified using
  set_final_data() in which case that specified pointer will be used.

  Any host side allocation of data will be performed using the provided
  allocator.
  @tparam COMPUTECPP_ENABLE_IF Only enabled when kDimensions greater than 1
  @param sharedPtr Shared pointer to the image data.
  @param order Image channel order.
  @param type Image channel type.
  @param rng Image range.
  @param pit Image pitch.
  @param allocator The allocator used to create internal storage
  @param propList List of image properties
  */
  template <COMPUTECPP_ENABLE_IF(kdims_t, (kDimensions > 1))>
  image(shared_ptr_class<void> sharedPtr, cl::sycl::image_channel_order order,
        cl::sycl::image_channel_type type, const range<kDimensions>& rng,
        const pitch_range_t& pit, AllocatorT allocator,
        const property_list& propList = {})
      : image_mem(
            sharedPtr, kDimensions, rng, pit, order, type,
            detail::pointer_origin::shared,
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                image_mem::calculate_element_size(order, type), allocator),
            write_back::enable_write_back, propList) {}

  image(cl_mem memObject, const context& syclContext, event availableEvent = {})
      : image_mem(
            memObject, kDimensions, syclContext,
            // 4 guarantees the size of the element is large enough
            detail::make_base_allocator<byte, AllocatorT>::get_image_allocator(
                4)) {
    (void)availableEvent;
  }

  /** @brief Create an accessor to the image.
   *
   * @tparam T Typename of the accessor
   * @tparam accessMode The data access mode descriptor
   */
  template <typename T, access::mode accessMode>
  accessor<T, kDimensions, accessMode, access::target::host_image>
  get_access() {
    return accessor<T, kDimensions, accessMode, access::target::host_image>(
        *this);
  }

  /** @brief Create an accessor to the image for command group \ref handler.
   *
   * @tparam T Typename of the accessor
   * @tparam accessMode The data access mode descriptor
   * @param cgh The command group handler
   */
  template <typename T, access::mode accessMode>
  accessor<T, kDimensions, accessMode, access::target::image> get_access(
      handler& cgh /* NOLINT */) {
    return accessor<T, kDimensions, accessMode, access::target::image>(*this,
                                                                       cgh);
  }

  /** @brief Returns the pitch of the image object.
   * @return pitch
   */
  template <int dims = kDimensions,
            class = typename std::enable_if<(2 == dims) || (3 == dims)>::type>
  range<dims - 1> get_pitch() const {
    return range<dims - 1>(this->get_pitch_impl());
  }

  /** @return The image range
   */
  cl::sycl::range<kDimensions> get_range() const {
    return cl::sycl::range<kDimensions>(this->get_range_impl());
  }

  /** @brief Returns whether this SYCL image was constructed with the property
   *        specified by propertyT
   * @tparam propertyT Property to check for
   * @return True if image constructed with the property
   */
  template <typename propertyT>
  bool has_property() const noexcept {
    return this->get_properties().template has_property<propertyT>();
  }

  /** @brief Returns a copy of the property of type propertyT that this SYCL
   *        image was constructed with. Throws an error if image was not
   *        constructed with the property.
   * @tparam propertyT Property to retrieve
   * @return Copy of the property
   */
  template <typename propertyT>
  propertyT get_property() const {
    return this->get_properties().template get_property<propertyT>();
  }

  /** @brief Returns the allocator provided to the image
   * @return Allocator that was provided to the image
   */
  AllocatorT get_allocator() const {
    return detail::cast_base_allocator<AllocatorT>(this->get_base_allocator());
  }
};

#if SYCL_LANGUAGE_VERSION >= 202001
/** Property trait specializations. Note, specializations of is_property are not
 * needed for image properties as they are just aliases to buffer property types
 */
template <int dimensions, typename AllocatorT>
struct is_property_of<property::image::use_host_ptr,
                      image<dimensions, AllocatorT>> : public std::true_type {};

template <int dimensions, typename AllocatorT>
struct is_property_of<property::image::context_bound,
                      image<dimensions, AllocatorT>> : public std::true_type {};

template <int dimensions, typename AllocatorT>
struct is_property_of<property::image::use_mutex, image<dimensions, AllocatorT>>
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
template <int kDimensions, typename AllocatorT>
struct hash<cl::sycl::image<kDimensions, AllocatorT>> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(
      const cl::sycl::image<kDimensions, AllocatorT>& object) const {
    hash<cl::sycl::dmem_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
#endif  // RUNTIME_INCLUDE_SYCL_IMAGE_H_
