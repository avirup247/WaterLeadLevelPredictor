/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file accessor_args.h
  @brief Internal file used by the @ref cl::sycl::accessor class
*/

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_ARGS_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_ARGS_H_

#include "SYCL/common.h"
#include "SYCL/index_array.h"
#include "SYCL/multi_pointer.h"

namespace cl {
namespace sycl {

/** @cond COMPUTECPP_DEV */
/*******************************************************************************
    forward declarations
*******************************************************************************/

/**
  This file contains:
  - forward declarations
  - @ref device_index_array
  - @ref device_arg
  - @ref device_arg_container

  The device_index_array is used as the range field of the accessor class on the
  device side. This can be used to alter the size of kernel arguments, as some
  OpenCL drivers do not allow kernel arguments larger than 8 bytes. This work
  around simply ensures that the device_index_array class is always of 8 bytes
  by
  changing the device_index_array element type based on the address bits of the
  compiler. See @ref subscript_op.
*/
class handler;
namespace detail {
class accessor;
// Forward declaration of the host definition of device_arg_container
struct host_arg_container;
}  // namespace detail

/*******************************************************************************
    access_range struct
*******************************************************************************/

namespace detail {

/**
@brief Struct for storing the range and offset information of an accessor
requested access to data. Used for maintaining a region of access that an
accessor has.
*/
struct access_range {
  /**
  @brief Inline constructor that takes an offset and range, and initializes the
  relative struct fields.
  @param accessOffset The offset of the access range.
  @param accessRange The range of the access range.
  */
  inline access_range(index_array accessOffset, index_array accessRange)
      : offset(accessOffset), range(accessRange) {}

  /**
  @brief Inline comparison operator that compares the values of both the origin
  and the range and returns true if the are all equal.
  @rhs The right hand side of the comparison expression.
  */
  inline bool operator==(const access_range& rhs) {
    return (range[0] == rhs.range[0] && range[1] == rhs.range[1] &&
            range[2] == rhs.range[2] && offset[0] == rhs.offset[0] &&
            offset[1] == rhs.offset[1] && offset[2] == rhs.offset[2]);
  }

  /** @brief Calculates the number of dimensions
   *        required to describe this access range
   * @return Number of dimensions of the access range
   */
  int num_dimensions() const noexcept {
    const auto is_point_dimension = [this](const int dim) noexcept {
      return (this->range[dim] == 1) && (this->offset[dim] == 0);
    };
    if (is_point_dimension(1)) {
      // 2nd dimension is a single point
      if (is_point_dimension(2)) {
        // 3rd dimension is a single point
        return 1;
      } else {
        return 3;
      }
    } else {
      // At least two dimensions
      if (is_point_dimension(2)) {
        // 3rd dimension is a single point
        return 2;
      } else {
        return 3;
      }
    }
  }

  /**
  @brief The offset of the access range.
  */
  index_array offset;

  /**
  @brief The range of the access range.
  */
  index_array range;
};

/*******************************************************************************
    device_index_array class
*******************************************************************************/

/** @brief Container of 3 integer indexes which are stored as sized integer
 * types such that an instance is equal to the sizeof(size_t). Provides a
 * subscript operator which returns an element of the index, converted to
 * size_t.
 * @note The actual size is 4 for compatibility reasons.
 *       The last element is never used, but is initialized to 1
 *       in order for all constexpr operations to be valid.
 */
class device_index_array
    : public detail::id_range_base<
          typename detail::device_arg_info<sizeof(void*)>::elem_type, 4,
          device_index_array> {
 private:
  using base_t = detail::id_range_base<
      typename detail::device_arg_info<sizeof(void*)>::elem_type, 4,
      device_index_array>;

 public:
  using base_t::value_type;

  /** @brief Constructs a device_index_array initializing all indexes to zero.
   */
  constexpr device_index_array() noexcept
      : base_t{typename base_t::values_tag{}, 0, 0, 0, 1} {}

  /** @brief Constructs a device_index_array from 3 size_t parameters
   * initializing each index to the corresponding parameter.
   * @param elem0 The zeroth index.
   * @param elem1 The first index.
   * @param elem2 The second index.
   */
  constexpr device_index_array(size_t elem0, size_t elem1, size_t elem2)
      : base_t{typename base_t::values_tag{}, elem0, elem1, elem2, 1} {}

  /** @brief Constructs a device_index_array from a detail::index_array
   * initializing each index to the corresponding element of the
   * detail::index_array parameter, cast to the index type specified by
   * device_arg_info.
   * @param indexArray The detail::index_array that the device_index_array is to
   * be constructed from.
   */
  explicit COMPUTECPP_CONSTEXPR_CPP14 device_index_array(
      const detail::index_array& indexArray)
      : base_t{typename base_t::values_tag{}, indexArray[0], indexArray[1],
               indexArray[2], 1} {}

  /** Converts this object to an index_array
   */
  explicit COMPUTECPP_CONSTEXPR_CPP14 operator detail::index_array() const {
    return {static_cast<size_t>(this->operator[](0)),
            static_cast<size_t>(this->operator[](1)),
            static_cast<size_t>(this->operator[](2))};
  }

  /** Calculates the number of elements covered by instance
   * @return Number of elements across three dimensions
   */
  COMPUTECPP_CONSTEXPR_CPP14 size_t size() const noexcept {
    return static_cast<size_t>(this->get(0) * this->get(1) * this->get(2));
  }
};

}  // namespace detail

/*******************************************************************************
    device_arg struct
*******************************************************************************/

/**
@brief Specialized template struct that contains a pointer or OpenCL image
object with the relevant address space or access qualifiers and aliases to
reference and pointer types with the relevant address space or access
qualifiers, and to the explicit pointer class. This is determined by the access
mode, the access target and the dimensions of the containing specialization of
the accessor template class:
- Access target access::target::global_buffer for all access modes and
dimensions has a pointer with the global address space qualifier.
- Access target access::target::constant_buffer for all access modes and
dimensions has a pointer with the constant address space qualifier.
- Access target access::target::local for all access modes and dimensions has a
pointer with the local address space qualifier.
- Access target access::target::host_buffer for all access modes and dimensions
has a pointer with no address space qualifier.
- Access target access::target::image has one of the OpenCL image types
(image1d_t, image2d_t, iage3d_t) depending on the dimensions with the read_only
access qualifier for access mode access::mode::read and the write_only access
qualifier for access mode access::mode::write.
- Access target access::target::host_image for all access modes and dimensions
all has a pointer with no address space qualifier.
All non-host specialization also have the COMPUTECPP_CONVERT_ATTR macro attached
to
the end of the struct declaration, this is used during the compilers parameter
flattening mechanism; where for the device side the SYCL public interface
structures are recursively deconstructed down to OpenCL kernel parameters that
are passed to the kernel function. This class defines the argument types for the
accessor when compiled for the device, however as the accessor class is compiled
for both host and device these structures must be host side compatible.
@tparam elemT Specifies the element type of the pointer and typedef.
@tparam kDims Specifies the dimensions.
@tparam kMode Specifies the access mode.
@tparam kTarget Specifies the access target.
*/
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder = access::placeholder::false_t>
struct device_arg;

#ifdef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT                                   \
  [[computecpp::opencl_mirror_convert(detail::host_arg_container)]]
#else  // !__SYCL_DEVICE_ONLY__
#define COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT
#endif  // __SYCL_DEVICE_ONLY__

namespace detail {

#if defined(COMPUTECPP_EXT_READ_ACC_CONST_PTR_ENABLE)

/** Specifies the data type to use for storing accessor data,
 *  whether it's const qualified or not
 * @tparam elemT Original data type
 * @tparam kMode Access mode used when accessing the data
 */
template <typename elemT, access::mode kMode>
using device_arg_element_t =
    detail::conditional_t<kMode == access::mode::read, const elemT, elemT>;

#else

/** Specifies the data type to use for storing accessor data
 * @tparam elemT Original data type
 */
template <typename elemT, access::mode>
using device_arg_element_t = elemT;

#endif  // COMPUTECPP_EXT_READ_ACC_CONST_PTR_ENABLE

}  // namespace detail

/**
@brief Specialization of device_arg for access::target::global_buffer.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::global_buffer> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = global_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::global_buffer
placeholder.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::global_buffer,
                  access::placeholder::true_t> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = global_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR_PLACEHOLDER;

/**
@brief Specialization of device_arg for access::target::constant_buffer.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::constant_buffer> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = constant_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::constant_buffer.
@tparam elemT the type or the argument.
@tparam kDims the dimensionality of the buffer
@tparam kMode the access mode of the argument
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::constant_buffer,
                  access::placeholder::true_t> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = constant_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR_PLACEHOLDER;

/**
@brief Specialization of device_arg for access::target::local.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::local> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = local_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR;

template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::subgroup_local> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using ptr_class_type = codeplay::subgroup_local_ptr<value_type>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = typename ptr_class_type::reference_t;
  using ptr_type = typename ptr_class_type::pointer_t;

  inline value_type* get_ptr() const { return ptr_class_type(m_ptr); }

 private:
  ptr_type m_ptr;
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_buffer.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::host_buffer> {
  using value_type = detail::device_arg_element_t<elemT, kMode>;
  using raw_ref_type = value_type&;
  using raw_ptr_type = value_type*;
  using ref_type = value_type&;
  using ptr_type = value_type*;
  using ptr_class_type = value_type*;

  inline value_type* get_ptr() const { return m_ptr; }

 private:
  value_type* m_ptr;
};

/**
@brief Specialization of device_arg for access::target::image,
access::mode::read and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::read, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_ro_t;
  using ptr_type = __sycl_image1d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::read and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::read, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_ro_t;
  using ptr_type = __sycl_image2d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::read and 3 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 3, access::mode::read, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image3d_ro_t;
  using ptr_type = __sycl_image3d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image3d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::write and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::write, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_wo_t;
  using ptr_type = __sycl_image1d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::write and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::write, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_wo_t;
  using ptr_type = __sycl_image2d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::write and 3 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 3, access::mode::write, access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image3d_wo_t;
  using ptr_type = __sycl_image3d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image3d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::discard_write and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::discard_write,
                  access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_wo_t;
  using ptr_type = __sycl_image1d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::discard_write and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::discard_write,
                  access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_wo_t;
  using ptr_type = __sycl_image2d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image,
access::mode::discard_write and 3 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 3, access::mode::discard_write,
                  access::target::image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image3d_wo_t;
  using ptr_type = __sycl_image3d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image3d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::read and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::read, access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_array_ro_t;
  using ptr_type = __sycl_image1d_array_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_array_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::read and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::read, access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_array_ro_t;
  using ptr_type = __sycl_image2d_array_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

  // Temporary fix!
  // private:
  __sycl_image2d_array_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::write and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::write, access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_array_wo_t;
  using ptr_type = __sycl_image1d_array_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_array_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::write and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::write, access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_array_wo_t;
  using ptr_type = __sycl_image2d_array_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_array_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::discard_write and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::discard_write,
                  access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_array_wo_t;
  using ptr_type = __sycl_image1d_array_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_array_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::image_array,
access::mode::discard_write and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::discard_write,
                  access::target::image_array> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_array_wo_t;
  using ptr_type = __sycl_image2d_array_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_array_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::read and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::read, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_ro_t;
  using ptr_type = __sycl_image1d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::read and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::read, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_ro_t;
  using ptr_type = __sycl_image2d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::read and 3 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 3, access::mode::read, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image3d_ro_t;
  using ptr_type = __sycl_image3d_ro_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image3d_ro_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::write and 1 dimension.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 1, access::mode::write, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image1d_wo_t;
  using ptr_type = __sycl_image1d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image1d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::write and 2 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 2, access::mode::write, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image2d_wo_t;
  using ptr_type = __sycl_image2d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image2d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image,
access::mode::write and 3 dimensions.
@ref cl::sycl::device_arg.
*/
template <typename elemT>
struct device_arg<elemT, 3, access::mode::write, access::target::host_image> {
#if defined(__SYCL_DEVICE_ONLY__)
  using ref_type = __sycl_image3d_wo_t;
  using ptr_type = __sycl_image3d_wo_t;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  __sycl_image3d_wo_t m_ptr;
#else   // defined(__SYCL_DEVICE_ONLY__)
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

 private:
  void* m_ptr;
#endif  // defined(__SYCL_DEVICE_ONLY__)
 public:
  inline ptr_type get_ptr() const { return m_ptr; }
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg for access::target::host_image.
@ref cl::sycl::device_arg.
*/
template <typename elemT, int kDims, access::mode kMode>
struct device_arg<elemT, kDims, kMode, access::target::host_image> {
  using ref_type = void*;
  using ptr_type = void*;
  using raw_ref_type = ref_type;
  using raw_ptr_type = ptr_type;
  using ptr_class_type = void*;

  inline ptr_type get_ptr() const { return m_ptr; }

 private:
  void* m_ptr;
};

/*******************************************************************************
    device_arg_container struct
*******************************************************************************/

/**
@brief Specialized template struct that contains the device side arguments for
the accessor class. This is determined by the access target. Access targets
access::target::global_buffer, access::target::constant_buffer,
access::target::local and access::target::host_buffer for all access modes and
dimensions contain a device_arg object with the same template arguments and an
device_index_array object. Access targets access::target:image and
access::target::host_image for all access modes and dimensions contain only a
device_arg object. All non-host specialization also have the
COMPUTECPP_CONVERT_ATTR
macro attached to the end of the struct declaration, this is used during the
compilers parameter flattening mechanism.
@tparam elemT Specifies the element type of the pointer and alias.
@tparam kDims Specifies the dimensions.
@tparam kMode Specifies the access mode.
@tparam kTarget Specifies the access target.
*/
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder = access::placeholder::false_t>
struct COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT device_arg_container {
  device_arg<elemT, kDims, kMode, kTarget, isPlaceholder> m_deviceArg;
  detail::device_index_array m_offset;
  detail::device_index_array m_range;
  detail::device_index_array m_fullRange;
  detail::plane_id_t m_planeId;
} COMPUTECPP_CONVERT_ATTR;

template <typename elemT, int kDims, access::mode kMode, access::target kTarget>
struct device_arg_container_image {
  device_arg<elemT, kDims, kMode, kTarget> m_deviceArg;
  detail::device_index_array m_range;
};

/**
@brief Specialization of device_arg_container for access::target::image.
@ref cl::sycl::device_arg_container.
*/
template <typename elemT, int kDims, access::mode kMode>
struct COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT
    device_arg_container<elemT, kDims, kMode, access::target::image>
    : public device_arg_container_image<elemT, kDims, kMode,
                                        access::target::image> {
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg_container for access::target::host_image.
@ref cl::sycl::device_arg_container.
*/
template <typename elemT, int kDims, access::mode kMode>
struct COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT
    device_arg_container<elemT, kDims, kMode, access::target::host_image>
    : public device_arg_container_image<elemT, kDims, kMode,
                                        access::target::host_image> {
} COMPUTECPP_CONVERT_ATTR;

/**
@brief Specialization of device_arg_container for access::target::image_array.
@ref cl::sycl::device_arg_container.
*/
template <typename elemT, int kDims, access::mode kMode>
struct COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT
    device_arg_container<elemT, kDims, kMode, access::target::image_array>
    : public device_arg_container_image<elemT, kDims, kMode,
                                        access::target::image_array> {
} COMPUTECPP_CONVERT_ATTR;

#undef COMPUTECPP_DEVICE_ARG_MIRROR_CONVERT

/******************************************************************************/

}  // namespace sycl
}  // namespace cl

/******************************************************************************/

/** COMPUTECPP_DEV @endcond */

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_ARGS_H_

/******************************************************************************/
