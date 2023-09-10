/*****************************************************************************

    Copyright (C) 2002-2020 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
  @file accessor.h

  @brief This file contains the API for the public facing
  @ref cl::sycl::accessor class


  @detail This file contains:
  - @ref ACCESSOR_ALIGNMENT
  - @ref accessor

  The template class accessor is defined entirely in the header file and is
  defined for both the host side and the device side.

  The interface for the accessor class is vastly different depending on the
  specialization of the template arguments for access mode, access target,
  dimensions, and element type. This is implemented using enable if to enable
  only
  the constructors and operators that are relevant for a particular
  specialization
  to be available.

  On the host side the accessor class does not contain any fields and inherits
  its members from its non-templated base class accessor_base. On the device
  side the accessor class contains a single field, that being the device
  arguments, which varies depending on the specialization. On the device side it
  also inherits from the non-template base class accessor_base, however this is
  in
  order to make the declaration of the public interface functions available to
  device compiler when parsing host side only code.

  Common macros and structs are defined in accessor_args.h.

  The accessor_base class is defined in accessor_base.h.

  The subscript_op and sampler_op classes are defined in accessor_ops.h.
*/

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_H_

#include "SYCL/accessor/accessor_base.h"
#include "SYCL/accessor/accessor_ops.h"
#include "SYCL/atomic.h"
#include "SYCL/backend.h"
#include "SYCL/buffer.h"
#include "SYCL/common.h"
#include "SYCL/compat_2020.h"
#include "SYCL/id.h"
#include "SYCL/type_traits.h"
#include "SYCL/vec_swizzles_impl.h"
#include "computecpp/gsl/gsl"

namespace cl {
namespace sycl {

/** @cond COMPUTECPP_DEV */

/*******************************************************************************
    diagrams
*******************************************************************************/
/**

The host side accessor class structure is:

         ,--------------------------.
         |detail::host_arg_container|
         `--------------------------'
                     |
              ,------o------.
              |accessor_base|
              `-------------'
                     |
          ,-----------------------.
          |detail::accessor_common|
          `-----------------------'
                     |
                 ,--------.
                 |accessor|
                 `--------'

The device side accessor class structure is:

        ,------------------.  ,----------.
        |device_index_array|  |device_arg|
        `------------------'  `----------'
                |___________________|
                        |
             ,----------o---------.
             |device_arg_container|
             `--------------------'
                        |
                        |
              ,---------o----------.
              |accessor_device_base|
              `--------------------'
                        |
             ,-----------------------.
             |detail::accessor_common|
             `-----------------------'
                        |
                    ,--------.
                    |accessor|
                    `--------'

Important things to know:
 - "device_arg_container" and "host_arg_container" must be seen by the device
     compiler
   - "device_arg_container" has an attribute that refers to "host_arg_container"
 - "host_arg_container" is pulled by the stubfile
 - "device_arg_container" and "host_arg_container" content can differ
 - the accessor inheritance hierarchy must be equivalent (same tree layout)
*/

/*******************************************************************************
    ACCESSOR_ALIGNMENT macros
*******************************************************************************/

/**
@def COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT()
The COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT macro is used for specifying the
alignment of the
accessor class for Windows, there reason for two separate macros is that on
Windows the attribute is placed at the start of declaration, whereas on all
other platforms the attribute is placed at the end of the declaration.
*/
/**
@def COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT()
The COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT macro is used for specifying the
alignment of the
accessor class for non-Windows, there reason for two separate macros is that on
Windows the attribute is placed at the start of declaration, where as on all
other platforms the attribute is placed at the end of the declaration.
*/
#ifdef COMPUTECPP_WINDOWS
#define COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT                                  \
  __declspec(align(COMPUTECPP_PTR_SIZE))
#define COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT
#else
#define COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT
#define COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT                                    \
  __attribute__((aligned(COMPUTECPP_PTR_SIZE)))
#endif

/*******************************************************************************
    accessor traits
*******************************************************************************/

namespace detail {

/** @brief Calculates the index of the error code location within the buffer
 * @param size The size in bytes of the original buffer
 * @return Index of the error code location within the buffer
 */
constexpr size_t error_code_begin(const size_t size) noexcept {
  return (size / sizeof(int)) + 1;
}

/**
@brief Trait which provides the maximum of two integer values.
@tparam lhs First param.
@tparam rhs Second param.
*/
template <int lhs, int rhs>
using max_dimensions = std::integral_constant<int, (lhs < rhs ? rhs : lhs)>;

template <int dims>
using acc_interface_dims = max_dimensions<dims, 1>;

}  // namespace detail

/*******************************************************************************
    forward declarations
*******************************************************************************/

/** Public facing interface class for allowing users access to buffer objects,
 * image objects and local memory from within kernel functions and the host.
 *
 * The accessor class has many different constructors and operators
 * available depending on the class template arguments including access target,
 * access mode and dimensions. These constructors and operators are implemented
 * using an enable_if technique in order to avoid a large amount of inheritance
 * and code duplication. In order to reduce the complexity the enable_if
 * conditions are predefined using static const const booleans and the enable_if
 * definitions themselves are defined using the COMPUTECPP_ENABLE_IF macro. The
 * accessor class also has the COMPUTECPP_CONVERT_ATTR macro attached to the end
 * of the struct declaration, this is used during the compilers parameter
 * flattening mechanism. The accessor class also has the
 * COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT and COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT
 * macros which align the accessor class to the pointer size, the reason for
 * this is that for environments where the host and device pointer sizes don't
 * match the kernel argument offsets can sometimes misalign, aligning the
 * accessor resolves this.
 * @tparam elemT Underlying data type
 * @tparam kDims Number of accessor dimensions
 * @tparam kMode Access mode
 * @tparam kTarget Access target
 * @tparam isPlaceholder Whether the accessor is a placeholder
 */
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor;

/*******************************************************************************
    accessor_common
*******************************************************************************/

/** COMPUTECPP_DEV @endcond*/

namespace detail {

template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
struct opencl_backend_traits<
    sycl::accessor<elemT, kDims, kMode, kTarget, isPlaceholder>> {
 private:
  using acc_t = sycl::accessor<elemT, kDims, kMode, kTarget, isPlaceholder>;

 public:
  // input_type not defined
  using return_type =
      typename decltype(std::declval<acc_t>().get_pointer())::ptr_t;
};

// The inheritance changes slightly when on host or device.
// The device accessor_base is templated, but the host one is not.
// We declare this macro here so that the class declaration remains readable.
#ifndef __SYCL_DEVICE_ONLY__
#define COMPUTECPP_ACCESSOR_BASE_CLASS(elemT, kDims, kMode, kTarget,           \
                                       isPlaceholder)                          \
  accessor_base
#else  // !__SYCL_DEVICE_ONLY__
#define COMPUTECPP_ACCESSOR_BASE_CLASS(elemT, kDims, kMode, kTarget,           \
                                       isPlaceholder)                          \
  accessor_device_base<                                                        \
      device_arg_container<elemT, kDims, kMode, kTarget, isPlaceholder>>
#endif  // __SYCL_DEVICE_ONLY__

/**
@brief Class that implements common functions of the templated accessor class
@tparam elemT Specifies the element type of the pointer and typedef.
@tparam kDims Specifies the dimensions.
@tparam kMode Specifies the access mode.
@tparam kTarget Specifies the access target.
*/
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class COMPUTECPP_ACCESSOR_WINDOWS_ALIGNMENT accessor_common
    : public COMPUTECPP_ACCESSOR_BASE_CLASS(elemT, kDims, kMode, kTarget,
                                            isPlaceholder) {
 private:
  using base_t = COMPUTECPP_ACCESSOR_BASE_CLASS(elemT, kDims, kMode, kTarget,
                                                isPlaceholder);
#undef COMPUTECPP_ACCESSOR_BASE_CLASS

 protected:
  static constexpr int interface_dims =
      detail::acc_interface_dims<kDims>::value;
  static constexpr int range_dims =
      (kTarget == cl::sycl::access::target::image_array) ? (kDims + 1)
                                                         : interface_dims;
  static constexpr bool is_read_only = (kMode == access::mode::read);

#ifdef __SYCL_DEVICE_ONLY__
  using base_t::m_deviceArgs;
#endif

 public:
#if SYCL_LANGUAGE_VERSION >= 202001
  using value_type = std::conditional_t<is_read_only, const elemT, elemT>;
  using reference = std::conditional_t<is_read_only, const elemT&, elemT&>;
#else
  /// Alias for the value type of the accessor
  using value_type = elemT;

  /// Alias for the reference type of the accessor
  using reference = elemT&;
#endif  // SYCL_LANGUAGE_VERSION

  /// Alias for the const reference type of the accessor
  using const_reference = const elemT&;

  /**
    @brief Forwarding constructors to accessor_base
  */
  accessor_common(cl::sycl::storage_mem& store, handler& commandHandler)
      : base_t(store, kMode, kTarget, sizeof(elemT), commandHandler) {}
  accessor_common(cl::sycl::storage_mem& store, handler& commandHandler,
                  detail::access_range accessRange)
      : base_t(store, kMode, kTarget, sizeof(elemT), commandHandler,
               accessRange) {}
  accessor_common(dim_t numDims, const detail::index_array& numElements,
                  handler& commandHandler)
      : base_t(numDims, numElements, kMode, kTarget, sizeof(elemT),
               commandHandler) {}
  accessor_common(cl::sycl::storage_mem& store)
      : base_t(store, kMode, kTarget, sizeof(elemT)) {}
  accessor_common(cl::sycl::storage_mem& store,
                  detail::access_range accessRange)
      : base_t(store, kMode, kTarget, sizeof(elemT), accessRange) {}
  accessor_common() : base_t(kMode, kTarget) {}

#ifdef __SYCL_DEVICE_ONLY__
  /*Dummy declarations used by device compiler*/
  accessor_common(const accessor_common& rhs) = default;
  accessor_common(accessor_common&& rhs) = default;
  accessor_common& operator=(const accessor_common& rhs) = default;
  accessor_common& operator=(accessor_common&& rhs) = default;
#else   //__SYCL_DEVICE_ONLY__

  /**
   @brief Copy constructor that delegates construction to accessor_base.
   @param rhs the accessor to be copied.
   */
  accessor_common(const accessor_common& rhs) : accessor_base{rhs} {}

  /**
   @brief Move constructor that delegates construction to accessor_base.
   @param rhs the accessor to be moved.
   */
  accessor_common(accessor_common&& rhs) noexcept
      : accessor_base{std::move(rhs)} {}
  /**
  @brief Copy Assignment operator that delegates the assignment to the
  accessor_base.
  @param rhs the accessor to be copied.
  */
  accessor_common& operator=(const accessor_common& rhs) {
    static_cast<accessor_base*>(this)->operator=(
        static_cast<const accessor_base&>(rhs));
    return *this;
  }
  /**
  @brief Move Assignment operator that delegates the assignment to the
  accessor_base.
  @param rhs the accessor to be moved.
  */
  accessor_common& operator=(accessor_common&& rhs) noexcept {
    static_cast<accessor_base*>(this)->operator=(
        static_cast<accessor_base&&>(std::move(rhs)));
    return *this;
  }
#endif  // __SYCL_DEVICE_ONLY__

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const accessor_common& lhs,
                                const accessor_common& rhs) {
    // get_impl() is not available on the device
#ifdef __SYCL_DEVICE_ONLY__
    return (lhs.get_device_ptr() == rhs.get_device_ptr());
#else
    return ((lhs.get_impl().get() == rhs.get_impl().get()) &&
            (lhs.get_device_ptr() == rhs.get_device_ptr()));
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const accessor_common& lhs,
                                const accessor_common& rhs) {
    return !(lhs == rhs);
  }

  /**
    @brief Indicates whether the accessor is a placeholder accessor
    @return True if accessor is a placeholder
  */
  inline bool is_placeholder() const {
    return (isPlaceholder == access::placeholder::true_t);
  }

  /**
  @brief Method that returns the device argument, which can be either a pointer
  with an address space or an OpenCL image type, this is deduced by the
  device_arg struct. For the host side this returns the raw host pointer that is
  in accessor_base, for the device side this returns the device pointer that is
  in the device_arg_container.
  @return The device argument deduced by the device_arg struct.
  */
  typename device_arg<elemT, kDims, kMode, kTarget, isPlaceholder>::raw_ptr_type
  get_device_ptr() const {
#if defined(__SYCL_DEVICE_ONLY__)
    return m_deviceArgs.m_deviceArg.get_ptr();
#else
    using ptrType = typename device_arg<elemT, kDims, kMode, kTarget,
                                        isPlaceholder>::raw_ptr_type;
    return static_cast<ptrType>(base_t::get_host_data());
#endif
  }

 private:
  size_t size_impl() const {
#ifndef __SYCL_DEVICE_ONLY__
    return detail::size(static_cast<const accessor_base&>(*this));
#else
    return m_deviceArgs.m_range.size();
#endif
  }

  size_t byte_size_impl() const {
#ifndef __SYCL_DEVICE_ONLY__
    return detail::byte_size(static_cast<const accessor_base&>(*this));
#else
    return (sizeof(elemT) * m_deviceArgs.m_range.size());
#endif
  }

 public:
  /**
  @brief Gets the number of elements the accessor can access.
  @return The number of elements the accessor can access.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use accessor::size() instead.")
  size_t get_count() const { return size_impl(); }

  /**
  @brief Gets the number of elements the accessor can access.
  @return The number of elements the accessor can access.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use accessor::byte_size() instead.")
  size_t get_size() const { return byte_size_impl(); }

#if SYCL_LANGUAGE_VERSION >= 202001
  /**
   * @brief Gets the number of elements the accessor can access.
   * @return The number of elements the accessor can access.
   */
  size_t size() const noexcept { return size_impl(); }

  /**
   * @brief Gets the number of elements the accessor can access.
   * @return The number of elements the accessor can access.
   */
  size_t byte_size() const noexcept { return byte_size_impl(); }
#endif  //  SYCL_LANGUAGE_VERSION >= 202001

  /**
  @brief Gets the range of the memory the accessor can access.
  @return The the range of the memory the accessor can access.
  */
  range<range_dims> get_range() const {
#ifndef __SYCL_DEVICE_ONLY__
    return accessor_base::get_range();
#else
    range<range_dims> range;
    for (int i = 0; i < range_dims; ++i) {
      range[i] = m_deviceArgs.m_range[i];
    }
    return range;
#endif
  }

  /**
  @brief Gets the offset of the memory the accessor can access.
  @return The the offset of the memory the accessor can access.
  */
  id<range_dims> get_offset() const {
#ifndef __SYCL_DEVICE_ONLY__
    return accessor_base::get_offset();
#else
    id<range_dims> offset;
    for (int i = 0; i < range_dims; ++i) {
      offset[i] = m_deviceArgs.m_offset[i];
    }
    return offset;
#endif
  }
} COMPUTECPP_ACCESSOR_LINUX_ALIGNMENT;
}  // namespace detail

}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the buffer class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <typename elemT, int kDims, cl::sycl::access::mode kMode,
          cl::sycl::access::target kTarget,
          cl::sycl::access::placeholder isPlaceholder>
struct hash<cl::sycl::accessor<elemT, kDims, kMode, kTarget, isPlaceholder>> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::accessor<elemT, kDims, kMode, kTarget,
                                             isPlaceholder>& rhs) const {
    hash<cl::sycl::daccessor_shptr> hasher;
    return hasher(rhs.get_impl());
  }
};
}  // namespace std

/******************************************************************************/

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_H_

/******************************************************************************/
