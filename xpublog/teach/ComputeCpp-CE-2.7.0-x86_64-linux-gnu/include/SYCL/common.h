/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp
*******************************************************************************/

/**
  @file common.h

  @brief File containing internal declarations relating to the implementation
*/

#ifndef RUNTIME_INCLUDE_SYCL_COMMON_H_
#define RUNTIME_INCLUDE_SYCL_COMMON_H_

// The predefines header needs to be the very first header included
#include "SYCL/predefines.h"

#include "SYCL/host_compiler_macros.h"
#include "SYCL/include_opencl.h"

COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wreserved-id-macro")
#define _SCL_SECURE_NO_WARNINGS 1
COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)

#include <algorithm>
#include <array>
#include <bitset>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace cl {
namespace sycl {

class device;

/** @cond COMPUTECPP_DEV */

/*
 * @brief typedef of unsigned short for use as number of dimensions.
 */
using dim_t = unsigned short;

/**
  @brief Enum class for specifying the type of a sycl_log object
*/
enum class log_type {
  none,            ///< default
  error,           ///< synchronous runtime error - results in an exception
  callback_error,  ///< asynchronous runtime error - results in an exception
                   ///  thrown to the async_handler
  warning,         ///< runtime warning - results in a warning written to
                   ///  standard output
  info,            ///< runtime information - results in a log being written to
                   ///  standard output
  assert,          ///< runtime assertion based on condition - results in an
                   ///  error being written to standard output
  unreachable,     ///< runtime unreachable for a code path that should not be
                   ///  reached - results in an error being written to standard
                   ///  output
  not_implemented  ///< not implemented feature - results in a not implemented
                   ///  feature specific error being written to standard output
};
/** COMPUTECPP_DEV @endcond */

namespace access {

/**
  @brief Enum class for specifying the access mode for the accessor.
*/
enum class mode : unsigned int {
  read = 0,                ///< read-only access
  write = 1,               ///< write-only access, previous contents not
                           ///  discarded
  read_write = 2,          ///< read and write access
  discard_write = 3,       ///< write-only access, previous contents discarded
  discard_read_write = 4,  ///< read and write access, previous contents
                           ///  discarded
  atomic = 5               ///< atomic access
};

/**
  @brief Enum class for specifying the access target for the accessor.
*/
enum class target : unsigned int {
  host_buffer = 0,      ///< Access a buffer immediately in host code
  global_buffer = 1,    ///< Access buffer via global memory
  constant_buffer = 2,  ///< Access buffer via constant memory
  local = 3,            ///< Access work-group-local memory
  host_image = 4,       ///< Access an image immediately in host code
  image = 5,            ///< Access an image
  image_array = 6,      ///< Access an image array
  subgroup_local = 9,   ///< Access buffer via subgroup local memory (extension)
};

}  // namespace access

#if SYCL_LANGUAGE_VERSION >= 202001
using access_mode = access::mode;
#endif  // SYCL_LANGUAGE_VERSION

namespace access {
#if SYCL_LANGUAGE_VERSION >= 202001
using sycl::access_mode;
#endif  // SYCL_LANGUAGE_VERSION

/**
  @brief Enum class for specifying whether the accessor is a placeholder
*/
enum class placeholder {
  false_t,  ///< Normal accessor
  true_t    ///< Placeholder accessor
};
}  // namespace access

#if SYCL_LANGUAGE_VERSION >= 202001

/** Used to help deduce the access mode
 * @tparam access_mode Access mode
 */
template <access_mode>
struct mode_tag_t {
  explicit mode_tag_t() = default;
};

/// Tag that helps with deducing a read-only access mode
inline constexpr mode_tag_t<access_mode::read> read_only{};

/// Tag that helps with deducing a read-write access mode
inline constexpr mode_tag_t<access_mode::read_write> read_write{};

/// Tag that helps with deducing a write access mode
inline constexpr mode_tag_t<access_mode::write> write_only{};

/** Used to help deduce the access mode and target
 * @tparam access_mode Access mode
 * @tparam access::target Access target
 */
template <access_mode, access::target>
struct mode_target_tag_t {
  explicit mode_target_tag_t() = default;
};

/// Tag that helps with deducing the access to a constant buffer
inline constexpr mode_target_tag_t<access_mode::read,
                                   access::target::constant_buffer>
    read_constant{};

namespace detail {

template <class dataT>
using default_access_mode =
    std::integral_constant<access_mode,
                           (std::is_const_v<dataT> ? access_mode::read
                                                   : access_mode::read_write)>;

template <class dataT>
inline constexpr auto default_access_mode_v = default_access_mode<dataT>::value;

}  // namespace detail

#endif  // SYCL_LANGUAGE_VERSION

namespace detail {

// Alias for planes
using plane_id_t = std::int8_t;

enum enum_access_mode {
  ACCESS_MODE_NONE,
  ACCESS_MODE_READ,
  ACCESS_MODE_WRITE,
  ACCESS_MODE_READ_WRITE,
  ACCESS_MODE_DISCARD_WRITE,
  ACCESS_MODE_DISCARD_READ_WRITE
};

enum enum_access_location {
  ACCESS_LOCATION_NONE,
  ACCESS_LOCATION_HOST,
  ACCESS_LOCATION_DEVICE
};

enum enum_access_type {
  ACCESS_TYPE_NONE,
  ACCESS_TYPE_BUFFER,
  ACCESS_TYPE_IMAGE,
  ACCESS_TYPE_LOCAL,
  ACCESS_TYPE_CLBUFFER,
  ACCESS_TYPE_CLIMAGE,
  ACCESS_TYPE_PLANE,
};

enum enum_access_address_space {
  ACCESS_ADDRESS_SPACE_NONE,
  ACCESS_ADDRESS_SPACE_NA,
  ACCESS_ADDRESS_SPACE_GLOBAL,
  ACCESS_ADDRESS_SPACE_CONSTANT,
  ACCESS_ADDRESS_SPACE_LOCAL
};

/** The source of the initial data for buffers or images, if any.
 */
enum enum_data_source {
  NO_DATA_SOURCE,
  DATA_SOURCE_HOST,
  DATA_SOURCE_MEM_OBJECT,
  DATA_SOURCE_GL_OBJECT,
  DATA_SOURCE_DEVICE,
  DATA_SOURCE_BUFFER  // Specific case for sub-buffers
};

/** @brief Indicates the type of the pointer the user passed to the buffer/image
 */
enum class pointer_origin {
  none,       ///< No user provided pointer
  raw,        ///< Non-const raw pointer
  raw_const,  ///< Raw pointer-to-const
  shared,     ///< Shared pointer
};

struct NullDeleter {
  void operator()(void*) {}
};
/**
The enum_accessor_type enumeration is used to specify the type of an accessor
object; this type is stored
within the shared base type of both accessor and host_accessor; accessor_impl.
ACCESSOR_TYPE_HOST refers
to a host_accessor. ACCESSOR_TYPE_DEVICE refers to an accessor.
*/
enum enum_accessor_type { ACCESSOR_TYPE_HOST, ACCESSOR_TYPE_DEVICE };

}  // namespace detail

/*******************************************************************************
    STL definitions
*******************************************************************************/

template <typename T, typename Alloc = std::allocator<T>>
using vector_class = std::vector<T, Alloc>;

using string_class = std::string;

template <typename T>
using function_class = std::function<T>;

using mutex_class = std::mutex;

template <typename T, class D = std::default_delete<T>>
using unique_ptr_class = std::unique_ptr<T, D>;

template <typename T>
using shared_ptr_class = std::shared_ptr<T>;

template <typename T>
using weak_ptr_class = std::weak_ptr<T>;

template <typename T>
using hash_class = std::hash<T>;

template <size_t Size>
using bitset_class = std::bitset<Size>;

/*******************************************************************************
    Byte type alias
*******************************************************************************/

using byte = unsigned char;

/*******************************************************************************
    accessor forward declaration
*******************************************************************************/

#if SYCL_LANGUAGE_VERSION >= 202001

template <typename dataT, int kDims = 1,
          access_mode kMode = detail::default_access_mode_v<dataT>,
          access::target kTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor;

template <typename elemT, int kDims = 1,
          access_mode kMode = detail::default_access_mode_v<elemT>>
class host_accessor;

#else

template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor;

#endif  // SYCL_LANGUAGE_VERSION >= 202001

namespace detail {
/**
  @brief Forward declaration of accessor_common.
*/
template <typename elemT, int kDims, access::mode kMode, access::target kTarget,
          access::placeholder isPlaceholder>
class accessor_common;
}  // namespace detail

/*******************************************************************************
    device_arg_info
*******************************************************************************/

namespace detail {

/**
@brief Specialized template struct that contains a typedef for the element of
device_index_array based on the address bits of the compiler. This struct is
specialized for different values for kPtrSize.
@tparam kPtrSize Specifies the pointer size in bytes.
*/
template <size_t kPtrSize>
struct device_arg_info {
  static_assert(kPtrSize, "Arch size not supported.");
};

template <>
struct device_arg_info<4> {
  using elem_type = int32_t;
};

template <>
struct device_arg_info<8> {
  using elem_type = int32_t;
};

/*******************************************************************************
    binary_info
*******************************************************************************/

/** @brief Alias for the address of a binary data.
 */
using binary_address = const unsigned char*;

/** @brief The binary_info struct is used to contain all of the meta data
 * associated with a particular ComputeCpp module.
 */
struct kernel_binary_info {
  /// Target for which the module blob was compiled for
  const char* const target;
  /// Architecture size for which the module blob was compiled for
  const size_t device_address_bits;
  /// Module blob data
  binary_address const data;
  /// Module blob size
  const size_t data_size;
  /// Extensions used by the module
  const char* const* const used_extensions;
};

/** Constructs empty binary info used by the host
 * @return Empty kernel binary info
 */
constexpr kernel_binary_info make_host_binary_info() {
  return {"", 0, nullptr, 0, nullptr};
}

/*******************************************************************************
    kernel_info
*******************************************************************************/

/** @brief The field_descriptor struct is used to store information on a
 * SYCL kernel functor field.
 */
struct field_descriptor {
  friend inline bool operator==(const field_descriptor& lhs,
                                const field_descriptor& rhs) {
    return lhs.size == rhs.size && lhs.offset == rhs.offset &&
           lhs.paramClass == rhs.paramClass;
  }

  friend inline bool operator!=(const field_descriptor& lhs,
                                const field_descriptor& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Size of the field.
   */
  size_t size;

  /** @brief Offset of the field in the functor.
   */
  size_t offset;

  /** @brief Class of the field.
   */
  parameter_class paramClass;

  /** @brief Offset into the array of argument descriptors to find the ones
   * corresponding to this field.
   */
  size_t argDescOffset;
};

/** @brief A "tuple" of an associated binary_info structure and arguments to
 * call a kernel with.
 *
 * An array of these entries is located on kernel_info, and describes all the
 * binary infos that that kernel is included in, as well as what arguments
 * should be set or unset.
 *
 * The list of arguments maps 1:1 to the arguments in the kernel information's
 * `arg_desc` structure.
 *
 * @tparam NArgs the number of arguments for this function
 */
template <unsigned NArgs>
struct kernel_definition {
  const kernel_binary_info* binary_info;
  std::array<bool, NArgs> arguments;
};

// NOLINTNEXTLINE(cert-dcl59-cpp)
namespace {

/** @brief The kernel_info struct is used to contain all of the meta data
 * associated with a particular SPIR kernel.
 *
 * The class is templated on the name of the
 * kernel. This will typically match the name associated in the kernel_lambda
 * function. The stub file generated by the device compiler specializes the
 * kernel_info struct by the kernel name to contain the relevant information
 * for a specific kernel function.
 */
template <typename T>
struct kernel_info {
  using functor_fields = const std::array<field_descriptor, 0>;
  using arg_descriptor_list = const std::array<parameter_kind, 0>;

  /// OpenCL kernel name.
  static const char* name;
  /// Binaries implementing the kernel.
  static const kernel_definition<0> bin_info[];
  /// Number of binaries implementing the kernel.
  static const size_t bin_count;
  /// SYCL kernel functor fields descriptors.
  static const functor_fields fields;
  /// Kernel argument descriptions
  static const arg_descriptor_list arg_desc;
};

template <typename T>
const char* kernel_info<T>::name = nullptr;

template <typename T>
const size_t kernel_info<T>::bin_count = 0;

template <typename T>
const kernel_definition<0> kernel_info<T>::bin_info[] = {{nullptr, {{}}}};

template <typename T>
const typename kernel_info<T>::functor_fields kernel_info<T>::fields = {{}};

template <typename T>
const typename kernel_info<T>::arg_descriptor_list kernel_info<T>::arg_desc = {
    {}};

}  // namespace

struct functor_arg_descriptor {
  template <typename T>
  functor_arg_descriptor(const kernel_info<T>&, const bool* used_,
                         const size_t nargs)
      : fields(kernel_info<T>::fields.data()),
        fields_size(kernel_info<T>::fields.size()),
        args(kernel_info<T>::arg_desc.data()),
        used{used_},
        args_size(nargs) {}

  const field_descriptor* fields;
  size_t fields_size;
  const parameter_kind* args;
  const bool* used;
  size_t args_size;
};

/*******************************************************************************
    global index linearization function
*******************************************************************************/

/**
@brief Global function for calculating a row major linearized index from an id
and range.
@param index0 Element 0 of the index.
@param index1 Element 1 of the index.
@param index2 Element 2 of the index.
@param range1 Element 1 of the range.
@param range2 Element 2 of the range.
@return The linearized index.
*/
template <size_t Dim = 3>
constexpr size_t construct_linear_row_major_index(size_t index0, size_t index1,
                                                  size_t index2, size_t,
                                                  size_t range1,
                                                  size_t range2) {
  return index2 + (index1 * range2) + (index0 * range1 * range2);
}

/**
@brief Global function for calculating a row major linearized index from an id
and range. This is an optimized specialization for 2D arguments.
@param index0 Element 0 of the index.
@param index1 Element 1 of the index.
@param range1 Element 1 of the range.
@return The linearized index.
*/
template <>
constexpr size_t construct_linear_row_major_index<2>(size_t index0,
                                                     size_t index1, size_t,
                                                     size_t, size_t range1,
                                                     size_t) {
  return index1 + (index0 * range1);
}

/**
@brief Global function for calculating a row major linearized index from an id
and range. This is an optimized specialization for 1D arguments.
@param index0 Element 0 of the index.
@return The linearized index.
*/
template <>
constexpr size_t construct_linear_row_major_index<1>(size_t index0, size_t,
                                                     size_t, size_t, size_t,
                                                     size_t) {
  return index0;
}

/*******************************************************************************
    Helper functions for the copy API methods
*******************************************************************************/

/**
  @brief Helper struct that checks whether an access mode includes read access
  @tparam accessMode Access mode to check
*/
template <cl::sycl::access::mode accessMode>
struct is_read_mode {
  /**
    @brief True if access mode includes read access
  */
  static constexpr bool value =
      ((accessMode == cl::sycl::access::mode::read) ||
       (accessMode == cl::sycl::access::mode::read_write) ||
       (accessMode == cl::sycl::access::mode::discard_read_write));
};

/**
  @brief Helper struct that checks whether an access mode includes write access
  @tparam accessMode Access mode to check
*/
template <cl::sycl::access::mode accessMode>
struct is_write_mode {
  /**
    @brief True if access mode includes write access
  */
  static constexpr bool value =
      ((accessMode == cl::sycl::access::mode::write) ||
       (accessMode == cl::sycl::access::mode::read_write) ||
       (accessMode == cl::sycl::access::mode::discard_write) ||
       (accessMode == cl::sycl::access::mode::discard_read_write));
};

/**
  @brief Helper struct that checks whether data of two different underlying
         types can be copied from origin to destination
  @tparam TOrig Underlying type of the origin data
  @tparam TDest Underlying type of the destination data
*/
template <typename TOrig, typename TDest>
struct can_copy_types {
  /**
    @brief True if types are the same,
           or if origin type is a const version of destination type,
           or if one type is void,
           or if origin is const void
  */
  static constexpr bool value = ((std::is_same<TOrig, TDest>::value) ||
                                 (std::is_same<TOrig, const TDest>::value) ||
                                 (std::is_same<TOrig, void>::value) ||
                                 (std::is_same<TDest, void>::value) ||
                                 (std::is_same<TOrig, const void>::value));
};

/// Wrapper for user callable device selectors
using device_selector_wrapper = std::function<int(const sycl::device& dev)>;

/** Used to differentiate detail constructors from public ones
 */
struct impl_constructor_tag {};

}  // namespace detail
}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTECPP_HARNESS_DEVICE
// This header has to be included before all the other headers
#include "SYCL/base.h"
#endif  // COMPUTECPP_HARNESS_DEVICE

#endif  // RUNTIME_INCLUDE_SYCL_COMMON_H_

////////////////////////////////////////////////////////////////////////////////
