/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file kernel.h

  @brief This file implements the kernel class as defined by the SYCL 1.2
  specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_KERNEL_H_
#define RUNTIME_INCLUDE_SYCL_KERNEL_H_

#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/common.h"
#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/include_opencl.h"
#include "SYCL/index_array.h"
#include "SYCL/info.h"
#include "SYCL/predefines.h"
#include "SYCL/range.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
#ifndef __SYCL_DEVICE_ONLY__
class accessor_base;
#else
template <typename T>
class accessor_device_base;
using accessor_base = accessor_device_base<void*>;
#endif
class handler;
class kernel;
class program;
class sampler;
namespace info {
template <typename T, T param>
struct param_traits;
}  // namespace info

namespace detail {

/** Kernel subgroup query parameter values */
enum class cl_kernel_subgroup_queries : cl_uint {
  // CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE in OpenCL 2.1
  max_sub_group_size_for_ndrange = 0x2033,
  // CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE in OpenCL 2.1
  sub_group_count_for_ndrange = 0x2034,
  // CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT in OpenCL 2.1
  local_size_for_sub_group_count = 0x11B8,
  // CL_KERNEL_MAX_NUM_SUB_GROUPS in OpenCL 2.1
  max_num_sub_groups = 0x11B9,
  // CL_KERNEL_COMPILE_NUM_SUB_GROUPS in OpenCL 2.1
  compile_num_sub_groups = 0x11BA,
  // CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL in cl_intel_required_subgroup_size
  compile_sub_group_size = 0x410A,
};
}  // namespace detail

namespace info {
/** @brief Kernel descriptor to query information about a kernel object
 */
enum class kernel : int {
  reference_count, /**< Get the reference count of the kernel object */
  num_args,        /**< Get the number of arguments taken by the kernel */
  function_name,   /**< Get the name of the kernel */
  attributes,      /**< Get kernel attributes specified by in the source file */
  context,         /**< Get the context associated with the kernel */
  program          /**< Get the program associated with the kernel */
};

enum class kernel_work_group : int {
  global_work_size,
  work_group_size,
  compile_work_group_size,
  preferred_work_group_size_multiple,
  private_mem_size
};

/** Kernel descriptors to query information about kernel sub-groups. */
enum class kernel_sub_group : int {
  /** Get the maximum number of sub-groups for a given work-group size. */
  max_sub_group_size_for_ndrange,
  /** Get the number of sub-groups for a given work-group size. */
  sub_group_count_for_ndrange,
  /** Get a work-group size that contains the given number of sub-groups. */
  local_size_for_sub_group_count,
  /** Get the maximum number of sub-groups for this kernel. */
  max_num_sub_groups,
  /** Get the number of sub-groups specified by this kernel. */
  compile_num_sub_groups,
  /** Get the required sub-group size specified by this kernel. */
  compile_sub_group_size,
};
}  // namespace info

/// @cond COMPUTECPP_DEV

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(kernel, cl_kernel_info, cl_kernel)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, reference_count,
                                      CL_KERNEL_REFERENCE_COUNT, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, num_args, CL_KERNEL_NUM_ARGS,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, function_name,
                                      CL_KERNEL_FUNCTION_NAME, string_class,
                                      char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, attributes, CL_KERNEL_ATTRIBUTES,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, context, CL_KERNEL_CONTEXT,
                                      context, cl_context)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(kernel, program, CL_KERNEL_PROGRAM,
                                      program, cl_program)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(kernel, reference_count, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(kernel, num_args, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(kernel, function_name, "")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(kernel, attributes, "")

namespace detail {
template <typename InfoEnum, InfoEnum EnumValue>
struct sycl_host_kernel_info;

template <typename InfoEnum>
COMPUTECPP_EXPORT void get_opencl_kernel_info(
    const dkernel_shptr& context, const cl_device_id device,
    const cl_kernel_work_group_info& param, size_t input_size,
    const void* input_value, const size_t output_size, void* output_buffer);

template <typename InfoEnum, InfoEnum EnumValue>
struct opencl_kernel_ext_info;

}  // namespace detail

/** @internal
 * Helper macro to set up the work-group and sub-group info query types.
 *
 * @param infoType The info descriptor enum type.
 * @param infoVal The info descriptor enum value.
 * @param clVal The OpenCL query value.
 * @param syclType The SYCL type to return from the kernel::get_*_info() query.
 * @param clType The OpenCL type returned from the clGet*Info query.
 * @param clNumElems The number of elements of type clType returned by the
 * clGet*Info query.
 */
#define COMPUTECPP_KERNEL_INFO_HELPER(infoType, infoVal, clVal, syclType,      \
                                      clType, clNumElems)                      \
  namespace detail {                                                           \
  template <>                                                                  \
  struct opencl_kernel_ext_info<infoType, infoType::infoVal> {                 \
    static constexpr cl_uint cl_param = clVal;                                 \
    using sycl_type = syclType;                                                \
    using cl_type = clType;                                                    \
    static constexpr size_t cl_type_num_elems = clNumElems;                    \
  };                                                                           \
  }

/** @internal
 * Helper macro to set up the work-group and sub-group info query types.
 * Provides the required specialisations of @c info::param_traits.
 *
 * @param infoType The info descriptor enum type.
 * @param infoVal The info descriptor enum value.
 * @param clVal The OpenCL query value.
 * @param syclType The SYCL type to return from the kernel::get_*_info() query.
 * @param clType The OpenCL type returned from the clGet*Info query.
 * @param clNumElems The number of elements of type clType returned by the
 * clGet*Info query.
 */
#define COMPUTECPP_KERNEL_INFO_PARAM(infoType, infoVal, clVal, syclType,       \
                                     clType, clNumElems)                       \
  COMPUTECPP_KERNEL_INFO_HELPER(infoType, infoVal, clVal, syclType, clType,    \
                                clNumElems)                                    \
  namespace info {                                                             \
  template <>                                                                  \
  struct param_traits<infoType, infoType::infoVal> {                           \
   public:                                                                     \
    using return_type = syclType;                                              \
  };                                                                           \
  }

/** @internal
 * Helper macro to set up the work-group and sub-group info query types for the
 * queries that take an input value.
 * Provides the required specialisations of @c info::param_traits.
 *
 * @param infoType The info descriptor enum type.
 * @param infoVal The info descriptor enum value.
 * @param clVal The OpenCL query value.
 * @param syclType The SYCL type to return from the kernel::get_*_info() query.
 * @param clType The OpenCL type returned from the clGet*Info query.
 * @param clNumElems The number of elements of type clType returned by the
 *        clGet*Info query.
 */
#define COMPUTECPP_KERNEL_INFO_PARAM_WITH_INPUT(                               \
    infoType, infoVal, clVal, syclType, clType, clNumElems, inputType)         \
  COMPUTECPP_KERNEL_INFO_HELPER(infoType, infoVal, clVal, syclType, clType,    \
                                clNumElems)                                    \
  namespace info {                                                             \
  template <>                                                                  \
  struct param_traits<infoType, infoType::infoVal> {                           \
   public:                                                                     \
    using return_type = syclType;                                              \
    using input_type = inputType;                                              \
  };                                                                           \
  }

/** @internal
 * Register a hardcoded host value to return from work-group and sub-group info
 * queries.
 *
 * @param infoType The info descriptor enum type.
 * @param infoVal The info descriptor enum value.
 * @param returnVal The hardcoded return value.
 */
#define COMPUTECPP_KERNEL_INFO_HOST_VALUE(infoType, infoVal, returnVal)        \
  namespace detail {                                                           \
  template <>                                                                  \
  struct sycl_host_kernel_info<infoType, infoType::infoVal> {                  \
    static                                                                     \
        typename info::param_traits<infoType, infoType::infoVal>::return_type  \
        get() {                                                                \
      return returnVal;                                                        \
    }                                                                          \
  };                                                                           \
  }

COMPUTECPP_KERNEL_INFO_PARAM(info::kernel_work_group, global_work_size,
                             CL_KERNEL_GLOBAL_WORK_SIZE, range<3>, size_t, 3)
COMPUTECPP_KERNEL_INFO_PARAM(info::kernel_work_group, work_group_size,
                             CL_KERNEL_WORK_GROUP_SIZE, size_t, size_t, 1)
COMPUTECPP_KERNEL_INFO_PARAM(info::kernel_work_group, compile_work_group_size,
                             CL_KERNEL_COMPILE_WORK_GROUP_SIZE, range<3>,
                             size_t, 3)
COMPUTECPP_KERNEL_INFO_PARAM(info::kernel_work_group,
                             preferred_work_group_size_multiple,
                             CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             size_t, size_t, 1)
COMPUTECPP_KERNEL_INFO_PARAM(info::kernel_work_group, private_mem_size,
                             CL_KERNEL_PRIVATE_MEM_SIZE, cl_ulong, cl_ulong, 1)

COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_work_group, global_work_size,
                                  range<3>(1, 1, 1))
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_work_group, work_group_size, 1)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_work_group,
                                  compile_work_group_size, range<3>(0, 0, 0))
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_work_group,
                                  preferred_work_group_size_multiple, 1)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_work_group, private_mem_size,
                                  8192)

COMPUTECPP_KERNEL_INFO_PARAM_WITH_INPUT(
    info::kernel_sub_group, max_sub_group_size_for_ndrange,
    static_cast<cl_uint>(
        cl_kernel_subgroup_queries::max_sub_group_size_for_ndrange),
    uint32_t, size_t, 1, range<3>)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group,
                                  max_sub_group_size_for_ndrange, 1)

COMPUTECPP_KERNEL_INFO_PARAM_WITH_INPUT(
    info::kernel_sub_group, sub_group_count_for_ndrange,
    static_cast<cl_uint>(
        cl_kernel_subgroup_queries::sub_group_count_for_ndrange),
    uint32_t, size_t, 1, range<3>)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group,
                                  sub_group_count_for_ndrange, 1)

COMPUTECPP_KERNEL_INFO_PARAM_WITH_INPUT(
    info::kernel_sub_group, local_size_for_sub_group_count,
    static_cast<cl_uint>(
        cl_kernel_subgroup_queries::local_size_for_sub_group_count),
    range<3>, size_t, 3, uint32_t)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group,
                                  local_size_for_sub_group_count,
                                  (range<3>{1, 1, 1}))

COMPUTECPP_KERNEL_INFO_PARAM(
    info::kernel_sub_group, max_num_sub_groups,
    static_cast<cl_uint>(cl_kernel_subgroup_queries::max_num_sub_groups),
    uint32_t, size_t, 1)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group, max_num_sub_groups, 1)

COMPUTECPP_KERNEL_INFO_PARAM(
    info::kernel_sub_group, compile_num_sub_groups,
    static_cast<cl_uint>(cl_kernel_subgroup_queries::compile_num_sub_groups),
    uint32_t, size_t, 1)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group,
                                  compile_num_sub_groups, 0)

COMPUTECPP_KERNEL_INFO_PARAM(
    info::kernel_sub_group, compile_sub_group_size,
    static_cast<cl_uint>(cl_kernel_subgroup_queries::compile_sub_group_size),
    size_t, size_t, 1)
COMPUTECPP_KERNEL_INFO_HOST_VALUE(info::kernel_sub_group,
                                  compile_sub_group_size, 0)

#undef COMPUTECPP_KERNEL_INFO_PARAM
#undef COMPUTECPP_KERNEL_INFO_PARAM_WITH_INPUT
#undef COMPUTECPP_KERNEL_INFO_HOST_VALUE

namespace detail {
inline std::pair<const void*, size_t> convert_sycl_to_ocl() {
  return std::pair<const void*, size_t>{nullptr, 0};
}

inline std::pair<const void*, size_t> convert_sycl_to_ocl(const size_t& input) {
  return std::pair<const void*, size_t>{&input, sizeof(size_t)};
}

template <int Dim>
inline std::pair<const void*, size_t> convert_sycl_to_ocl(
    const cl::sycl::range<Dim>& input) {
  auto indexArray = static_cast<const index_array&>(input);
  return std::pair<const void*, size_t>{indexArray.get(), sizeof(size_t) * Dim};
}

template <typename InfoEnum, InfoEnum EnumValue, typename... Args>
typename info::param_traits<InfoEnum, EnumValue>::
    return_type inline get_kernel_info_impl(const dkernel_shptr& context,
                                            const cl_device_id device,
                                            const Args&... args) {
  using cl_type = typename opencl_kernel_ext_info<InfoEnum, EnumValue>::cl_type;
  using sycl_type =
      typename opencl_kernel_ext_info<InfoEnum, EnumValue>::sycl_type;

  constexpr auto clParam =
      opencl_kernel_ext_info<InfoEnum, EnumValue>::cl_param;

  constexpr auto numOutputElems =
      opencl_kernel_ext_info<InfoEnum, EnumValue>::cl_type_num_elems;
  constexpr auto outputSize = sizeof(cl_type) * numOutputElems;

  auto oclInputPair = convert_sycl_to_ocl(args...);
  auto inputPtr = oclInputPair.first;
  auto inputSize = oclInputPair.second;

  char buffer[outputSize];
  get_opencl_kernel_info<InfoEnum>(context, device, clParam, inputSize,
                                   inputPtr, outputSize, buffer);
  cl_type* clPtr = reinterpret_cast<cl_type*>(buffer);
  return info_convert<cl_type*, sycl_type>::cl_to_sycl(clPtr, numOutputElems,
                                                       clParam);
}

template <>
struct opencl_backend_traits<sycl::kernel> {
 public:
  using input_type = cl_kernel;
  using return_type = input_type;
};

}  // namespace detail

/// COMPUTECPP_DEV @endcond

/** SYCL Kernel interface.
 *
 * See Section 3.5.4 of the Specification.
 */
class COMPUTECPP_EXPORT kernel {
  friend class handler;

  dkernel_shptr m_impl;

  // This method is available only inside command-group scope
  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
  void set_arg(int argIndex,
               accessor<dataT, dimensions, accessMode, accessTarget> accObj,
               handler& cgh) {
#ifndef __SYCL_DEVICE_ONLY__
    this->set_arg_impl(argIndex, accObj, cgh);
#else
    /* Empty for the device, different accessor implementation */
    (void)argIndex;
    (void)accObj;
    (void)cgh;
#endif
  }

  // This method is available only inside command-group scope
  template <typename T>
  void set_arg(int argIndex, T scalar_value, handler& cgh /* NOLINT */) {
    this->set_arg_impl(argIndex, &scalar_value, sizeof(T), cgh);
  }

  // This method is available only inside command-group scope
  void set_arg(int argIndex, const sampler& samplerObj,
               handler& cgh /* NOLINT */);

 protected:
  /// @cond COMPUTECPP_DEV
  /** @internal
   * Internal empty kernel
   */
  kernel();

  /** @internal
   * Internal kernel constructor
   */
  kernel(cl_kernel clKernel, dprogram_shptr program);

  /** @internal
   * set_arg method implementation
   */
  void set_arg_impl(int argIndex, accessor_base acc, handler& cgh /* NOLINT */);
  /** @internal
   * set_arg method implementation
   */
  void set_arg_impl(int argIndex, void* scalar_value, size_t size,
                    handler& cgh /* NOLINT */);

  /// COMPUTECPP_DEV @endcond

 public:
  /// @cond COMPUTECPP_DEV

  /** @internal
   */
  dkernel_shptr get_impl() const;

  /** @internal
   */
  explicit kernel(dkernel_shptr detail);

  /// COMPUTECPP_DEV @endcond

  /** @brief Copy constructor. Create a copy of a kernel.
   */
  kernel(const kernel& rhs) = default;

  /** @brief Assignment operator. Assign a copy of a kernel.
   */
  kernel& operator=(const kernel& rhs) = default;

  /** @brief Move constructor. Create a copy of a kernel.
   */
  kernel(kernel&& rhs) = default;

  /** @brief Move assignment operator. Assign a copy of a kernel.
   */
  kernel& operator=(kernel&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const kernel& lhs, const kernel& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const kernel& lhs, const kernel& rhs) {
    return !(lhs == rhs);
  }

  COMPUTECPP_TEST_VIRTUAL ~kernel() = default;

  /** Create a kernel object from cl_kernel object created by an OpenCL runtime.
   * @param clKernel an OpenCL kernel created using the OpenCL API.
   * @deprecated Provide a context as the second argument
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703,
      "Use the OpenCL interop constructor which takes a SYCL context instead.")
  kernel(cl_kernel clKernel);

  /** @brief Constructs a kernel object from an OpenCL cl_kernel object
   * @param clKernel Kernel object created by an OpenCL runtime
   * @param syclContext Context associated with the OpenCL kernel object
   */
  kernel(cl_kernel clKernel, const context& syclContext);

  /** Gets the SYCL program object this kernel is associated to.
   *
   * @return The SYCL program this kernel belongs to
   */
  COMPUTECPP_TEST_VIRTUAL program get_program() const;

  /** Gets the SYCL Context this kernel has been constructed to.
   *
   * @return The SYCL context this kernel belongs to
   */
  COMPUTECPP_TEST_VIRTUAL context get_context() const;

  /** @brief Get the underlying OpenCL kernel object.
   *
   * @return A cl_kernel object usable with the OpenCL API.
   */
  COMPUTECPP_TEST_VIRTUAL cl_kernel get() const;

  /** Query information about the kernel.
   *
   * @tparam param The kernel information descriptor
   * @return The kernel information requested with \ref param.
   */
  template <info::kernel param>
  COMPUTECPP_EXPORT
      typename info::param_traits<info::kernel, param>::return_type
      get_info() const;

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
  get_work_group_info(const cl::sycl::device& device) const {
    if (use_host_info_definitions(device.is_host())) {
      return detail::sycl_host_kernel_info<info::kernel_work_group,
                                           param>::get();
    } else {
      dkernel_shptr kernelPtr = get_impl();
      cl_device_id oclDevice = device.get();
      return detail::get_kernel_info_impl<info::kernel_work_group, param>(
          kernelPtr, oclDevice);
    }
  }
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(const cl::sycl::device& device) const {
    if (use_host_info_definitions(device.is_host())) {
      return detail::sycl_host_kernel_info<info::kernel_sub_group,
                                           param>::get();
    } else {
      dkernel_shptr kernelPtr = get_impl();
      cl_device_id oclDevice = device.get();
      return detail::get_kernel_info_impl<info::kernel_sub_group, param>(
          kernelPtr, oclDevice);
    }
  }
  template <info::kernel_sub_group param>
  typename info::param_traits<info::kernel_sub_group, param>::return_type
  get_sub_group_info(
      const cl::sycl::device& device,
      typename info::param_traits<info::kernel_sub_group, param>::input_type
          value) const {
    if (use_host_info_definitions(device.is_host())) {
      return detail::sycl_host_kernel_info<info::kernel_sub_group,
                                           param>::get();
    } else {
      dkernel_shptr kernelPtr = get_impl();
      cl_device_id oclDevice = device.get();
      return detail::get_kernel_info_impl<info::kernel_sub_group, param>(
          kernelPtr, oclDevice, value);
    }
  }

  /** @brief Returns whether the kernel was constructed from a host context
   * @return True if kernel constructed from a host context
   */
  bool is_host() const;

#if SYCL_LANGUAGE_VERSION >= 202001

  /** Returns the SYCL backend
   * @return Backend associated with the event
   */
  inline backend get_backend() const { return this->get_backend_impl(); }

#endif  // SYCL_LANGUAGE_VERSION >= 202001

 private:
  /** Returns the SYCL backend
   * @return Backend associated with the kernel
   */
  backend get_backend_impl() const;

 protected:
  /** @brief Get the underlying OpenCL kernel object without retaining it.
   * @return A cl_kernel object usable with the OpenCL API.
   */
  COMPUTECPP_TEST_VIRTUAL cl_kernel get_no_retain() const;
};

}  // namespace sycl
}  // namespace cl

namespace std {
/**
@brief provides a specialization for std::hash for the buffer class. An
std::hash<std::shared_ptr<...>> object is created and its function call
operator is used to hash the contents of the shared_ptr. The returned hash is
actually the result of (size_t) object.get_impl().get()
*/
template <>
struct hash<cl::sycl::kernel> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::kernel& object) const {
    hash<cl::sycl::dkernel_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std

#endif  // RUNTIME_INCLUDE_SYCL_KERNEL_H_

////////////////////////////////////////////////////////////////////////////////
