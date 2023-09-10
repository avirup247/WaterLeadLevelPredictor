/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file device_info.h
  @brief This file provides various types and functions relating to
  @ref cl::sycl::device::get_info
*/
#ifndef RUNTIME_INCLUDE_SYCL_DEVICE_INFO_H_
#define RUNTIME_INCLUDE_SYCL_DEVICE_INFO_H_

#include "SYCL/assert.h"
#include "SYCL/cl_types.h"
#include "SYCL/common.h"
#include "SYCL/id.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"
#include "SYCL/predefines.h"
#include "SYCL/version.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cl {
namespace sycl {
namespace detail {

enum class cl_ext_identifier : cl_uint {
  onchip_memory = 0x1A00,
};

/**
  Values corresponding to the Intel USM extension for OpenCL
*/
enum class cl_usm_extensions_intel : cl_uint {
  host_mem_capabilities = 0x4190,
  device_mem_capabilities = 0x4191,
  single_device_shared_mem_capabilities = 0x4192,
  cross_device_shared_mem_capabilities = 0x4193,
  shared_system_mem_capabilities = 0x4194,
};

enum class cl_device_subgroup_queries : cl_uint {
  // CL_DEVICE_MAX_NUM_SUB_GROUPS in OpenCL 2.1
  max_num_sub_groups = 0x105C,
  // CL_DEVICE_SUBGROUP_INDEPENDENT_FORWARD_PROGRESS in OpenCL 2.1
  independent_forward_progress = 0x105D,
  // CL_DEVICE_SUB_GROUP_SIZES_INTEL in cl_intel_required_subgroup_size
  sub_group_sizes = 0x4108,
};

enum class cl_usm_capabilities_intel : cl_uint {
  usm_access_intel = (1 << 0),
  usm_atomic_access_intel = (1 << 1),
  usm_concurrent_access_intel = (1 << 2),
  usm_atomic_concurrent_access_intel = (1 << 3),
};
}  // namespace detail

// Forward class declaration
class platform;
class device;

namespace info {

/**
  @brief Queue properties for a device constructor in the form of a bitfield.

  Describes the command-queue properties supported by the device.

  This is a bitfield instead of an unsigned int (as per the specification)
  because it follows OpenCL, which uses a bitfield.
*/
using device_queue_properties = cl_bitfield;

/**
  @brief Properties for describing Unified Shared Memory allocations.
*/
using cl_usm_mem_properties = cl_bitfield;

/**
  @brief Enum representing values that can be queried using device::get_info.
*/
enum class device : int {
  device_type,
  vendor_id,
  max_compute_units,
  max_work_item_dimensions,
  max_work_item_sizes,
  max_work_group_size,
  preferred_vector_width_char,
  preferred_vector_width_short,
  preferred_vector_width_int,
  preferred_vector_width_long,
  preferred_vector_width_float,
  preferred_vector_width_double,
  preferred_vector_width_half,
  native_vector_width_char,
  native_vector_width_short,
  native_vector_width_int,
  native_vector_width_long,
  native_vector_width_float,
  native_vector_width_double,
  native_vector_width_half,
  max_clock_frequency,
  address_bits,
  max_mem_alloc_size,
  image_support,
  max_read_image_args,
  max_write_image_args,
  image2d_max_height,
  image2d_max_width,
  image3d_max_height,
  image3d_max_width,
  image3d_max_depth,
  image_max_buffer_size,
  image_max_array_size,
  max_samplers,
  max_parameter_size,
  mem_base_addr_align,
  half_fp_config,
  single_fp_config,
  double_fp_config,
  global_mem_cache_type,
  global_mem_cache_line_size,
  global_mem_cache_size,
  global_mem_size,
  max_constant_buffer_size,
  max_constant_args,
  local_mem_type,
  local_mem_size,
  error_correction_support,
  host_unified_memory,
  profiling_timer_resolution,
  is_endian_little,
  is_available,
  is_compiler_available,
  is_linker_available,
  execution_capabilities,
  queue_profiling,
  built_in_kernels,
  platform,
  name,
  vendor,
  driver_version,
  profile,
  version,
  opencl_c_version,
  extensions,
  printf_buffer_size,
  preferred_interop_user_sync,
  parent_device,
  partition_max_sub_devices,
  partition_properties,
  partition_affinity_domains,
  partition_type_property,
  partition_type_affinity_domain,
  reference_count,
  codeplay_onchip_memory_size,
  usm_device_allocations,
  usm_host_allocations,
  usm_shared_allocations,
  usm_restricted_shared_allocations,
  usm_system_allocator,
  max_num_sub_groups,
  sub_group_independent_forward_progress,
  sub_group_sizes,
  usm_atomic_host_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations,
};

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::device_type>() query.

  The SYCL device type.
*/
enum class device_type : unsigned int {
  cpu,
  gpu,
  accelerator,
  custom,
  automatic,
  host,
  all
};

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::partition_properties>() query.

  Partition types supported by device.
*/
enum class partition_property : int {
  no_partition,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::partition_affinity_domain>() query.

  Supported affinity domains for partitioning the device using
  info::device_affinity domain.
*/
enum class partition_affinity_domain : int {
  not_applicable,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::local_mem_type>() query.

  Type of local memory supported.
*/
enum class local_mem_type : int { none, local, global };

/* Number of elements in the fp_config enum */
static const unsigned fp_config_size = 8u;

/**
  @brief Enum representing possible values returned from different fp_config
  queries using the device::get_info call.

  Describes single precision floating-point capabilities of the device.

  Returned when using these queries:
  - half_fp_config
  - single_fp_config
  - double_fp_config
*/
enum class fp_config : int {
  denorm = 0,
  inf_nan,
  round_to_nearest,
  round_to_zero,
  round_to_inf,
  fma,
  correctly_rounded_divide_sqrt,
  soft_float
};

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::global_mem_cache_type>() query.

  Type of global memory cache supported.
*/
enum class global_mem_cache_type : int { none, read_only, read_write };

/**
  @brief Enum representing possible values returned from a
  device::get_info<info::device::execution_capabilities>() query.

  Describes the execution capabilities of the device.
*/
enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

}  // namespace info

/** @cond COMPUTECPP_DEV */

/** this declaration is requiired for when CL_DEVICE_IL_VERSION is not defined
 * in the system that the code compiles
 */
#ifndef CL_DEVICE_IL_VERSION
#define CL_DEVICE_IL_VERSION 0x105B
#endif
COMPUTECPP_DEFINE_SYCL_DETAIL_INFO_HOST(string_class, CL_DEVICE_IL_VERSION,
                                        "NO_IL")

COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(device, cl_device_info, cl_device_id)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, device_type, CL_DEVICE_TYPE,
                                      info::device_type, cl_device_type)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, vendor_id, CL_DEVICE_VENDOR_ID,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_compute_units,
                                      CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_work_item_dimensions,
                                      CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_work_item_sizes,
                                      CL_DEVICE_MAX_WORK_ITEM_SIZES, id<3>,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_work_group_size,
                                      CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_char,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_short,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_int,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_long,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_float,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_double,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_vector_width_half,
                                      CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_char,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_short,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_int,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_long,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_float,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_double,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, native_vector_width_half,
                                      CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_clock_frequency,
                                      CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, address_bits,
                                      CL_DEVICE_ADDRESS_BITS, cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_mem_alloc_size,
                                      CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image_support,
                                      CL_DEVICE_IMAGE_SUPPORT, bool, cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_read_image_args,
                                      CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_write_image_args,
                                      CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image2d_max_width,
                                      CL_DEVICE_IMAGE2D_MAX_WIDTH, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image2d_max_height,
                                      CL_DEVICE_IMAGE2D_MAX_HEIGHT, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image3d_max_width,
                                      CL_DEVICE_IMAGE3D_MAX_WIDTH, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image3d_max_height,
                                      CL_DEVICE_IMAGE3D_MAX_HEIGHT, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image3d_max_depth,
                                      CL_DEVICE_IMAGE3D_MAX_DEPTH, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image_max_buffer_size,
                                      CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, image_max_array_size,
                                      CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_samplers,
                                      CL_DEVICE_MAX_SAMPLERS, cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_parameter_size,
                                      CL_DEVICE_MAX_PARAMETER_SIZE, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, mem_base_addr_align,
                                      CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, single_fp_config,
                                      CL_DEVICE_SINGLE_FP_CONFIG,
                                      vector_class<info::fp_config>,
                                      cl_device_fp_config)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, double_fp_config,
                                      CL_DEVICE_DOUBLE_FP_CONFIG,
                                      vector_class<info::fp_config>,
                                      cl_device_fp_config)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, half_fp_config,
                                      CL_DEVICE_HALF_FP_CONFIG,
                                      vector_class<info::fp_config>,
                                      cl_device_fp_config)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, global_mem_cache_type,
                                      CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                                      info::global_mem_cache_type,
                                      cl_device_mem_cache_type)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, global_mem_cache_line_size,
                                      CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, global_mem_cache_size,
                                      CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, global_mem_size,
                                      CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_constant_buffer_size,
                                      CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                      cl_ulong, cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, max_constant_args,
                                      CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, local_mem_type,
                                      CL_DEVICE_LOCAL_MEM_TYPE,
                                      info::local_mem_type,
                                      cl_device_local_mem_type)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, local_mem_size,
                                      CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong,
                                      cl_ulong)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, error_correction_support,
                                      CL_DEVICE_ERROR_CORRECTION_SUPPORT, bool,
                                      cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, host_unified_memory,
                                      CL_DEVICE_HOST_UNIFIED_MEMORY, bool,
                                      cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, profiling_timer_resolution,
                                      CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                                      size_t, size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, is_endian_little,
                                      CL_DEVICE_ENDIAN_LITTLE, bool, cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, is_available, CL_DEVICE_AVAILABLE,
                                      bool, cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, is_compiler_available,
                                      CL_DEVICE_COMPILER_AVAILABLE, bool,
                                      cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, is_linker_available,
                                      CL_DEVICE_LINKER_AVAILABLE, bool, cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, execution_capabilities,
                                      CL_DEVICE_EXECUTION_CAPABILITIES,
                                      vector_class<info::execution_capability>,
                                      cl_device_exec_capabilities)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(device, queue_profiling,
                                                  CL_DEVICE_QUEUE_PROPERTIES,
                                                  cl_bool,
                                                  cl_command_queue_properties,
                                                  CL_QUEUE_PROFILING_ENABLE)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, built_in_kernels,
                                      CL_DEVICE_BUILT_IN_KERNELS,
                                      vector_class<string_class>, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, platform, CL_DEVICE_PLATFORM,
                                      cl::sycl::platform, cl_platform_id)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, name, CL_DEVICE_NAME,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, vendor, CL_DEVICE_VENDOR,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, driver_version, CL_DRIVER_VERSION,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, profile, CL_DEVICE_PROFILE,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, version, CL_DEVICE_VERSION,
                                      string_class, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, opencl_c_version,
                                      CL_DEVICE_OPENCL_C_VERSION, string_class,
                                      char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, extensions, CL_DEVICE_EXTENSIONS,
                                      vector_class<string_class>, char)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, printf_buffer_size,
                                      CL_DEVICE_PRINTF_BUFFER_SIZE, size_t,
                                      size_t)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, preferred_interop_user_sync,
                                      CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,
                                      bool, cl_bool)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, parent_device,
                                      CL_DEVICE_PARENT_DEVICE, cl::sycl::device,
                                      cl_device_id)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, partition_max_sub_devices,
                                      CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                                      cl_uint, cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, partition_properties,
                                      CL_DEVICE_PARTITION_PROPERTIES,
                                      vector_class<info::partition_property>,
                                      cl_device_partition_property)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(
    device, partition_affinity_domains, CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
    vector_class<info::partition_affinity_domain>, cl_device_affinity_domain)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, partition_type_property,
                                      CL_DEVICE_PARTITION_TYPE,
                                      info::partition_property,
                                      cl_device_partition_property)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, partition_type_affinity_domain,
                                      CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
                                      info::partition_affinity_domain,
                                      cl_device_affinity_domain)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(device, reference_count,
                                      CL_DEVICE_REFERENCE_COUNT, cl_uint,
                                      cl_uint)
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(
    device, codeplay_onchip_memory_size,
    static_cast<cl_uint>(detail::cl_ext_identifier::onchip_memory), cl_ulong,
    cl_ulong)

// usm_device_allocations = CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL,
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_device_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::device_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

// usm_host_allocations = CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL,
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_host_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::host_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_shared_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::single_device_shared_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_restricted_shared_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::cross_device_shared_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_system_allocator,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::shared_system_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_system_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::shared_system_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_atomic_host_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::host_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_atomic_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(
    device, usm_atomic_shared_allocations,
    static_cast<cl_uint>(
        detail::cl_usm_extensions_intel::shared_system_mem_capabilities),
    bool, info::cl_usm_mem_properties,
    static_cast<cl_bitfield>(
        detail::cl_usm_capabilities_intel::usm_atomic_access_intel))

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(
    device, max_num_sub_groups,
    static_cast<cl_uint>(
        detail::cl_device_subgroup_queries::max_num_sub_groups),
    cl_uint, cl_uint)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(
    device, sub_group_independent_forward_progress,
    static_cast<cl_uint>(
        detail::cl_device_subgroup_queries::independent_forward_progress),
    bool, cl_bool)

COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(
    device, sub_group_sizes,
    static_cast<cl_uint>(detail::cl_device_subgroup_queries::sub_group_sizes),
    vector_class<size_t>, size_t)

namespace detail {
namespace {  // NOLINT(cert-dcl59-cpp)
/** @brief Maximum size of a single allocation on a host device, in bytes.
 *
 * @note Value selected to fit into a 32-bit size_t
 */
// NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr size_t hostMemoryMaxAlloc = (1u << 31);

/** @brief Size of memory available on the host device, in bytes.
 */
// NOLINTNEXTLINE(misc-definitions-in-headers)
constexpr size_t hostMemorySize = hostMemoryMaxAlloc;

}  // namespace
}  // namespace detail

COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, device_type, info::device_type::host)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, vendor_id, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, codeplay_onchip_memory_size, (1 << 20))
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_compute_units, 1024)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_work_item_dimensions, 3)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_work_item_sizes,
                                 id<3>(4096, 4096, 4096))
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_work_group_size, 1024)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_char, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_short, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_int, 4)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_long, 2)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_float, 4)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_double, 2)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_vector_width_half, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_char, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_short, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_int, 4)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_long, 2)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_float, 4)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_double, 2)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, native_vector_width_half, 8)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_clock_frequency, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, address_bits, sizeof(size_t))
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_mem_alloc_size,
                                 detail::hostMemoryMaxAlloc)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image_support, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_read_image_args, 128)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_write_image_args, 128)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image2d_max_width, 8192)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image2d_max_height, 8192)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image3d_max_width, 4096)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image3d_max_height, 4096)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image3d_max_depth, 4096)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image_max_buffer_size,
                                 detail::hostMemorySize)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, image_max_array_size, 2048)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_samplers, 128)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_parameter_size, 1024)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, mem_base_addr_align, 1024)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, single_fp_config,
                                 vector_class<info::fp_config>{})
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, double_fp_config,
                                 vector_class<info::fp_config>{})
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, half_fp_config,
                                 vector_class<info::fp_config>{})
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, global_mem_cache_type,
                                 info::global_mem_cache_type::none)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, global_mem_cache_line_size, 64)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, global_mem_cache_size, 4096)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, global_mem_size,
                                 detail::hostMemorySize)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_constant_buffer_size,
                                 detail::hostMemorySize)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_constant_args, 128)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, local_mem_type,
                                 info::local_mem_type::global)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, local_mem_size, detail::hostMemorySize)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, error_correction_support, false)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, host_unified_memory, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, profiling_timer_resolution, 0)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, is_endian_little, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, is_available, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, is_compiler_available, false)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, is_linker_available, false)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, execution_capabilities,
                                 vector_class<info::execution_capability>())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, queue_profiling, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, built_in_kernels,
                                 vector_class<string_class>())
COMPUTECPP_DEFINE_SYCL_INFO_HOST_DECL(device, platform)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, name, "Host Device")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, vendor, "Codeplay Software Ltd.")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, driver_version, __COMPUTECPP__)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, profile, "FULL_PROFILE")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, version, "1.2.1")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, opencl_c_version, "OpenCL 1.2")
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, extensions,
                                 vector_class<string_class>{" "})
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, printf_buffer_size, 4096)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, preferred_interop_user_sync, false)
COMPUTECPP_DEFINE_SYCL_INFO_HOST_DECL(device, parent_device)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, partition_max_sub_devices, 1)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, partition_properties,
                                 vector_class<info::partition_property>())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(
    device, partition_affinity_domains,
    vector_class<info::partition_affinity_domain>())
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, partition_type_property,
                                 info::partition_property::no_partition)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(
    device, partition_type_affinity_domain,
    info::partition_affinity_domain::not_applicable)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, reference_count, 0)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, usm_device_allocations, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, usm_host_allocations, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, usm_shared_allocations, true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, usm_restricted_shared_allocations,
                                 true)
/// usm_system_allocator and usm_system_allocations share an identical
/// opencl_host_info definition.
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, usm_system_allocator, true)

COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, max_num_sub_groups, 1u)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, sub_group_independent_forward_progress,
                                 true)
COMPUTECPP_DEFINE_SYCL_INFO_HOST(device, sub_group_sizes,
                                 vector_class<size_t>{1u})

template <>
struct info_convert<cl_device_type*, info::device_type> {
  static info::device_type cl_to_sycl(cl_device_type* clValue,
                                      size_t /*numElems*/,
                                      cl_uint /*clParam*/) {
    if ((*clValue) & CL_DEVICE_TYPE_CPU) {
      return info::device_type::cpu;
    }
    if ((*clValue) & CL_DEVICE_TYPE_GPU) {
      return info::device_type::gpu;
    }
    if ((*clValue) & CL_DEVICE_TYPE_ACCELERATOR) {
      return info::device_type::accelerator;
    }
    if ((*clValue) & CL_DEVICE_TYPE_CUSTOM) {
      return info::device_type::custom;
    }
    if ((*clValue) & CL_DEVICE_TYPE_DEFAULT) {
      return info::device_type::automatic;
    }
    return info::device_type::automatic;
  }
};

template <>
struct info_convert<cl_device_local_mem_type*, info::local_mem_type> {
  static info::local_mem_type cl_to_sycl(cl_device_local_mem_type* clValue,
                                         size_t /*numElems*/,
                                         cl_uint /*clParam*/) {
    switch (*clValue) {
      case CL_GLOBAL:
        return info::local_mem_type::global;
      case CL_LOCAL:
        return info::local_mem_type::local;
      case CL_NONE:
        return info::local_mem_type::none;
      default:
        COMPUTECPP_UNREACHABLE(
            "Invalid conversion from cl_device_local_mem_type to "
            "info::local_mem_type");
    }
  }
};

template <>
struct info_convert<cl_device_partition_property*,
                    vector_class<info::partition_property>> {
  static vector_class<info::partition_property> cl_to_sycl(
      cl_device_partition_property* clValue, size_t numElems,
      cl_uint /*clParam*/) {
    vector_class<info::partition_property> syclVector;
    for (size_t i = 0; i < numElems; i++) {
      switch (clValue[i]) {
        case CL_DEVICE_PARTITION_EQUALLY:
          syclVector.push_back(info::partition_property::partition_equally);
          break;
        case CL_DEVICE_PARTITION_BY_COUNTS:
          syclVector.push_back(info::partition_property::partition_by_counts);
          break;
        case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
          syclVector.push_back(
              info::partition_property::partition_by_affinity_domain);
          break;
      }
    }
    return syclVector;
  }
};

template <>
struct info_convert<cl_device_partition_property*, info::partition_property> {
  static info::partition_property cl_to_sycl(
      cl_device_partition_property* clValue, size_t /*numElems*/,
      cl_uint /*clParam*/) {
    if (*clValue & CL_DEVICE_PARTITION_EQUALLY) {
      return info::partition_property::partition_equally;
    } else if (*clValue & CL_DEVICE_PARTITION_BY_COUNTS) {
      return info::partition_property::partition_by_counts;
    } else if (*clValue & CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN) {
      return info::partition_property::partition_by_affinity_domain;
    } else {
      return info::partition_property::no_partition;
    }
  }
};

template <>
struct info_convert<cl_device_affinity_domain*,
                    vector_class<info::partition_affinity_domain>> {
  static vector_class<info::partition_affinity_domain> cl_to_sycl(
      cl_device_affinity_domain* clValue, size_t numElems,
      cl_uint /*clParam*/) {
    vector_class<info::partition_affinity_domain> syclVector;
    for (size_t i = 0; i < numElems; i++) {
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_NUMA) {
        syclVector.push_back(info::partition_affinity_domain::numa);
      }
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE) {
        syclVector.push_back(info::partition_affinity_domain::L4_cache);
      }
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE) {
        syclVector.push_back(info::partition_affinity_domain::L3_cache);
      }
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE) {
        syclVector.push_back(info::partition_affinity_domain::L2_cache);
      }
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE) {
        syclVector.push_back(info::partition_affinity_domain::L1_cache);
      }
      if (clValue[i] & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE) {
        syclVector.push_back(
            info::partition_affinity_domain::next_partitionable);
      }
      if (syclVector.size() == 0) {
        syclVector.push_back(info::partition_affinity_domain::not_applicable);
      }
    }
    return syclVector;
  }
};

template <>
struct info_convert<cl_device_affinity_domain*,
                    info::partition_affinity_domain> {
  static info::partition_affinity_domain cl_to_sycl(
      cl_device_affinity_domain* clValue, size_t /*numElems*/,
      cl_uint /*clParam*/) {
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_NUMA) {
      return info::partition_affinity_domain::numa;
    }
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE) {
      return info::partition_affinity_domain::L4_cache;
    }
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE) {
      return info::partition_affinity_domain::L3_cache;
    }
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE) {
      return info::partition_affinity_domain::L2_cache;
    }
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE) {
      return info::partition_affinity_domain::L1_cache;
    }
    if (*clValue & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE) {
      return info::partition_affinity_domain::next_partitionable;
    }
    return info::partition_affinity_domain::not_applicable;
  }
};

template <>
struct info_convert<cl_device_fp_config*, vector_class<info::fp_config>> {
  static vector_class<info::fp_config> cl_to_sycl(cl_device_fp_config* clValue,
                                                  size_t /*numElems*/,
                                                  cl_uint /*clParam*/) {
    vector_class<info::fp_config> syclVector;
    if ((*clValue) & CL_FP_DENORM) {
      syclVector.push_back(info::fp_config::denorm);
    }
    if ((*clValue) & CL_FP_INF_NAN) {
      syclVector.push_back(info::fp_config::inf_nan);
    }
    if ((*clValue) & CL_FP_ROUND_TO_NEAREST) {
      syclVector.push_back(info::fp_config::round_to_nearest);
    }
    if ((*clValue) & CL_FP_ROUND_TO_ZERO) {
      syclVector.push_back(info::fp_config::round_to_zero);
    }
    if ((*clValue) & CL_FP_ROUND_TO_INF) {
      syclVector.push_back(info::fp_config::round_to_inf);
    }
    if ((*clValue) & CL_FP_FMA) {
      syclVector.push_back(info::fp_config::fma);
    }
    if ((*clValue) & CL_FP_SOFT_FLOAT) {
      syclVector.push_back(info::fp_config::soft_float);
    }
    if ((*clValue) & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
      syclVector.push_back(info::fp_config::correctly_rounded_divide_sqrt);
    }

    return syclVector;
  }
};

template <>
struct info_convert<cl_device_mem_cache_type*, info::global_mem_cache_type> {
  static info::global_mem_cache_type cl_to_sycl(
      cl_device_mem_cache_type* clValue, size_t /*numElems*/,
      cl_uint /*clParam*/) {
    switch (*clValue) {
      case CL_READ_ONLY_CACHE:
        return info::global_mem_cache_type::read_only;
      case CL_READ_WRITE_CACHE:
        return info::global_mem_cache_type::read_write;
      case CL_NONE:
        return info::global_mem_cache_type::none;
      default:
        COMPUTECPP_UNREACHABLE(
            "Invalid conversion from cl_device_local_mem_type to "
            "info::local_mem_type");
    }
  }
};

template <>
struct info_convert<cl_device_exec_capabilities*,
                    vector_class<info::execution_capability>> {
  static vector_class<info::execution_capability> cl_to_sycl(
      cl_device_exec_capabilities* clValue, size_t numElems,
      cl_uint /*clParam*/) {
    vector_class<info::execution_capability> syclVector;
    for (size_t i = 0; i < numElems; i++) {
      switch (clValue[i]) {
        case CL_EXEC_KERNEL:
          syclVector.push_back(info::execution_capability::exec_kernel);
          break;
        case CL_EXEC_NATIVE_KERNEL:
          syclVector.push_back(info::execution_capability::exec_native_kernel);
          break;
      }
    }
    return syclVector;
  }
};

template <>
struct info_convert<size_t*, id<3>> {
  static id<3> cl_to_sycl(size_t* clValue, size_t numElems,
                          cl_uint /*clParam*/) {
    COMPUTECPP_ASSERT(numElems == 3,
                      "Invalid conversion from size_t* to id<3>");
    (void)numElems;
    return id<3>(clValue[0], clValue[1], clValue[2]);
  }
};

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEVICE_INFO_H_
