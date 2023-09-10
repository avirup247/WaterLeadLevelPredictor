/********************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*********************************************************************/

/** @file sampler.h
 *
 * @brief This file implements the sampler class interface as defined by the
 * SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_SAMPLER_H_
#define RUNTIME_INCLUDE_SYCL_SAMPLER_H_

#include "SYCL/base.h"
#include "SYCL/include_opencl.h"
#include "SYCL/predefines.h"
#include "SYCL/property.h"  // IWYU pragma: keep

#include <iosfwd>
#include <memory>
#include <system_error>

#include "computecpp_export.h"

/** @cond COMPUTECPP_DEV */
/**
  @def COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT()
  @brief The COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT macro is used for specifying
  the alignment of the accessor class for Windows, the reason for two separate
  macros is that on Windows the attribute is placed at the start of
  declaration,whereas on all other platforms the attribute is placed at the end
  of the declaration.
*/
/**
  @def COMPUTECPP_SAMPLER_LINUX_ALIGNMENT()
  @brief The COMPUTECPP_SAMPLER_LINUX_ALIGNMENT macro is used for specifying the
  alignment of the accessor class for non-Windows, the reason for two separate
  macros is that on Windows the attribute is placed at the start of declaration,
  whereas on all other platforms the attribute is placed at the end of the
  declaration.
*/
#ifdef COMPUTECPP_WINDOWS
#define COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT                                   \
  __declspec(align(COMPUTECPP_PTR_SIZE))
#define COMPUTECPP_SAMPLER_LINUX_ALIGNMENT
#else
#define COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT
#define COMPUTECPP_SAMPLER_LINUX_ALIGNMENT                                     \
  __attribute__((aligned(COMPUTECPP_PTR_SIZE)))
#endif
/** COMPUTECPP_DEV @endcond */

namespace cl {
namespace sycl {
class context;

/**
  @brief Enum class for specifying the addressing mode of a sampler. Values are
  hard coded to match those of OpenCL in order to allow a simple cast when
  converting.
*/
enum class addressing_mode : unsigned int {
  none = 4400,
  clamp_to_edge = 4401,
  clamp = 4402,
  repeat = 4403,
  mirrored_repeat = 4404
};

/**
  @brief Enum class for specifying the filter mode of a sampler. Values are hard
  coded to match those of OpenCL in order to allow a simple cast when
  converting.
*/
enum class filtering_mode : unsigned int { nearest = 4416, linear = 4417 };

/**
  @brief Filtering mode description
 */
enum class coordinate_normalization_mode : unsigned int {
  normalized,
  unnormalized
};

/**
@brief Public sampler class. Encapsulates an OpenCL sampler and host device
sampler.
*/
#ifndef __SYCL_DEVICE_ONLY__
class COMPUTECPP_EXPORT COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT sampler {
#else

#define COMPUTECPP_SAMPLER_MIRROR_CONVERT                                      \
  [[computecpp::opencl_mirror_convert(sampler)]]

class COMPUTECPP_EXPORT COMPUTECPP_SAMPLER_MIRROR_CONVERT sampler {

#undef COMPUTECPP_SAMPLER_MIRROR_CONVERT
#endif

 public:
  /**
   @brief Constructor that creates a sampler from the sampler addressing mode
   sampler filter mode and a boolean specifying whether normalized coordinates
   are enabled.
   @param normalizedCoords Boolean specifying whether normalized coordinates are
   enabled.
   @param addressMode The sampler addressing mode.
   @param filterMode The sampler filter mode.
   @deprecated Use sampler::sampler(coordinate_normalization_mode,
   addressing_mode, filtering_mode) instead.
   */
  COMPUTECPP_DEPRECATED_API(
      "sampler::sampler(bool, addressing_mode, filtering_mode) deprecated. Use"
      "sampler::sampler(coordinate_normalization_mode, addressing_mode,"
      "filtering_mode) instead.")
  sampler(const bool normalizedCoords, const addressing_mode addressMode,
          const filtering_mode filterMode)
      : sampler{normalizedCoords ? coordinate_normalization_mode::normalized
                                 : coordinate_normalization_mode::unnormalized,
                addressMode, filterMode} {}

  /** Constructs a sampler
   * @param normalizedCoords Whether normalized coordinates are enabled
   * @param addressMode Sampler addressing mode
   * @param filterMode Sampler filtering mode
   * @param propList Additional properties
   */
  sampler(coordinate_normalization_mode normalizedCoords,
          addressing_mode addressMode, filtering_mode filterMode,
          const property_list& propList = {});

  /**
  @brief Inter-op constructor that creates a sampler from a cl_sampler object.
  @param clSampler OpenCL cl_sampler object/
  @deprecated Please also provide a SYCL context
  */
  COMPUTECPP_DEPRECATED_API("sampler(cl_sampler) deprecated in SYCL 1.2.1, "
                            "please also provide a SYCL context")
  explicit sampler(cl_sampler clSampler);

  /** @brief Inter-op constructor that creates a sampler from a cl_sampler
   * object
   * @param clSampler OpenCL cl_sampler object
   * @param syclContext Context associated with the OpenCL sampler object
   */
  sampler(cl_sampler clSampler, const context& syclContext);

  /**
  @brief Default copy constructor.
  */
  sampler(const cl::sycl::sampler& rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    m_sampler = rhs.m_sampler;
#else
    m_impl = rhs.get_impl();
#endif
  }

  /**
  @brief Default move constructor.
  */
  sampler(cl::sycl::sampler&& rhs) noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    m_sampler = std::move(rhs.m_sampler);
#else
    m_impl = rhs.get_impl();
#endif
  }

  /**
  @brief Default destructor.
  */
  ~sampler() = default;

  /**
  @brief Default assignment operator.
  */
  sampler& operator=(const cl::sycl::sampler& rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    m_sampler = rhs.m_sampler;
#else
    m_impl = rhs.get_impl();
#endif
    return *this;
  }

  /**
  @brief Default move assignment operator.
  */
  sampler& operator=(cl::sycl::sampler&& rhs) noexcept {
#ifdef __SYCL_DEVICE_ONLY__
    m_sampler = std::move(rhs.m_sampler);
#else
    m_impl = rhs.get_impl();
#endif
    return *this;
  }

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const cl::sycl::sampler& lhs,
                                const cl::sycl::sampler& rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    (void)lhs;
    (void)rhs;
    return false;
#else
    return (lhs.get_impl() == rhs.get_impl());
#endif
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const sampler& lhs, const sampler& rhs) {
    return !(lhs == rhs);
  }

  /**
  @brief Returns true if this sampler is a host sampler
  @return True is a host sampler, false otherwise
  */
  bool is_host() const;

  /**
  @brief Get the addressing mode.
  @return Sampler addressing mode.
  */
  addressing_mode get_addressing_mode() const;

  /**
  @brief Get the filter mode.
  @return Sampler filter mode.
  */
  filtering_mode get_filtering_mode() const;

  /**
  @brief Get the coordinate normalization mode.
  @return Sampler normalization mode.
  */
  coordinate_normalization_mode get_coordinate_normalization_mode() const;

  /**
  @brief Get the sampler object for OpenCL.
  @return The OpenCL sampler that is associated with the latest context the
  sampler was used in.
  */
  cl_sampler get() const;

#ifndef __SYCL_DEVICE_ONLY__
  /**
    @brief Get implementation object.
    @return Implementation shared_ptr.
  */
  dsampler_shptr get_impl() const;
#endif

#ifdef __SYCL_DEVICE_ONLY__
  __sycl_sampler_t m_sampler;
/**
@brief Padding is added to the sampler class here to ensure that the class
is always seen as the same size by both the host and the device compilers
to prevent mismatched alignment when setting kernel arguments.
*/
#else
 private:
  dsampler_shptr m_impl;
#endif
} COMPUTECPP_SAMPLER_LINUX_ALIGNMENT COMPUTECPP_CONVERT_ATTR_SAMPLER;

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
struct hash<cl::sycl::sampler> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::sampler& object) const {
#ifndef __SYCL_DEVICE_ONLY__
    hash<cl::sycl::dsampler_shptr> hasher;
    return hasher(object.get_impl());
#else
    (void)object;
    // returned on host devices as get_impl() isn't available when compiling for
    // device, so in that case we return 0
    return 0;
#endif
  }
};
}  // namespace std
#endif  // RUNTIME_INCLUDE_SYCL_SAMPLER_H_

#undef COMPUTECPP_SAMPLER_WINDOWS_ALIGNMENT
#undef COMPUTECPP_SAMPLER_LINUX_ALIGNMENT
