/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file device_selector.h

  @brief This file contains the API for the @ref cl::sycl::device_selector class
*/
#ifndef RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_
#define RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_

#include "SYCL/aspect.h"
#include "SYCL/common.h"
#include "SYCL/device.h"
#include "SYCL/offline_compilation.h"
#include "SYCL/predefines.h"

#include <memory>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
namespace detail {
// Implementation class declaration.
class device_selector;
}  // namespace detail

/**
  @brief Abstract class that can be implemented to tell the runtime how to
  perform device selection.

  The function call operator is a pure virtual function that needs to be
  implemented within derived classes.
*/
class COMPUTECPP_EXPORT device_selector {
 public:
  /** @brief Constructs a device_selector.
   */
  device_selector();

  /** @brief Constructs a device_selector from another device_selector.
   */
  device_selector(const device_selector& rhs);

  /** @brief Empty destructor.
   */
  virtual ~device_selector();

  /** @brief Performs a platform and device selection and returns a pointer to
   * the resulting cl::sycl::device object.
   * @return a pointer to the cl::sycl::device object that is selected.
   */
  COMPUTECPP_TEST_VIRTUAL device select_device() const;

  /** @brief Performs the scoring of a single device, called once for every
   * device discovered. Needs to be overloaded.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  virtual int operator()(const device& device) const = 0;

 protected:
  /** @brief Evaluates devices and returns the most suitable one.
   * @return chosenDevice The device selected via evaluation.
   */
  device evaluate_devices() const;

  /* @brief Pointer to the implementation object. */
  unique_ptr_class<detail::device_selector> m_impl;
};

/** @brief Implementation of a device_selector that selects either a CPU or a
  GPU, and falls back to a host mode device if none can be found.
*/
class COMPUTECPP_EXPORT default_selector : public device_selector {
 protected:
  /**
   * @copydoc default_selector(string_class target);
   */
  explicit default_selector(const char* target);
  /**
   * @brief Constructs a default_selector
   * @param target String representing a device target
   */
  explicit default_selector(string_class target)
      : default_selector{target.c_str()} {}

 public:
  /** @brief Constructs a default_selector
   */
  default_selector() : default_selector("") {}

  /** @brief Overload that scores both CPUs and GPUs positive if they have SPIR
    support, GPUs are scored higher, scores host mode devices as positive but
    lower than a non-host device. This should never fail.
  * @param device The device that is to be scored.
  * @return an integer representing the allocated score for the device.
  */
  int operator()(const device& Device) const override;

 protected:
  /** This function sets explicitly the m_compilationInfo member and it's used
   * as a helper for unit testing
   */
  void set_offline_backend(detail::offline_backend m) { m_compilationInfo = m; }

  /** @brief Get the cached offline compilation query result
   */
  inline detail::offline_backend get_offline_backend() const noexcept {
    return m_compilationInfo;
  }

 private:
  /** @brief Caches the offline compilation result from the offline compilation
   * query
   */
  detail::offline_backend m_compilationInfo;
};

/** @brief Implementation of an opencl_selector that selects either a CPU or a
 * GPU.
 */
class COMPUTECPP_EXPORT opencl_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  opencl_selector() = default;

  /** @brief Overload that scores both CPUs and GPUs positive if they have SPIR
    support, GPUs are scored higher. Will fail if no CPU or GPU is found.
  * @param device The device that is to be scored.
  * @return An integer representing the allocated score for the device.
  */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects a CPU device.
 */
class COMPUTECPP_EXPORT cpu_selector : public device_selector {
 public:
  cpu_selector() = default;

  /** @brief Overload that scores CPUs positive if they have SPIR support. Fails
   * if a CPU cannot be found.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects a GPU device.
 */
class COMPUTECPP_EXPORT gpu_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  gpu_selector() = default;

  /** @brief Overload that scores GPUs positive if they have SPIR support. Fails
   * if a GPU cannot be found.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an accelerator
 * device
 */
class COMPUTECPP_EXPORT accelerator_selector : public device_selector {
 public:
  /** @brief Default constructor
   */
  accelerator_selector() = default;

  /** @brief Overload that scores accelerators positive
   *        if they have SPIR support.
   *        Fails if an accelerator cannot be found.
   * @param device The device that is to be scored
   * @return an integer representing the allocated score for the device
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an Intel platform.
 */
class COMPUTECPP_EXPORT intel_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  intel_selector() = default;

  /** @brief Overload that scores devices with an Intel platform positive if
  they have SPIR support.
  * @param device The device that is to be scored.
  * @return an integer representing the allocated score for the device.
  */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a device_selector that selects an AMD platform.
 */
class COMPUTECPP_EXPORT amd_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  amd_selector() = default;

  /** @brief Overload that scores devices with an AMD platform positive if they
   * have SPIR support.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @brief Implementation of a host_selector that selects the host device.
 * This selector will always return a valid host device
 */
class COMPUTECPP_EXPORT host_selector : public device_selector {
 public:
  /** @brief Default constructor.
   */
  host_selector() = default;

  /** @brief Overload that scores host mode devices positively.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** @cond COMPUTECPP_DEV */

/** @brief Implementation of a device_selector that selects an ARM platform.
 */
class COMPUTECPP_EXPORT arm_selector : public device_selector {
 public:
  /**
    @brief Default constructor.
  */
  arm_selector() = default;

  /** @brief Overload that scores devices with an ARM platform positive if they
   * have SPIR support.
   * @param device The device that is to be scored.
   * @return an integer representing the allocated score for the device.
   */
  int operator()(const device& device) const override;
};

/** COMPUTECPP_DEV @endcond */

namespace detail {

class aspect_selector_impl;

/** Represents a device selector object
 *  obtained from calling `sycl::aspect_selector`.
 */
class COMPUTECPP_EXPORT aspect_selector : public sycl::device_selector {
 public:
  /** Constructs an instance from a list of aspects
   * @param aspectList List of required aspects when selecting device
   */
  aspect_selector(const std::vector<aspect_impl>& aspectList = {});

  /** Constructs an instance from a list of aspects
   * @param aspectList List of required aspects when selecting device
   * @param denyList List of aspects to be avoided when selecting device
   */
  aspect_selector(const std::vector<aspect_impl>& aspectList,
                  const std::vector<aspect_impl>& denyList);

  /** Evaluates a device
   * @param dev SYCL device to provide a score for
   * @return Device score.
   *         Negative if the device doesn't have all the required aspects.
   */
  int operator()(const sycl::device& dev) const final;

 private:
  /// Implementation object
  std::shared_ptr<aspect_selector_impl> m_impl;
};

}  // namespace detail

#if SYCL_LANGUAGE_VERSION >= 202002

/** Construct a device selector object based on the aspects provided in the list
 * @param aspectList List of required aspects when selecting device
 * @return Device selector object
 */
inline auto aspect_selector(const std::vector<aspect>& aspectList) {
  return detail::aspect_selector(aspectList);
}

/** Construct a device selector object based on the aspects provided in the list
 * @param aspectList List of required aspects when selecting device
 * @param denyList List of aspects to avoid when selecting device
 * @return Device selector object
 */
inline auto aspect_selector(const std::vector<aspect>& aspectList,
                            const std::vector<aspect>& denyList) {
  return detail::aspect_selector(aspectList, denyList);
}

/** Construct a device selector object based on the aspects provided
 * @tparam aspectListTN Types of aspects passed in.
 *         Must all be of type sycl::aspect.
 *         Expected to be deduced instead of explicitly provided.
 * @param aspectList Required aspects when selecting device
 * @return Device selector object
 */
template <class... aspectListTN>
auto aspect_selector(aspectListTN... aspectList) {
  return aspect_selector(std::vector<aspect>{aspectList...});
}

/** Construct a device selector object based on the aspects provided
 *  as template parameters.
 * @tparam aspectListN Required aspects when selecting device
 * @return Device selector object
 */
template <aspect... aspectListN>
auto aspect_selector() {
  return aspect_selector(std::vector<aspect>{aspectListN...});
}

#endif  // SYCL_LANGUAGE_VERSION >= 202002

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_DEVICE_SELECTOR_H_
