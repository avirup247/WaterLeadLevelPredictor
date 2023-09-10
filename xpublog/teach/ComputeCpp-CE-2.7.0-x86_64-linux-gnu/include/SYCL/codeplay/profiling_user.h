
#ifndef RUNTIME_INCLUDE_SYCL_CODEPLAY_PROFILING_H_
#define RUNTIME_INCLUDE_SYCL_CODEPLAY_PROFILING_H_

#include "computecpp_export.h"

#include <memory>

namespace cl {
namespace sycl {
namespace codeplay {
namespace profiling {

/** @brief RAII object used to create user defined profiling zones. User zones
 * are displayed in the profilers together with API calls. This allows the user
 * to add annotations to the source code
 *
 * Suppose there is a function of interest that should be displayed in the
 * profiler:
 *
 * @code
 * void function() {
 *    cl::sycl::codeplay::profiling::profiling_zone Zone{"My Function"};
 *    // Some heavy workload
 * }
 * @endcode
 *
 * This will show "My Function" as an API call in the profiler.
 * Zones can be nested, which allows complex call chains to be displayed in the
 * profiler.
 */
class profiling_zone {
 public:
  /** @brief Starts a new user-defined zone
   * @param name The name of zone
   */
  COMPUTECPP_EXPORT explicit profiling_zone(const char* name);

  /** @brief Closes a user-defined zone
   */
  COMPUTECPP_EXPORT ~profiling_zone();

 private:
  /// Reference to the implementation details
  std::unique_ptr<class profiling_zone_impl> m_pimpl;
};

}  // namespace profiling
}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_CODEPLAY_PROFILING_H_
