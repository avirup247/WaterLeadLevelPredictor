/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file buffer.h

  @brief This file contains the @ref cl::sycl::buffer class API
*/

#ifndef RUNTIME_INCLUDE_SYCL_BUFFER_H_
#define RUNTIME_INCLUDE_SYCL_BUFFER_H_

#include "SYCL/allocator.h"
#include "SYCL/backend.h"
#include "SYCL/base.h"
#include "SYCL/common.h"
#include "SYCL/compat_2020.h"
#include "SYCL/context.h"
#include "SYCL/error_log.h"
#include "SYCL/event.h"
#include "SYCL/id.h"
#include "SYCL/include_opencl.h"
#include "SYCL/index_array.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/predefines.h"
#include "SYCL/property.h"  // IWYU pragma: keep
#include "SYCL/range.h"
#include "SYCL/storage_mem.h"

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {
template <typename T, int dimensions = 1,
          typename AllocatorT = buffer_allocator>
class buffer;

class handler;
class queue;

#if SYCL_LANGUAGE_VERSION >= 202001
template <typename elemT, int kDims, access::access_mode kMode>
class host_accessor;
#endif  // SYCL_LANGUAGE_VERSION

namespace property {
namespace buffer {

/** @brief The use_host_ptr property adds the requirement that the SYCL runtime
 *        must not allocate any memory for the SYCL buffer and instead uses the
 *        provided host pointer directly.
 **/
class COMPUTECPP_EXPORT use_host_ptr : public detail::property_base {
 public:
  use_host_ptr() : detail::property_base(detail::property_enum::use_host_ptr) {}
};

/** @brief The use_mutex property adds the requirement that the memory which is
 *        owned by the SYCL buffer can be shared with the application via a
 *        mutex_class provided to the property. The mutex is locked by the
 *        runtime whenever the data is in use and unlocked otherwise. Data is
 *        synchronized with host data when the mutex is unlocked by the runtime.
 **/
class COMPUTECPP_EXPORT use_mutex : public detail::property_base {
 public:
  /** @brief Constructs a SYCL use_mutex property instance with a reference to
   *        mutexRef parameter provided.
   * @param mutexRef Mutex to be associated with the property and the buffer.
   */
  use_mutex(mutex_class& mutexRef)
      : detail::property_base(detail::property_enum::use_mutex),
        m_mutexRef(&mutexRef) {}

  /** @brief Retrieve the mutex provided on construction
   * @return Mutex associated with this property
   */
  inline mutex_class* get_mutex_ptr() const { return m_mutexRef; }

 private:
  /** @brief Store a pointer to the mutex provided by the user
   */
  mutex_class* m_mutexRef;
};

/** @brief The context_bound property adds the requirement that the SYCL buffer
 *        can only be associated with a single SYCL context that is provided to
 *        the property.
 */
class COMPUTECPP_EXPORT context_bound : public detail::property_base {
 public:
  /** @brief Constructs a SYCL context_bound property instance with a copy of a
   *        SYCL context.
   * @param boundContext Context to be bound to the buffer.
   */
  context_bound(const context& boundContext)
      : detail::property_base(detail::property_enum::context_bound),
        m_boundContext(boundContext.get_impl()) {}

  /** @brief Retrieves the context provided on construction
   * @return The context bound to the buffer
   */
  inline context get_context() const { return context(m_boundContext); }

 private:
  /** @brief Store the context provided by the user
   */
  dcontext_shptr m_boundContext;
};

}  // namespace buffer
}  // namespace property

namespace detail {
template <typename T, int dimensions, typename AllocatorT>
struct opencl_backend_traits<sycl::buffer<T, dimensions, AllocatorT>> {
 public:
  using input_type = cl_mem;
  using return_type = std::vector<cl_mem>;
};
}  // namespace detail

/** @cond COMPUTECPP_DEV */

/** Class.
 *  buffer_mem hides the implementation details using a PIMPL pattern,
 *  Redirects all methods to the implementation, and only requires the
 *  declaration of detail::buffer to exist, but not the full definition.
 *  Methods defined here will be available and visible to the user.
 */
class COMPUTECPP_EXPORT buffer_mem : public storage_mem {
 protected:
  explicit buffer_mem(std::string errorMessage) : storage_mem() {
    COMPUTECPP_CL_ERROR_CODE_MSG(CL_SUCCESS,
                                 detail::cpp_error_code::NOT_SUPPORTED_ERROR,
                                 nullptr, errorMessage);
    (void)errorMessage;
  }

 public:
  /** @brief Constructs an storage mem object for a buffer from the
   * given host pointer, passed as a shared_pointer Rvalue.
   *
   * @param shared_ptr_class<void> Shared pointer containing the host pointer
   * @param short  Number of dimensions (1..3)
   * @param index_array Range/shape of the buffer
   * @param size_t Size of each element of the buffer in bytes
   * @param unique_ptr<detail::base_allocator> Allocator provided by the user
   * @param propList List of buffer properties
   */
  buffer_mem(shared_ptr_class<void>&& hostPointer, dim_t numDims,
             const detail::index_array& rI, size_t elementSize,
             detail::pointer_origin pointerOrigin,
             unique_ptr_class<detail::base_allocator>&& bA,
             const property_list& propList);

  /** @brief Constructs a zero-sized storage mem object without doing the check
   * for zero-size.
   *
   * @param numDims Number of dimensions (1..3)
   * @param rI The range of the buffer (expected to be zero)
   * @param elementSize Size of each element of the buffer in bytes
   * @param bA User-provided allocator
   */
  buffer_mem(dim_t numDims, const detail::index_array& rI, size_t elementSize,
             unique_ptr_class<detail::base_allocator>&& bA);

  /** @brief Constructs a sub-buffer from the given buffer.
   *
   * @param buffer_mem The parent buffer of this sub-buffer
   * @param index_array The base offset to start the sub-buffer
   * @param subRange  The range/shape of the sub-buffer
   */
  buffer_mem(buffer_mem& parentBuf, const detail::index_array& baseIndex,
             const detail::index_array& subRange);

  /** @brief Constructs an interop buffer from an existing OpenCL mem object.
   *
   * @param memObject OpenCL memory object
   * @param fromQueue Queue where the cl_mem object is used
   * @param syclEvent Event that, if available, the runtime has to wait before
   *        using the cl_mem object
   * @param numDims Number of dimensions
   * @param elementSize Size of each element
   * @param bA Allocator provided by the user
   */
  COMPUTECPP_DEPRECATED_API(
      "This constructor is deprecated in SYCL 1.2.1, please use the one "
      "accepting a SYCL Context instead.")
  buffer_mem(cl_mem memObject, queue& fromQueue, event syclEvent, dim_t numDims,
             size_t elementSize, unique_ptr_class<detail::base_allocator>&& bA);

  /** @brief Constructs an interop buffer from an existing OpenCL mem object.
   *
   * @param memObject OpenCL memory object
   * @param syclContext context where the cl_mem object is used
   * @param syclEvent Event that, if available, the runtime has to wait before
   *        using the cl_mem object
   * @param numDims Number of dimensions
   * @param elementSize Size of each element
   * @param bA Allocator provided by the user
   */
  buffer_mem(cl_mem memObject, const context& syclContext, event syclEvent,
             dim_t numDims, size_t elementSize,
             unique_ptr_class<detail::base_allocator>&& bA);

 private:
  /** @brief Constructs an interop buffer from an existing OpenCL mem object,
   *        for an internal queue object.
   *
   * @param memObject OpenCL memory object
   * @param syclInternalQ Internal queue where the cl_mem object is used
   * @param syclEvent Event that, if available, the runtime has to wait
   *        before using the cl_mem object
   * @param numDims Number of dimensions
   * @param elementSize Size of each element
   * @param bA Allocator provided by the user
   */
  buffer_mem(cl_mem memObject, const dqueue_shptr& syclInternalQ,
             event syclEvent, dim_t numDims, size_t elementSize,
             unique_ptr_class<detail::base_allocator>&& bA);

 public:
  /** @brief Default destructor
   */
  ~buffer_mem() override = default;

  /** @brief Internal constructor used for testing
   */
  explicit buffer_mem(dmem_shptr impl);

  /** @brief Default copy constructor.
   */
  buffer_mem(const buffer_mem& rhs) = default;

  /** @brief Default move constructor.
   * @param rhs will be destroyed after the operation.
   */
  buffer_mem(buffer_mem&& rhs) = default;

  /** @brief Copy assignment.
   * @param rhs the buffer_mem to be copied.
   */
  buffer_mem& operator=(const buffer_mem& rhs) = default;

  /** @brief Move assignment operator.
   * @param rhs the buffer_mem to have its contents moved. The object will be
   * destroyed after the operation.
   */
  buffer_mem& operator=(buffer_mem&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const buffer_mem& lhs, const buffer_mem& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const buffer_mem& lhs, const buffer_mem& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Returns whether or not the buffer has any storage
   * @return True when the buffer has storage attached
   */
  bool has_storage() const;

  /** @brief bool conversion calls has_storage(). Explicit still allows for use
   * in `if (buffer) { // use buffer here }` code.
   * @return True when the buffer has storage attached
   */
  explicit operator bool() const { return has_storage(); }

  /** @brief True if the buffer is a sub-buffer
   */
  bool is_sub_buffer() const;

 protected:
  /** Constructs a reinterpreted buffer. Checks whether the ranges match.
   * @param reinterpretElementSize the size of an individual element in bytes
   * @param reinterpretDims the dimensions count of the reinterpreted buffer.
   * @param reinterpretRange the range that the new buffer will use.
   * @return Reinterpreted buffer with the requested parameters
   * @throw invalid_object_error if the size of the reinterpreted buffer
   *        and the size of the original buffer do not match
   */
  dmem_shptr reinterpret_buffer(
      size_t reinterpretElementSize, size_t reinterpretDims,
      const detail::index_array& reinterpretRange) const;

  /** Calculates the reinterpreted range
   * @tparam ReinterpretT New underlying type
   * @tparam ReinterpretDim New buffer dimension
   * @return Determined range of the reinterpreted buffer
   */
  template <class ReinterpretT, int ReinterpretDim>
  detail::index_array get_reinterpret_range() const noexcept {
    if (ReinterpretDim == 1) {
      return {detail::byte_size(*this) / sizeof(ReinterpretT), 1, 1};
    }
    return this->get_range_impl();
  }
};
/** @endcond */

/**
@brief buffer is the public interface for the buffer object implementation. The
template allows the creation of specific types and number of dimensions.
 */
template <typename T, int dims, typename AllocatorT>
class buffer : public buffer_mem {
 public:
  /** @brief Helper for the user, alias for type T */
  using value_type = T;

  /** @brief Helper for the user, alias for reference to type T */
  using reference = T&;

  /** @brief Helper for the user, alias for const reference to type T */
  using const_reference = const T&;

  /** @brief Helper for the user, alias for the type of the allocator */
  using allocator_type = AllocatorT;

  /// Number of buffer dimensions (Codeplay extension)
  static constexpr const auto dimensions = dims;

  /**
  @brief Constructs a buffer without a host pointer.
  The runtime will only use internally allocated memory and
  no copy in or out is defined.
  The given allocator is used to create internal storage in case the
  runtime requires it.
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(const range<dims>& r, const property_list& propList = {})
      : buffer_mem(
            shared_ptr_class<void>(nullptr, detail::NullDeleter()), dims, r,
            sizeof(T), detail::pointer_origin::none,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer without a host pointer.
  The runtime will only use internally allocated memory and
  no copy in or out is defined.
  The given allocator is used to create internal storage in case the
  runtime requires it.
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage in case the
             runtime requires it.
  @param propList List of buffer properties
  */
  buffer(const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            shared_ptr_class<void>(nullptr, detail::NullDeleter()), dims, r,
            sizeof(T), detail::pointer_origin::none,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  In this case there is a host pointer, potentially initialized, but
  the user has not given the runtime any ownership, therefore the
  deleter has to be null.
  A Copy in is performed if the pointer is not null.
  If it is null, the data is initialized (new) inside the runtime.
  A copy out to hostPointer is performed by default.
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(T* hostPointer, const range<dims>& r,
         const property_list& propList = {})
      : buffer_mem(
            std::move(shared_ptr_class<void>(static_cast<void*>(hostPointer),
                                             detail::NullDeleter())),
            dims, r, sizeof(T), detail::pointer_origin::raw,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  In this case there is a host pointer, potentially initialized, but
  the user has not given the runtime any ownership, therefore the
  deleter has to be null.
  A Copy in is performed if the pointer is not null.
  If it is null, the data is initialized (new) inside the runtime.
  A copy out to hostPointer is performed by default.
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage in case the
             runtime requires it.
  @param propList List of buffer properties
  */
  buffer(T* hostPointer, const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            std::move(shared_ptr_class<void>(static_cast<void*>(hostPointer),
                                             detail::NullDeleter())),
            dims, r, sizeof(T), detail::pointer_origin::raw,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
  @brief Constructs a buffer with host pointer.
  The user has given a const pointer, but the buffer is not const, so
  the runtime copies the data into a temporary space created using the
  given allocator. If the given allocator is a null allocator, this fails.
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(const T* hostPointer, const range<dims>& r,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T, AllocatorT>(hostPointer, r.size()), dims, r,
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer with host pointer.
  The user has given a const pointer, but the buffer is not const, so
  the runtime copies the data into a temporary space created using the
  given allocator. If the given allocator is a null allocator, this fails.
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage
  @param propList List of buffer properties
  */
  buffer(const T* hostPointer, const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T>(hostPointer, r.size(), allocator), dims, r,
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  The user has given a shared pointer, therefore the data is explicitly
  shared between the user and the runtime.
  If the hostPointer is null, no data is copied in, and data is initialized
  inside the runtime. Data is copied out if the reference count of the
  runtime is less than the reference count of the shared pointer.
  @param hostPointer Shared pointer to host data
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(const shared_ptr_class<T>& hostPointer,  // NOLINT(runtime/references)
         const range<dims>& r, const property_list& propList = {})
      : buffer_mem(
            hostPointer, dims, r, sizeof(T), detail::pointer_origin::shared,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  The user has given a shared pointer, therefore the data is explicitly
  shared between the user and the runtime.
  If the hostPointer is null, no data is copied in, and data is initialized
  inside the runtime. Data is copied out if the reference count of the
  runtime is less than the reference count of the shared pointer.
  @param hostPointer Shared pointer to host data
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage
  @param propList List of buffer properties
  */
  buffer(const shared_ptr_class<T>& hostPointer,  // NOLINT(runtime/references)
         const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            hostPointer, dims, r, sizeof(T), detail::pointer_origin::shared,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
  @brief Construct a buffer as a subset from an existing buffer.
  @param b Existing buffer that will act as the parent
  @param base_index Offset of the original data to where the sub-buffer data
             starts
  @param sub_range Range of the original data that will be used in the
             sub-buffer
  */
  buffer(buffer<T, dims>& b, const id<dims>& base_index,
         const range<dims>& sub_range)
      : buffer_mem(b, base_index, sub_range) {}

  /**
  @brief Construct a buffer from an OpenCL object.
  @param mem_object the user-provided OpenCL object that will be used by the
  buffer
  @param fromQueue the queue holding the context associated with the mem_object
  object
  @param available_event if provided signals that the cl_mem object has been
  created and is ready to be used
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703,
      "Use the OpenCL interop constructor which takes a SYCL context instead.")
  buffer(cl_mem mem_object, queue& fromQueue,  // NOLINT(runtime/references)
         event available_event = event())
      : buffer_mem(
            mem_object, fromQueue, available_event, dims, sizeof(T),
            detail::make_base_allocator<T,
                                        AllocatorT>::get_buffer_allocator()) {}

  /**
  @brief Construct a buffer from an OpenCL object.
  @brief Construct a buffer from an OpenCL object.
  @param mem_object the user-provided OpenCL object that will be used by the
  buffer
  @param syclContext the context associated with the mem_object object
  @param available_event if provided signals that the cl_mem object has been
  created and is ready to be used
  */
  buffer(cl_mem mem_object,
         const context& syclContext,  // NOLINT(runtime/references)
         event available_event = event())
      : buffer_mem(
            mem_object, syclContext, available_event, dims, sizeof(T),
            detail::make_base_allocator<T,
                                        AllocatorT>::get_buffer_allocator()) {}
#if SYCL_LANGUAGE_VERSION >= 202002
  /**
   * @brief Constructs a buffer from a contiguous container, such
   * as std::vector or std::array. This constructor is only available when:
   * - std::data(container) and std::size(container) are well formed
   * - return type of std::data(container) is convertible to T*
   * @tparam Container Type of the container
   * @param container Container to construct the buffer from
   * @param allocator Allocator used to create internal storage
   * @param propList List of buffer properties
   */
  template <typename Container,
            typename = detail::is_contiguous_container<T, Container>>
  buffer(Container& container, AllocatorT allocator,
         const property_list& propList = {})
      : buffer(std::begin(container), std::end(container), allocator,
               propList) {}

  /**
   * @brief Constructs a buffer from a contiguous container, such
   * as std::vector or std::array. This constructor is only available when:
   * - std::data(container) and std::size(container) are well formed
   * - return type of std::data(container) is convertible to T*
   * @tparam Container Type of the container
   * @param container Container to construct the buffer from
   * @param propList List of buffer properties
   */
  template <typename Container,
            typename = detail::is_contiguous_container<T, Container>>
  buffer(Container& container, const property_list& propList = {})
      : buffer(std::begin(container), std::end(container), propList) {}

#else

  /**
  @brief Constructs a buffer from an std::vector, this is non-standard.
  A shared pointer is created from the raw data of the pointer,
  and a null-deleter is used to avoid the runtime clearing up the
  user-pointer.
  The range of the buffer is extracted from the size of the vector.
  The allocator from the vector is used as an allocator for the buffer.
  @param v Vector to construct the buffer from
  @param propList List of buffer properties
  */
  explicit buffer(vector_class<T>& v,  // NOLINT(runtime/references)
                  const property_list& propList = {})
      : buffer_mem(shared_ptr_class<T>(v.data(), detail::NullDeleter()), dims,
                   range<1>(static_cast<unsigned int>(v.size())), sizeof(T),
                   detail::pointer_origin::raw,
                   detail::make_base_allocator<
                       T, typename std::vector<T>::allocator_type>::
                       get_buffer_allocator(),
                   propList) {}

#endif  // SYCL_LANGUAGE_VERSION >= 202002

  /**
  @brief Constructs a buffer initialized by the given iterator range.
  This range is read-only, is not written.
  @param begin Iterator starting the range
  @param end  Iterator ending the range
  @param propList List of buffer properties
   */
  template <typename Iterator>
  buffer(Iterator begin, Iterator end, const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T, AllocatorT>(begin, end), dims,
            range<1>(std::distance(begin, end)), sizeof(T),
            detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer initialized by the given iterator range.
  This range is read-only, is not written.
  @param begin Iterator starting the range
  @param end  Iterator ending the range
  @param allocator The allocator used to create internal storage
  @param propList List of buffer properties
   */
  template <typename Iterator>
  buffer(Iterator begin, Iterator end, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T>(begin, end, allocator), dims,
            range<1>(std::distance(begin, end)), sizeof(T),
            detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /** @brief Default-constructs a buffer. The buffer is not valid for use in
   * SYCL kernels.
   */
  buffer()
      : buffer_mem(
            dims, range<dims>{detail::index_array{}}, sizeof(T),
            detail::make_base_allocator<T,
                                        AllocatorT>::get_buffer_allocator()) {}

  ~buffer() override = default;

  /**
  @brief Returns an accessor to the buffer, only used on the host side
  @tparam accessMode All access::mode values are accepted
  @return Host accessor
  */
  template <access::mode accessMode>
  accessor<T, dims, accessMode, access::target::host_buffer> get_access() {
    return accessor<T, dims, accessMode, access::target::host_buffer>(*this);
  }

#if SYCL_LANGUAGE_VERSION >= 202001
  /** @brief Returns a valid host_accessor as if constructed via passing the
   * buffer and all provided arguments to the SYCL host_accessor
   */
  template <typename... Ts>
  auto get_host_access(Ts... args) {
    return host_accessor{*this, args...};
  }
#endif  // SYCL_LANGUAGE_VERSION

  /**
  @brief this function returns an accessor to the buffer in the given
  command_group
  scope.
  @tparam accessMode all access::mode values are accepted
  @tparam accessTarget defaults to global_buffer, can accept
  global_buffer or constant_buffer
  @param cgh Reference to the command group scope where the accessor is
  requested.
  @return Device accessor
  */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  accessor<T, dims, accessMode, accessTarget> get_access(
      handler& cgh /* NOLINT */) {
    return accessor<T, dims, accessMode, accessTarget>(*this, cgh);
  }

  /**
  @brief Returns an accessor to the buffer in the given
             command_group scope.
  @tparam accessMode All access::mode values are accepted
  @tparam accessTarget Defaults to global_buffer, can accept
             global_buffer or constant_buffer
  @param cgh Reference to the command group scope where the accessor is
             requested.
  @param offset the offset that the accessor will be able to update from.
  @param range the range in which the accessor will be updating the data.
  @return Device accessor
  @deprecated Need to reverse the order of the access offset and range,
              see 4.7.2.1 Buffer Interface in SYCL 1.2.1
  */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor<T, dims, accessMode, accessTarget> get_access(
      handler& cgh, id<dims> offset,
      range<dims> range) {  // NOLINT
    return accessor<T, dims, accessMode, accessTarget>(*this, cgh, range,
                                                       offset);
  }

  /** @brief Returns an accessor to the buffer in the given command group scope.
   * @tparam accessMode All access::mode values are accepted
   * @tparam accessTarget Defaults to global_buffer, can accept
   *         global_buffer or constant_buffer
   * @param cgh Reference to the command group scope where the accessor is
   *            requested
   * @param range The range in which the accessor will be updating the data
   * @param offset The offset that the accessor will be able to update from
   * @return Device accessor
   */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  accessor<T, dims, accessMode, accessTarget> get_access(handler& cgh,
                                                         range<dims> range,
                                                         id<dims> offset = {}) {
    return accessor<T, dims, accessMode, accessTarget>(*this, cgh, range,
                                                       offset);
  }

  /**
  @brief Returns an accessor to the buffer, only used on the host side
  @tparam accessMode All access::mode values are accepted
  @param offset The offset that the accessor will be able to update from.
  @param range The range in which the accessor will be updating the data.
  @return Host accessor
  @deprecated Need to reverse the order of the access offset and range,
              see 4.7.2.1 Buffer Interface in SYCL 1.2.1
  */
  template <access::mode accessMode>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor<T, dims, accessMode, access::target::host_buffer> get_access(
      id<dims> offset, range<dims> range) {
    return accessor<T, dims, accessMode, access::target::host_buffer>(
        *this, range, offset);
  }

  /** @brief Returns an accessor to the buffer, only used on the host side
   * @tparam accessMode All access::mode values are accepted
   * @param range The range in which the accessor will be updating the data
   * @param offset The offset that the accessor will be able to update from
   * @return Host accessor
   */
  template <access::mode accessMode>
  accessor<T, dims, accessMode, access::target::host_buffer> get_access(
      range<dims> range, id<dims> offset = {}) {
    return accessor<T, dims, accessMode, access::target::host_buffer>(
        *this, range, offset);
  }

  /** @cond COMPUTECPP_DEV */
  /**
  @brief Creates a new public buffer object given an internal memory object.
  Implementation only
  */
  explicit buffer(dmem_shptr impl) : buffer_mem(impl) {}
  /** @endcond */

  /**
  @brief Returns the range of the buffer.
  */
  cl::sycl::range<dims> get_range() const {
    return cl::sycl::range<dims>(this->get_range_impl());
  }

  /**
  @brief Returns whether this SYCL buffer was constructed with the property
             specified by propertyT
  @tparam propertyT Property to check for
  @return True if buffer constructed with the property
  */
  template <typename propertyT>
  bool has_property() const noexcept {
    return this->get_properties().template has_property<propertyT>();
  }

  /**
  @brief Returns a copy of the property of type propertyT that this SYCL
             buffer was constructed with. Throws an error if buffer was not
             constructed with the property.
  @tparam propertyT Property to retrieve
  @return Copy of the property
  */
  template <typename propertyT>
  propertyT get_property() const {
    return this->get_properties().template get_property<propertyT>();
  }

  /**
  @brief Returns the allocator provided to the buffer
  @return Allocator that was provided to the buffer
  */
  AllocatorT get_allocator() const {
    return detail::cast_base_allocator<AllocatorT>(this->get_base_allocator());
  }

  /** Creates and returns a reinterpreted SYCL buffer
   * @tparam ReinterpretT New underlying type
   * @tparam ReinterpretDim New buffer dimension
   * @param reinterpretRange New buffer range
   * @return Reinterpreted buffer with the requested parameters
   * @throw invalid_object_error if the size of the reinterpreted buffer
   *        and the size of the original buffer do not match
   */
  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT> reinterpret(
      range<ReinterpretDim> reinterpretRange) const {
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        this->reinterpret_buffer(sizeof(ReinterpretT), ReinterpretDim,
                                 reinterpretRange));
  }

  /** Creates and returns a reinterpreted SYCL buffer
   * @tparam ReinterpretT New underlying type
   * @tparam ReinterpretDim New buffer dimension
   * @return Reinterpreted buffer with the requested parameters
   * @throw invalid_object_error if the size of the reinterpreted buffer
   *        and the size of the original buffer do not match
   */
  template <typename ReinterpretT, int ReinterpretDim = dims>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT> reinterpret() const {
    static_assert(
        (ReinterpretDim == 1) ||
            ((ReinterpretDim == dims) && (sizeof(T) == sizeof(ReinterpretT))),
        "Must provide a reinterpret range");
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        this->reinterpret_buffer(
            sizeof(ReinterpretT), ReinterpretDim,
            this->get_reinterpret_range<ReinterpretT, ReinterpretDim>()));
  }
};

#if SYCL_LANGUAGE_VERSION >= 202001

/** Deduction guide for buffer class template.
 */

template <typename InputIterator>
buffer(InputIterator, InputIterator, const property_list& = {})
    ->buffer<typename std::iterator_traits<InputIterator>::value_type, 1>;

template <typename InputIterator, typename AllocatorT>
buffer(InputIterator, InputIterator, AllocatorT, const property_list& = {})
    ->buffer<typename std::iterator_traits<InputIterator>::value_type, 1,
             AllocatorT>;

template <typename T, int dims>
buffer(const T*, const range<dims>&, const property_list& = {})
    ->buffer<T, dims>;

#if SYCL_LANGUAGE_VERSION >= 202002
/* Deduction guide for contiguous container constructors.
 */
template <typename Container>
buffer(Container&, const property_list& = {})
    ->buffer<typename Container::value_type, 1>;

template <typename Container, typename AllocatorT>
buffer(Container&, AllocatorT, const property_list& = {})
    ->buffer<typename Container::value_type, 1, AllocatorT>;

#endif  // SYCL_LANGUAGE_VERSION >= 202002

/** Property trait specializations
 */
template <>
struct is_property<property::buffer::use_host_ptr> : public std::true_type {};

template <>
struct is_property<property::buffer::context_bound> : public std::true_type {};

template <>
struct is_property<property::buffer::use_mutex> : public std::true_type {};

template <typename T, int dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_host_ptr,
                      buffer<T, dimensions, AllocatorT>>
    : public std::true_type {};

template <typename T, int dimensions, typename AllocatorT>
struct is_property_of<property::buffer::context_bound,
                      buffer<T, dimensions, AllocatorT>>
    : public std::true_type {};

template <typename T, int dimensions, typename AllocatorT>
struct is_property_of<property::buffer::use_mutex,
                      buffer<T, dimensions, AllocatorT>>
    : public std::true_type {};

#endif  // SYCL_LANGUAGE_VERSION >= 202001

/**
@brief Specialization for const buffers, that allows the creation of buffers
on the device from const data.
Any allocator, but the map allocator, can be used to create host data.
The allocator must remove the constness of the data in order to create
temporary objects, but host accessors will only be read only always.
*/
template <typename T, int dims, typename AllocatorT>
class buffer<const T, dims, AllocatorT> : public buffer_mem {
 public:
  /** @brief Helper for the user, alias for type T */
  using value_type = T;

  /** @brief Helper for the user, alias for reference to type T */
  using reference = T&;

  /** @brief Helper for the user, alias for const reference to type T */
  using const_reference = const T&;

  /** @brief Helper for the user, alias for the type of the allocator */
  using allocator_type = AllocatorT;

  /// Number of buffer dimensions (Codeplay extension)
  static constexpr const auto dimensions = dims;

  /**
  @brief Constructs a buffer with host pointer
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(const T* hostPointer, const range<dims>& r,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T, AllocatorT>(hostPointer, r.size()), dims, r,
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer with host pointer
  @param hostPointer Pointer to host data
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage
  @param propList List of buffer properties
  */
  buffer(const T* hostPointer, const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T>(hostPointer, r.size(), allocator), dims, r,
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  @param hostPointer Shared pointer to host data
  @param r Range of the buffer
  @param propList List of buffer properties
  */
  buffer(const shared_ptr_class<const T>&
             hostPointer,  // NOLINT(runtime/references)
         const range<dims>& r, const property_list& propList = {})
      : buffer_mem(
            std::const_pointer_cast<T>(hostPointer), dims, r, sizeof(T),
            detail::pointer_origin::shared,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

  /**
  @brief Constructs a buffer with a host pointer.
  @param hostPointer Shared pointer to host data
  @param r Range of the buffer
  @param allocator The allocator used to create internal storage
  @param propList List of buffer properties
  */
  buffer(const shared_ptr_class<const T>&
             hostPointer,  // NOLINT(runtime/references)
         const range<dims>& r, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            std::const_pointer_cast<T>(hostPointer), dims, r, sizeof(T),
            detail::pointer_origin::shared,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

#if SYCL_LANGUAGE_VERSION >= 202002
  /**
   * @brief Constructs a buffer from a contiguous container, such
   * as std::vector or std::array. This constructor is only available when:
   * - T is const
   * - std::data(container) and std::size(container) are well formed
   * - return type of std::data(container) is convertible to T*
   * @tparam Container Type of the container
   * @param container Container to construct the buffer from
   * @param allocator Allocator used to create internal storage
   * @param propList List of buffer properties
   */
  template <typename Container,
            typename = detail::is_contiguous_container<T, Container>>
  buffer(Container& container, AllocatorT allocator,
         const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T>(std::begin(container), std::end(container),
                                  allocator),
            dims,
            range<1>(std::distance(std::begin(container), std::end(container))),
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(
                allocator),
            propList) {}

  /**
   * @brief Constructs a buffer from a contiguous container, such
   * as std::vector or std::array. This constructor is only available when:
   * - T is const
   * - std::data(container) and std::size(container) are well formed
   * - return type of std::data(container) is convertible to T*
   * @tparam Container Type of the container
   * @param container Container to construct the buffer from
   * @param propList List of buffer properties
   */
  template <typename Container,
            typename = detail::is_contiguous_container<T, Container>>
  buffer(Container& container, const property_list& propList = {})
      : buffer_mem(
            detail::clone_data<T, AllocatorT>(std::begin(container),
                                              std::end(container)),
            dims,
            range<1>(std::distance(std::begin(container), std::end(container))),
            sizeof(T), detail::pointer_origin::raw_const,
            detail::make_base_allocator<T, AllocatorT>::get_buffer_allocator(),
            propList) {}

#endif
  /** @brief Default-constructs a buffer. The buffer is not valid for use in
   * SYCL kernels.
   */
  buffer()
      : buffer_mem(
            dims, range<dims>{detail::index_array{}}, sizeof(T),
            detail::make_base_allocator<T,
                                        AllocatorT>::get_buffer_allocator()) {}

  ~buffer() override = default;

  /**
  @brief this function returns an accessor to the buffer, this is only used on
  the host side
  @tparam accessMode all access::mode values are accepted
  @return Host accessor
  */
  template <access::mode accessMode>
  accessor<const T, dims, accessMode, access::target::host_buffer>
  get_access() {
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE host accessor from a CONST buffer");
    return accessor<const T, dims, accessMode, access::target::host_buffer>(
        *this);
  }

  /**
  @brief this function returns an accessor to the buffer in the given
         command_group scope.
  @tparam accessMode all access::mode values are accepted
  @tparam accessTarget defaults to global_buffer, can accept
  global_buffer or constant_buffer
  @param cgh Reference to the command group scope where the accessor is
  requested.
  @return Device accessor
  */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  accessor<const T, dims, accessMode, accessTarget> get_access(
      handler& cgh) {  // NOLINT
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE accessor from a CONST buffer");
    return accessor<const T, dims, accessMode, accessTarget>(*this, cgh);
  }

  /**
  @brief this function returns an accessor to the buffer in the given
         command_group scope.
  @tparam accessMode all access::mode values are accepted
  @tparam accessTarget defaults to global_buffer, can accept
  global_buffer or constant_buffer
  @param cgh Reference to the command group scope where the accessor is
  requested.
  @param offset the offset that the accessor will be able to update from.
  @param range the range in which the accessor will be updating the data.
  @return Device accessor
  @deprecated Need to reverse the order of the access offset and range,
              see 4.7.2.1 Buffer Interface in SYCL 1.2.1
  */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor<const T, dims, accessMode, accessTarget> get_access(
      handler& cgh, id<dims> offset, range<dims> range) {  // NOLINT
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE accessor from a CONST buffer");
    return accessor<const T, dims, accessMode, accessTarget>(*this, cgh, range,
                                                             offset);
  }

  /** @brief Returns an accessor to the buffer in the given command group scope
   * @tparam accessMode All access::mode values are accepted
   * @tparam accessTarget Defaults to global_buffer, can accept
   *                      global_buffer or constant_buffer
   * @param cgh Reference to the command group scope where the accessor is
   *        requested
   * @param range The range in which the accessor will be updating the data
   * @param offset The offset that the accessor will be able to update from
   * @return Device accessor
   */
  template <access::mode accessMode,
            access::target accessTarget = access::target::global_buffer>
  accessor<const T, dims, accessMode, accessTarget> get_access(
      handler& cgh, range<dims> range, id<dims> offset = {}) {
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE accessor from a CONST buffer");
    return accessor<const T, dims, accessMode, accessTarget>(*this, cgh, range,
                                                             offset);
  }

  /**
  @brief Returns an accessor to the buffer, only used on the host side
  @tparam accessMode All access::mode values are accepted
  @param offset the offset that the accessor will be able to update from.
  @param range the range in which the accessor will be updating the data.
  @return Device accessor
  @deprecated Need to reverse the order of the access offset and range,
              see 4.7.2.1 Buffer Interface in SYCL 1.2.1
  */
  template <access::mode accessMode>
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use overload where the range comes before the offset.")
  accessor<T, dims, accessMode, access::target::host_buffer> get_access(
      id<dims> offset, range<dims> range) {
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE host accessor from a CONST buffer");
    return accessor<T, dims, accessMode, access::target::host_buffer>(
        *this, range, offset);
  }

  /** @brief Returns an accessor to the buffer, only used on the host side
   * @tparam accessMode All access::mode values are accepted
   * @param range the range in which the accessor will be updating the data
   * @param offset the offset that the accessor will be able to update from
   * @return Host accessor
   */
  template <access::mode accessMode>
  accessor<T, dims, accessMode, access::target::host_buffer> get_access(
      range<dims> range, id<dims> offset = {}) {
    static_assert((accessMode != access::mode::read_write) &&
                      (accessMode != access::mode::write),
                  "Cannot create a WRITE host accessor from a CONST buffer");
    return accessor<T, dims, accessMode, access::target::host_buffer>(
        *this, range, offset);
  }

  explicit buffer(dmem_shptr impl) : buffer_mem(impl) {}

  /** @brief Retrieves the range of the buffer
   */
  cl::sycl::range<dims> get_range() const {
    return cl::sycl::range<dims>(this->get_range_impl());
  }

  /**
  @brief Returns whether this SYCL buffer was constructed with the property
         specified by propertyT
  @tparam propertyT Property to check for
  @return True if buffer constructed with the property
  */
  template <typename propertyT>
  bool has_property() const {
    return this->get_properties().template has_property<propertyT>();
  }

  /**
  @brief Returns a copy of the property of type propertyT that this SYCL
         buffer was constructed with. Throws an error if buffer was not
         constructed with the property.
  @tparam propertyT Property to retrieve
  @return Copy of the property
  */
  template <typename propertyT>
  propertyT get_property() const {
    return this->get_properties().template get_property<propertyT>();
  }

  /**
  @brief Returns the allocator provided to the buffer
  @return Allocator that was provided to the buffer
  */
  AllocatorT get_allocator() const {
    return detail::cast_base_allocator<AllocatorT>(this->get_base_allocator());
  }

  /** Creates and returns a reinterpreted SYCL buffer
   * @tparam ReinterpretT New underlying type
   * @tparam ReinterpretDim New buffer dimension
   * @param reinterpretRange New buffer range
   * @return Reinterpreted buffer with the requested parameters
   * @throw invalid_object_error if the size of the reinterpreted buffer
   *        and the size of the original buffer do not match
   */
  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT> reinterpret(
      range<ReinterpretDim> reinterpretRange) const {
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        this->reinterpret_buffer(sizeof(ReinterpretT), reinterpretRange));
  }

  /** Creates and returns a reinterpreted SYCL buffer
   * @tparam ReinterpretT New underlying type
   * @tparam ReinterpretDim New buffer dimension
   * @return Reinterpreted buffer with the requested parameters
   * @throw invalid_object_error if the size of the reinterpreted buffer
   *        and the size of the original buffer do not match
   */
  template <typename ReinterpretT, int ReinterpretDim = dims>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT> reinterpret() const {
    static_assert((ReinterpretDim == 1) || (sizeof(T) == sizeof(ReinterpretT)),
                  "Must provide a reinterpret range");
    return buffer<ReinterpretT, ReinterpretDim, AllocatorT>(
        this->reinterpret_buffer(
            sizeof(ReinterpretT), ReinterpretDim,
            this->get_reinterpret_range<ReinterpretT, ReinterpretDim>()));
  }
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
template <typename T, int dimensions, typename AllocatorT>
struct hash<cl::sycl::buffer<T, dimensions, AllocatorT>> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(
      const cl::sycl::buffer<T, dimensions, AllocatorT>& object) const {
    hash<cl::sycl::dmem_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_BUFFER_H_

////////////////////////////////////////////////////////////////////////////////
