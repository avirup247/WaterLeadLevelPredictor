/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file item.h
 *
 * @brief This file implements the @ref cl::sycl::item and @ref
 * cl::sycl::nd_item classes as defined by the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_ITEM_H_
#define RUNTIME_INCLUDE_SYCL_ITEM_H_

#include "SYCL/builtins/extended.h"
#include "SYCL/common.h"
#include "SYCL/device_event.h"
#include "SYCL/experimental/sub_group.h"
#include "SYCL/group_base.h"
#include "SYCL/item_base.h"
#include "SYCL/multi_pointer.h"
#include "SYCL/type_traits.h"
#include "computecpp/gsl/gsl"

namespace cl {
namespace sycl {

namespace detail {

/** @brief Asserts that the dimensionality of the accessors are valid when
 * used in begin_dma_transfer
 * @tparam N Number of dimensions
 */
template <int N>
inline void assert_plane_range() {
  static_assert((N == 1) || (N == 2),
                "codeplay_begin_dma_transfer only supports 1 or 2 "
                "dimensional accessors");
}

/** @brief Asserts that the access mode used allows reading data
 * @tparam mode Access mode
 */
template <access::mode mode>
inline void assert_read_mode() {
  static_assert(detail::is_read_mode<mode>::value,
                "Access mode must allow reading");
}

/** @brief Asserts that the access mode used allows writing data
 * @tparam mode Access mode
 */
template <access::mode mode>
inline void assert_write_mode() {
  static_assert(detail::is_write_mode<mode>::value,
                "Access mode must allow writing");
}

}  // namespace detail

/** \brief The cl::sycl::item object is a container for all information about a
 * work item. The cl::sycl::item object is used within the \ref
 * handler::parallel_for and
 * \ref parallel_for_work_item functions. The cl::sycl::item object can return
 * information about the local and global sizes of an enqueued nd_range as well
 * as the local and global ids of the work item.
 * item<dimensions> is a derived class of detail::item_base, which is a
 * non-templated class implementing most of the functionality of
 * cl::sycl::item<dimensions>.
 * \tparam dimensions Number of dimensions of the item object
 * \tparam with_offset Whether the object contains an offset or not
 */
template <int dimensions = 1, bool with_offset = true>
class item : public detail::item_base {
  static_assert(((dimensions > 0) && (dimensions < 4)),
                "Dimensions outside the domain [1,3]");

  /** @brief Helper boolean type to perform SFINAE on the conversion function
   */
  using offset_t = typename std::decay<decltype(
      std::integral_constant<bool, with_offset>::value)>::type;

  /** @brief SFINAE helper function that checks that with_offset is false
   * @tparam T Dummy type that should be equal to bool
   * @return True if with_offset is false
   */
  template <class T>
  static constexpr bool has_no_offset() {
    return (std::is_same<T, bool>::value && !with_offset);
  }

 private:
  using base_t = detail::item_base;
  /** \brief This constructor should not be called. \ref item are constructed
   * by the runtime.
   * \warning Cannot be used (deleted).
   */
  item() = delete;

  /** @internal
   * \brief This constructor should not called outside of the runtime.
   * \ref  are constructed by the runtime.
   */
  item(detail::index_array id, detail::index_array range)
      : detail::item_base(id, range, detail::index_array(0, 0, 0)) {}

 public:
  /** \brief Copy constructor. Create a copy of another \ref item object.
   */
  item(const item& rhs) = default;

  /** @internal
   * Constructor from detail::item_base
   * \brief converts the derived index_array class to its base class
   * Its used for APIs that use id structs.
   */
  item(const detail::item_base& itemBase)  // NOLINT false +,  conversion
      : detail::item_base(itemBase) {}

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const item& lhs, const item& rhs) {
    return lhs.is_equal<dimensions>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const item& lhs, const item& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Gets the global range in a specified dimension if provided by a
   * handler::parallel_for and gets the local range in a specified dimension if
   * provided by a \ref parallel_for_work_item.
   * @param dimension The dimension of the range to retrieve.
   * @return The value of the global range in the dimension specified.
   */
  size_t get_range(int dimension) const { return this->m_range[dimension]; }

  /** \brief Get the global range of the enqueued kernel if provided by a \ref
   * handler::parallel_for.
   * Get the local range if used if provided by a \ref parallel_for_work_item.
   * @return the values of the global range for all dimensions.
   */
  range<dimensions> get_range() const {
    return range<dimensions>(this->m_range);
  }

  /** \brief Get the invocation offset.
   * \tparam COMPUTECPP_ENABLE_IF Only available if with_offset is true
   * \return the offset used to invoke the kernel.
   */
  template <COMPUTECPP_ENABLE_IF(offset_t, with_offset)>
  id<dimensions> get_offset() const {
    return id<dimensions>(this->m_offset);
  }

  /**\brief Get the id for a specific dimension.
   * @param dimension of the id, in the range [0,2]
   * @return the id for the specified dimension.
   */
  size_t get_id(int dimension) const { return base_t::get_id(dimension); }

  /** \brief Get the id associated with this item for all dimensions.
   * @return The global \ref id if the item is provided by a \ref
   * handler::parallel_for.
   * The local \ref id if the item is provided by a \ref parallel_for_work_item.
   */
  id<dimensions> get_id() const { return id<dimensions>(this->m_id); }

  /** @brief Returns an item representing the same information as the object
   *        holds but also includes the offset set to 0
   * @tparam sfinae Dummy type to signal that this conversion is only available
   *         if with_offset is false
   * @return This object with an offset added
   */
  template <class sfinae = offset_t>
  operator detail::enable_if_t<has_no_offset<sfinae>(),
                               item<dimensions, true>>() const {
    return item<dimensions, true>(this->get_id(), this->get_range());
  }
};

/**
  @brief The cl::sycl::nd_item object is a container for all information about
  a work-item.

  The cl::sycl::nd_item object is used within the \ref handler::parallel_for
  functions. The cl::sycl::item object can return information about the local
  and global sizes of an enqueued \ref nd_range as well as the local and global
  ids of the work item.
*/
template <int dimensions = 1>
class nd_item : public detail::nd_item_base {
  static_assert(((dimensions > 0) && (dimensions < 4)),
                "Dimensions outside the domain [1,3]");

 protected:
  using base_t = detail::nd_item_base;
  /** \brief This constructor should not be called. nd_item has to be
   * constructed from the runtime. \warning Cannot be used (deleted).
   */
  nd_item() = delete;

  /** \brief Get an \ref item object embedding global information from this \ref
   * nd_range.
   * \return A global \ref item object.
   */
  item<dimensions> get_global_item() const {
    return item<dimensions>(base_t::get_global_item());
  }
  /** \brief Get an \ref item object embedding local information from this \ref
   * nd_range.
   * \return A local \ref item object.
   */
  item<dimensions> get_local_item() const {
    return item<dimensions>(base_t::get_local_item());
  }

  /** @brief Retrieves the group ID
   * @return ID of the group associated with this object
   */
  inline id<dimensions> get_group_id() const {
    return id<dimensions>(base_t::get_group_id());
  }

  /** @brief Checks if the ID of this nd_item is all zeros
   * @return True if current ID is (0, 0, 0)
   */
  bool is_zero_id() const {
    const auto id = this->get_local_item().get_id();
    bool isZeroId = (id[0] == 0);
    for (int i = 1; i < dimensions; ++i) {
      isZeroId = isZeroId && (id[i] == 0);
    }
    return isZeroId;
  }

 public:
  /** \brief copy constructor is public
   */
  nd_item(const nd_item& rhs) = default;

  /** Copy Constructor from nd_item<dims>
   * @brief copies the given nd_item with the same dimensionality.
   * If the dimensionality is different, a compiler error is produced.
   */
  template <int dimensions2>
  nd_item(const nd_item<dimensions2>& rhs) : detail::nd_item_base(rhs) {
    static_assert(dimensions2 == dimensions,
                  "Cannot convert nd_item to another dimensionality");
  }

  /** @internal
   * Constructor from detail::nd_item_base
   * \brief converts the derived nd_item<dimensions> class to its base class
   * Its as for APIs that use id structs.
   */
  nd_item(const detail::nd_item_base& i)  // NOLINT conversion
      : detail::nd_item_base(i) {}

  /** @brief Access to the subgroup functionality
   * @return A new instance of the class @ref sub_group
   */
  experimental::sub_group get_sub_group() const {
    using experimental::sub_group;
#ifdef __SYCL_DEVICE_ONLY__
    auto sub_group_size = detail::get_sub_group_size();
    auto sub_group_range = base_t::get_global_range(0) / sub_group_size;
    return sub_group(detail::get_sub_group_id(), sub_group_range,
                     sub_group_range, detail::get_sub_group_item_id(),
                     sub_group_size, sub_group_size);
#else
    return sub_group(0, 1, 1, get_local_id(0), 1, 1);
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Access to subgroup barrier
   * @param accessSpace Barrier fence space
   */
  COMPUTECPP_DEPRECATED_API("Use sub_group::barrier instead")
  void sub_group_barrier(access::fence_space accessSpace =
                             access::fence_space::global_and_local) const {
    detail::sub_group_barrier_impl(accessSpace);
  }

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const nd_item& lhs, const nd_item& rhs) {
    return lhs.is_equal<dimensions>(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const nd_item& lhs, const nd_item& rhs) {
    return !(lhs == rhs);
  }

  /** \brief Returns the global id for a specific dimension.
   * @param dimension of global id to return. Must be in the range [0,2].
   * @return the global id for the specified dimension.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use nd_item::get_global_id(unsigned) instead.")
  size_t get_global(unsigned int dimension) const {
    return this->get_global_id(dimension);
  }

  /** \brief Returns the global id for a specific dimension.
   * @param dimension of global id to return. Must be in the range [0,2].
   * @return the global id for the specified dimension.
   */
  size_t get_global_id(unsigned int dimension) const {
    return base_t::get_global_id(dimension);
  }

  /** \brief Return the global id for all dimension.
   * @return An \ref id object representing the global id for all dimension.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(201703,
                                    "Use nd_item::get_global_id() instead.")
  id<dimensions> get_global() const { return this->get_global_id(); }

  /** \brief Return the global id for all dimension.
   * @return An \ref id object representing the global id for all dimension.
   */
  id<dimensions> get_global_id() const { return get_global_item().get_id(); }

  /** \brief Returns the local id for a specific dimension.
   * @param dimension of local id to return. Must be in the range [0,2].
   * @return the local id for the specified dimension.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use nd_item::get_local_id(unsigned) instead.")
  size_t get_local(unsigned int dimension) const {
    return this->get_local_id(dimension);
  }

  /** \brief Returns the local id for a specific dimension.
   * @param dimension of local id to return. Must be in the range [0,2].
   * @return the local id for the specified dimension.
   */
  size_t get_local_id(unsigned int dimension) const {
    return base_t::get_local_id(dimension);
  }

  /** \brief Return the local id for all dimension.
   * @return An \ref id object representing the local id for all dimension.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(201703,
                                    "Use nd_item::get_local_id() instead.")
  id<dimensions> get_local() const { return this->get_local_id(); }

  /** \brief Return the local id for all dimension.
   * @return An \ref id object representing the local id for all dimension.
   */
  id<dimensions> get_local_id() const { return get_local_item().get_id(); }

  /** \brief Get the global range for a specified dimension.
   * @param dimension of the global range to be returned. Must be in the range
   * [0,2].
   * @return the value of the global range for the specified dimension.
   */
  size_t get_global_range(int dimension) const {
    return base_t::get_global_range(dimension);
  }

  /** \brief Get the global range of the enqueued nd_range.
   * @return the values of the global range for all dimensions.
   */
  range<dimensions> get_global_range() const {
    return get_global_item().get_range();
  }

  /** \brief Get the local range for a specified dimension.
   * @param dimension of the local range to be returned. Must be in the range
   * [0,2].
   * @return the value of the local range for the specified dimension.
   */
  size_t get_local_range(int dimension) const {
    return base_t::get_local_range(dimension);
  }

  /** \brief Get the local range of the enqueued nd_range.
   * @return the values of the local range for all dimensions.
   */
  range<dimensions> get_local_range() const {
    return get_local_item().get_range();
  }

  /** @brief Returns the current group id in a given dimension.
   * @param dimension of the id to be returned. Must be in the range [0,2].
   * @return the value of the group range in the specified dimension.
   */
  size_t get_group(unsigned int dim) const { return base_t::get_group(dim); }

  /** @brief Returns the group.
   * @return A \ref group object.
   */
  group<dimensions> get_group() const {
    return group<dimensions>(
        detail::group_base(get_group_id(), get_group_range(),
                           get_global_range(), get_local_range()));
  }

  /** \brief Get the offset of the enqueued nd_range.
   * @return The offset.
   */
  id<dimensions> get_offset() const { return get_global_item().get_offset(); }

  /** @brief Returns the group range of the enqueued nd_range.
   * @return the value of the group range for all dimensions.
   */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(201703,
                                    "Use nd_item::get_group_range() instead.")
  range<dimensions> get_num_groups() const { return this->get_group_range(); }

  /** @brief Returns the group range of the enqueued nd_range.
   * @return the value of the group range for all dimensions.
   */
  range<dimensions> get_group_range() const {
    return range<dimensions>(base_t::get_group_range());
  }

  /** @brief Returns the group range of the enqueued nd_range for a specific
    dimension.
  * @param dimension of the range to be returned. Must be in the range [0,2]
  * @return the value of the group range for all dimensions.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_VER(
      201703, "Use nd_item::get_group_range(unsigned) instead.")
  size_t get_num_groups(int dimension) const {
    return this->get_group_range(dimension);
  }

  /** @brief Returns the group range of the enqueued nd_range for a specific
    dimension.
  * @param dimension of the range to be returned. Must be in the range [0,2]
  * @return the value of the group range for all dimensions.
  */
  size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  /** @brief Returns the group linear id.
   * @return the group linear id.
   */
  size_t get_group_linear_id() const { return get_group().get_linear_id(); }

  /** @brief Returns the enqueued nd_range.
   * @return the enqueued nd_range.
   */
  nd_range<dimensions> get_nd_range() const {
    return detail::nd_range_base(get_global_range(), get_local_range(),
                                 get_offset());
  }

  /** @brief Asynchronous work group copy from a local pointer to global.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in local memory
   * @param src Pointer to the source in global memory
   * @param numElements Number of elements to copy
   * @param destStride Stride in the origin
   * @todo Use builtins
   */
  template <typename dataT>
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements,
                                     size_t srcStride = 1) const {
    return detail::async_work_group_copy_src_strided(
        dest, src, numElements, srcStride, this->is_zero_id());
  }

  /** @brief Asynchronous work group copy from a local pointer to global.
   * @tparam dataT Data type of the pointer
   * @param dest Pointer to the destination in global memory
   * @param src Pointer to the source in local memory
   * @param numElements Number of elements to copy
   * @param destStride Stride in the destination
   * @todo Use builtins
   */
  template <typename dataT>
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements,
                                     size_t destStride = 1) const {
    return detail::async_work_group_copy_dest_strided(
        dest, src, numElements, destStride, this->is_zero_id());
  }

  /** @brief Waits until codeplay_begin_dma_transfer completes.
   */
  COMPUTECPP_DEPRECATED_API(
      "Deprecated Codeplay extension, use the codeplay_await_dma_transfer free "
      "function instead.")
  void codeplay_await_dma_transfer() const {
#ifdef __SYCL_DEVICE_ONLY__
    ::cl::sycl::detail::end_dma_transfer();
#else
    // Nothing to do on host
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Performs an asynchronous copy from global memory plane to subgroup
   * local memory.
   * @param source The region of memory to copy data from.
   * @param destination The region of memory to write the data to.
   * @param copyBounds The shape of the region.
   * @param offset The offset into the planar region of memory.
   * @param stride The subgroup local memory stride.
   */
  template <class dataT, int sourceDim, int destinationDim,
            access::mode sourceMode, access::placeholder isPlaceholderSrc>
  COMPUTECPP_DEPRECATED_API(
      "Deprecated Codeplay extension, use the codeplay_begin_dma_transfer free "
      "function instead.")
  void codeplay_begin_dma_transfer(
      const accessor<dataT, sourceDim, sourceMode,
                     access::target::global_buffer, isPlaceholderSrc>& source,
      const accessor<dataT, destinationDim, access::mode::read_write,
                     access::target::subgroup_local>& destination,
      const range<2> copyBounds, size_t offset, size_t stride) {
    detail::assert_plane_range<sourceDim>();
    detail::assert_plane_range<destinationDim>();
    detail::assert_read_mode<sourceMode>();
#ifdef __SYCL_DEVICE_ONLY__
    const auto width = copyBounds[0];
    const auto height = copyBounds[1];
    ::cl::sycl::detail::begin_dma_transfer(destination.get_pointer(),
                                           source.get_device_plane_id(), offset,
                                           width, height, stride);
#else
    (void)source;
    (void)destination;
    (void)copyBounds;
    (void)offset;
    (void)stride;
    COMPUTECPP_NOT_IMPLEMENTED(
        "ComputeCpp has not yet implemented codeplay_begin_dma_transfer "
        "for host.");
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Performs an asynchronous copy from a global memory plane to
   * subgroup local memory.
   * @param source The region of memory to copy data from.
   * @param destination The region of memory to write the data to.
   * @param copyBounds The shape of the region.
   * @param offset The offset into the planar region of memory.
   * @param stride The subgroup local memory stride.
   */
  template <class dataT, int dim, access::mode sourceMode,
            access::placeholder isPlaceholderSrc>
  COMPUTECPP_DEPRECATED_API(
      "Deprecated Codeplay extension, use the codeplay_begin_dma_transfer free "
      "function instead.")
  void codeplay_begin_dma_transfer(
      const accessor<dataT, dim, sourceMode, access::target::global_buffer,
                     isPlaceholderSrc>& source,
      const multi_ptr<dataT, access::address_space::subgroup_local_space>
          destination,
      const range<2> copyBounds, size_t offset, size_t stride) {
    detail::assert_plane_range<dim>();
    detail::assert_read_mode<sourceMode>();
#ifdef __SYCL_DEVICE_ONLY__
    const auto width = copyBounds[0];
    const auto height = copyBounds[1];
    ::cl::sycl::detail::begin_dma_transfer(destination,
                                           source.get_device_plane_id(), offset,
                                           width, height, stride);
#else
    (void)source;
    (void)destination;
    (void)copyBounds;
    (void)offset;
    (void)stride;
    COMPUTECPP_NOT_IMPLEMENTED(
        "ComputeCpp has not yet implemented codeplay_begin_dma_transfer "
        "for host.");
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Performs an asynchronous copy from subgroup local memory to a
   * global memory plane.
   * @param source The region of memory to copy data from.
   * @param destination The region of memory to write the data to.
   * @param copyBounds The shape of the region.
   * @param offset The offset into the planar region of memory.
   * @param stride The subgroup local memory stride.
   */
  template <class dataT, int sourceDim, int destinationDim,
            access::mode destinationMode, access::placeholder isPlaceholderDst>
  COMPUTECPP_DEPRECATED_API(
      "Deprecated Codeplay extension, use the codeplay_begin_dma_transfer free "
      "function instead.")
  void codeplay_begin_dma_transfer(
      const accessor<dataT, sourceDim, access::mode::read_write,
                     access::target::subgroup_local>& source,
      const accessor<dataT, destinationDim, destinationMode,
                     access::target::global_buffer, isPlaceholderDst>&
          destination,
      const range<2> copyBounds, size_t offset, size_t stride) {
    detail::assert_plane_range<sourceDim>();
    detail::assert_plane_range<destinationDim>();
    detail::assert_write_mode<destinationMode>();
#ifdef __SYCL_DEVICE_ONLY__
    const auto width = copyBounds[0];
    const auto height = copyBounds[1];
    ::cl::sycl::detail::begin_dma_transfer(destination.get_device_plane_id(),
                                           offset, source.get_pointer(), width,
                                           height, stride);
#else
    (void)source;
    (void)destination;
    (void)copyBounds;
    (void)offset;
    (void)stride;
    COMPUTECPP_NOT_IMPLEMENTED(
        "ComputeCpp has not yet implemented codeplay_begin_dma_transfer "
        "for host.");
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Performs an asynchronous copy from subgroup local memory to a
   * global memory plane.
   * @param source The region of memory to copy data from.
   * @param destination The region of memory to write the data to.
   * @param copyBounds The shape of the region.
   * @param offset The offset into the planar region of memory.
   * @param stride The subgroup local memory stride.
   */
  template <class dataT, int dim, access::mode destinationMode,
            access::placeholder isPlaceholderDst>
  COMPUTECPP_DEPRECATED_API(
      "Deprecated Codeplay extension, use the codeplay_begin_dma_transfer free "
      "function instead.")
  void codeplay_begin_dma_transfer(
      const multi_ptr<dataT, access::address_space::subgroup_local_space>
          source,
      const accessor<dataT, dim, destinationMode, access::target::global_buffer,
                     isPlaceholderDst>& destination,
      const range<2> copyBounds, size_t offset, size_t stride) {
    detail::assert_plane_range<dim>();
    detail::assert_write_mode<destinationMode>();
#ifdef __SYCL_DEVICE_ONLY__
    const auto width = copyBounds[0];
    const auto height = copyBounds[1];
    ::cl::sycl::detail::begin_dma_transfer(destination.get_device_plane_id(),
                                           offset, source, width, height,
                                           stride);
#else
    (void)source;
    (void)destination;
    (void)copyBounds;
    (void)offset;
    (void)stride;
    COMPUTECPP_NOT_IMPLEMENTED(
        "ComputeCpp has not yet implemented codeplay_begin_dma_transfer "
        "for host.");
#endif  // __SYCL_DEVICE_ONLY__
  }

  /** @brief Waits on each given device_event
   * @tparam eventTN Pack of device_event types
   * @param events Pack of device_events
   */
  template <typename... eventTN>
  void wait_for(eventTN... events) const {
    static_assert(computecpp::gsl::conjunction<
                      std::is_same<cl::sycl::device_event, eventTN>...>::value,
                  "All events must be of type device_event");
    auto eventList = {events...};
    for (auto& event : eventList) {
      event.wait();
    }
  }
};

/** @brief Identifies an instance of a parallel_for_work_item function object
 *        executing at each point in a local range passed to a
 *        parallel_for_work_item call.
 *
 *        It encapsulates enough information to identify the work-item's local
 *        and global items according to the information given to
 *        parallel_for_work_group (physical ids) as well as the work-item's
 *        logical local items in the flexible range.
 *        All returned item objects are offset-less.
 * @tparam dimensions Number of dimensions of the h_item object
 */
template <int dimensions>
class h_item : public detail::h_item_base {
 private:
  using base_t = detail::h_item_base;

 public:
  /** @brief Not user constructible
   */
  h_item() = delete;

  /// @cond COMPUTECPP_DEV

  /** @brief Constructs an instance from a base object
   * @param base The base object to copy from
   */
  explicit h_item(const base_t& base) : base_t(base) {}

  /// COMPUTECPP_DEV @endcond

  /** @brief Equality operator
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  friend inline bool operator==(const h_item& lhs, const h_item& rhs) {
    return lhs.is_equal(rhs);
  }

  /** @brief Non-equality operator
   * @param rhs Object to compare to
   * @return True if at least one member variable not the same
   *         as the corresponding rhs member variable
   */
  friend inline bool operator!=(const h_item& lhs, const h_item& rhs) {
    return !(lhs == rhs);
  }

  /** @brief Retrieves the constituent global item representing the work-item's
   *        position in the global iteration space
   * @return Item representing the global ID and range
   */
  inline item<dimensions, false> get_global() const {
    return this->get_global_item_base();
  }

  /** @brief Retrieves the constituent logical local item representing the
   *        work-item's position in the local iteration space as provided upon
   *        the invocation of parallel_for_work_item
   * @return Item representing the logical local ID and range
   */
  inline item<dimensions, false> get_local() const {
    return this->get_logical_local();
  }

  /** @brief Retrieves the constituent logical local item representing the
   *        work-item's position in the local iteration space as provided upon
   *        the invocation of parallel_for_work_item
   * @return Item representing the logical local ID and range
   */
  inline item<dimensions, false> get_logical_local() const {
    return this->get_logical_local_item_base();
  }

  /** @brief Retrieves the constituent physical local item representing the
   *        work-item's position in the local iteration space as provided upon
   *        the invocation of parallel_for_work_group
   * @return Item representing the physical local ID and range
   */
  inline item<dimensions, false> get_physical_local() const {
    return this->get_physical_local_item_base();
  }

  /** @brief Retrieves the range representing the sizes
   *        of the global iteration space
   * @return Global range
   */
  inline range<dimensions> get_global_range() const {
    return this->get_global().get_range();
  }

  /** @brief Retrieves the value of the global range for the specified dimension
   * @param dimension Which value of the range to retrieve
   * @return Element of the global range
   */
  inline size_t get_global_range(int dimension) const {
    return this->get_global_range().get(dimension);
  }

  /** @brief Retrieves the id representing the position of the item in the
   *        global iteration space
   * @return Global ID
   */
  inline id<dimensions> get_global_id() const {
    return this->get_global().get_id();
  }

  /** @brief Retrieves the value of the global ID for the specified dimension
   * @param dimension Which value of the ID to retrieve
   * @return Element of the global ID
   */
  inline size_t get_global_id(int dimension) const {
    return this->get_global().get_id(dimension);
  }

  /** @brief Retrieves the range representing the sizes
   *        of the logical local iteration space
   * @return Global range
   */
  inline range<dimensions> get_local_range() const {
    return this->get_local().get_range();
  }

  /** @brief Retrieves the value of the logical local range for the specified
   *        dimension
   * @param dimension Which value of the range to retrieve
   * @return Element of the logical local range
   */
  inline size_t get_local_range(int dimension) const {
    return this->get_local_range().get(dimension);
  }

  /** @brief Retrieves the id representing the position of the item in the
   *        logical local iteration space
   * @return Global ID
   */
  inline id<dimensions> get_local_id() const {
    return this->get_local().get_id();
  }

  /** @brief Retrieves the value of the logical local ID for the specified
   *        dimension
   * @param dimension Which value of the ID to retrieve
   * @return Element of the logical local ID
   */
  inline size_t get_local_id(int dimension) const {
    return this->get_local().get_id(dimension);
  }

  /** @brief Retrieves the range representing the sizes
   *        of the logical local iteration space
   * @return Global range
   */
  inline range<dimensions> get_logical_local_range() const {
    return this->get_logical_local().get_range();
  }

  /** @brief Retrieves the value of the logical local range for the specified
   *        dimension
   * @param dimension Which value of the range to retrieve
   * @return Element of the logical local range
   */
  inline size_t get_logical_local_range(int dimension) const {
    return this->get_logical_local_range().get(dimension);
  }

  /** @brief Retrieves the id representing the position of the item in the
   *        logical local iteration space
   * @return Global ID
   */
  inline id<dimensions> get_logical_local_id() const {
    return this->get_logical_local().get_id();
  }

  /** @brief Retrieves the value of the logical local ID for the specified
   *        dimension
   * @param dimension Which value of the ID to retrieve
   * @return Element of the logical local ID
   */
  inline size_t get_logical_local_id(int dimension) const {
    return this->get_logical_local().get_id(dimension);
  }

  /** @brief Retrieves the range representing the sizes
   *        of the physical local iteration space
   * @return Global range
   */
  inline range<dimensions> get_physical_local_range() const {
    return this->get_physical_local().get_range();
  }

  /** @brief Retrieves the value of the physical local range for the specified
   *        dimension
   * @param dimension Which value of the range to retrieve
   * @return Element of the physical local range
   */
  inline size_t get_physical_local_range(int dimension) const {
    return this->get_physical_local_range().get(dimension);
  }

  /** @brief Retrieves the id representing the position of the item in the
   *        physical local iteration space
   * @return Global ID
   */
  inline id<dimensions> get_physical_local_id() const {
    return this->get_physical_local().get_id();
  }

  /** @brief Retrieves the value of the physical local ID for the specified
   *        dimension
   * @param dimension Which value of the ID to retrieve
   * @return Element of the physical local ID
   */
  inline size_t get_physical_local_id(int dimension) const {
    return this->get_physical_local_id()[dimension];
  }

  /** @brief Helper function for calling operator==()
   * @param rhs Object to compare to
   * @return True if all member variables are equal to rhs member variables
   */
  inline bool is_equal(const h_item& rhs) const {
    return base_t::is_equal<dimensions>(rhs);
  }
};

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_ITEM_H_
