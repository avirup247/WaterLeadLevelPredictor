/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

#ifndef RUNTIME_INCLUDE_SYCL_REDUCTION_H_
#define RUNTIME_INCLUDE_SYCL_REDUCTION_H_

#include "SYCL/accessor.h"
#include "SYCL/common.h"
#include "SYCL/functional.h"
#include "SYCL/group_functions.h"
#include "SYCL/item.h"
#include "SYCL/property.h"  // IWYU pragma: keep

#include <tuple>

namespace cl {
namespace sycl {

namespace detail {

/** Takes a linearized index and the range associated with that index and
 * recreates a sycl::id.
 * @param range The 1D range associated with the index
 * @param index The linear index.
 * @return A new sycl::id<1> based on the range and index
 */
inline id<1> get_delinearized_id(const range<1>&, size_t index) {
  return {index};
}

/** Takes a linearized index and the range associated with that index and
 * recreates a sycl::id.
 * @param range The 2D range associated with the index
 * @param index The linear index.
 * @return A new sycl::id<2> based on the range and index
 */
inline id<2> get_delinearized_id(const range<2>& range, size_t index) {
  size_t x = index % range[1];
  size_t y = index / range[1];
  return {y, x};
}

/** Takes a linearized index and the range associated with that index and
 * recreates a sycl::id.
 * @param range The 3D range associated with the index
 * @param index The linear index.
 * @return A new sycl::id<3> based on the range and index
 */
inline id<3> get_delinearized_id(const range<3>& range, size_t index) {
  size_t d1d2 = range[1] * range[2];
  size_t z = index / d1d2;
  size_t zRest = index % d1d2;
  size_t y = zRest / range[2];
  size_t x = zRest % range[2];
  return {z, y, x};
}

#if SYCL_LANGUAGE_VERSION >= 202002

template <typename T, class binaryOp>
using is_plus = bool_constant<std::is_same<binaryOp, sycl::plus<T>>::value ||
                              std::is_same<binaryOp, sycl::plus<void>>::value>;

template <typename T, class binaryOp>
using is_multiplies =
    bool_constant<std::is_same<binaryOp, sycl::multiplies<T>>::value ||
                  std::is_same<binaryOp, sycl::multiplies<void>>::value>;

template <typename T, class binaryOp>
using is_bit_and =
    bool_constant<std::is_same<binaryOp, sycl::bit_and<T>>::value ||
                  std::is_same<binaryOp, sycl::bit_and<void>>::value>;

template <typename T, class binaryOp>
using is_bit_or =
    bool_constant<std::is_same<binaryOp, sycl::bit_or<T>>::value ||
                  std::is_same<binaryOp, sycl::bit_or<void>>::value>;

template <typename T, class binaryOp>
using is_bit_xor =
    bool_constant<std::is_same<binaryOp, sycl::bit_xor<T>>::value ||
                  std::is_same<binaryOp, sycl::bit_xor<void>>::value>;

template <typename T, class binaryOp>
using is_maximum =
    bool_constant<std::is_same<binaryOp, sycl::maximum<T>>::value ||
                  std::is_same<binaryOp, sycl::maximum<void>>::value>;

template <typename T, class binaryOp>
using is_minimum =
    bool_constant<std::is_same<binaryOp, sycl::minimum<T>>::value ||
                  std::is_same<binaryOp, sycl::minimum<void>>::value>;

// maximum is currently excluded from this due to incorrect behaviour with
// fetch_max, however should be reinstated once that is addressed.
template <typename T, class binaryOp>
using is_atomic_fetch_available = bool_constant<
    (is_plus<T, binaryOp>::value || is_bit_and<T, binaryOp>::value ||
     is_bit_or<T, binaryOp>::value || is_bit_xor<T, binaryOp>::value ||
     is_minimum<T, binaryOp>::value /* || is_maximum<T, binaryOp>::value */) &&
    std::is_integral_v<T>>;

template <typename reductionT>
struct has_atomic_fetch {
  static constexpr bool value =
      is_atomic_fetch_available<typename reductionT::value_type,
                                typename reductionT::binary_operation>::value;
};

template <typename T, typename BinaryOp>
using has_zero_identity =
    bool_constant<(std::is_integral_v<T> && (is_bit_or<T, BinaryOp>::value ||
                                             is_bit_xor<T, BinaryOp>::value)) ||
                  is_plus<T, BinaryOp>::value>;

template <typename T, typename BinaryOp>
using has_one_identity = bool_constant<is_multiplies<T, BinaryOp>::value>;

template <typename T, typename BinaryOp>
using has_bit_ones_identity =
    bool_constant<std::is_integral_v<T> && is_bit_and<T, BinaryOp>::value>;

template <typename T, typename BinaryOp>
using has_minimum_identity = bool_constant<is_minimum<T, BinaryOp>::value>;

template <typename T, typename BinaryOp>
using has_maximum_identity = bool_constant<is_maximum<T, BinaryOp>::value>;

template <typename T, typename BinaryOp>
using has_known_identity_impl =
    bool_constant<has_zero_identity<T, BinaryOp>::value ||
                  has_one_identity<T, BinaryOp>::value ||
                  has_bit_ones_identity<T, BinaryOp>::value ||
                  has_minimum_identity<T, BinaryOp>::value ||
                  has_maximum_identity<T, BinaryOp>::value>;

/** This struct is used to get the correct identity value for different binary
 * operation / data type combinations.
 * @tparam T Data type of the operation
 * @tparam BinaryOp The binary operation to be used.
 */
template <typename T, typename BinaryOp, typename = void>
struct known_identity_helper {};

/** Only available if the binary operation / data type combination has an
 * identity of zero and the data type is not half.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_zero_identity<T, BinaryOp>::value &&
                              !std::is_same_v<T, cl::sycl::half>>> {

  static constexpr T value = T{0};
};

/** Only available if the binary operation / data type combination has an
 * identity of zero and the data type is half. Specialization for half type due
 * to difference between half on host and device.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_zero_identity<T, BinaryOp>::value &&
                              std::is_same_v<T, cl::sycl::half>>> {

#ifdef __SYCL_DEVICE_ONLY__
  // Need to ignore this warning or the compiler complains that we have not
  // defined cl_khr_fp16 but that should be defined in user code.
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wsycl-kernel-spec")
  static constexpr half value = 0;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)  // -Wsycl-kernel-spec
#else
  static constexpr half value =
      cl::sycl::half(cl::sycl::half::value_tag{}, static_cast<uint16_t>(0));
#endif
};

/** Only available if the binary operation / data type combination has an
 * identity of one and data type is not half.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_one_identity<T, BinaryOp>::value &&
                              !std::is_same_v<T, cl::sycl::half>>> {

  static constexpr T value = T{1};
};

/** Only available if the binary operation / data type combination has an
 * identity of one and data type is half. Specialization for half type due to
 * difference between half on host and device.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_one_identity<T, BinaryOp>::value &&
                              std::is_same_v<T, cl::sycl::half>>> {

#ifdef __SYCL_DEVICE_ONLY__
  // Need to ignore this warning or the compiler complains that we have not
  // defined cl_khr_fp16 but that should be defined in user code.
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(push)
  COMPUTECPP_CLANG_CXX_DIAGNOSTIC(ignored "-Wsycl-kernel-spec")
  static constexpr half value = 1;
  COMPUTECPP_HOST_CXX_DIAGNOSTIC(pop)  // -Wsycl-kernel-spec
#else
  static constexpr half value = cl::sycl::half(cl::sycl::half::value_tag{},
                                               static_cast<uint16_t>(0x3C00));
#endif
};

/** Only available if the binary operation / data type combination has an
 * identity which is a bit representation which consists of all ones.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_bit_ones_identity<T, BinaryOp>::value>> {

  static constexpr T value = ~static_cast<T>(0);
};

/** Only available if the binary operation / data type combination has an
 * identity which is the maximum possible number for that type.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_minimum_identity<T, BinaryOp>::value>> {

  static constexpr T value = std::numeric_limits<T>::max();
};

/** Only available if the binary operation / data type combination has an
 * identity which is the lowest possible number for that type.
 */
template <typename T, typename BinaryOp>
struct known_identity_helper<
    T, BinaryOp,
    typename std::enable_if_t<has_maximum_identity<T, BinaryOp>::value>> {

  static constexpr T value = std::numeric_limits<T>::lowest();
};

/** Helper class for extracting elements from the input tuple
 * @tparam Ts Types of elements in the input tuple
 */
template <class... Ts>
struct extract_tuple_helper {
  /** Extracts elements from the input tuple, applies the @ref TransformF
   * operation on each of them, and returns the result in a new tuple.
   * @tparam TransformF Type of the transform to apply
   * @tparam Is Sequence used when indexing the input tuple
   * @param value Input tuple
   * @param transform Object representing the transform operation
   * @return New tuple containing extracted and transformed elements
   */
  template <class TransformF, std::size_t... Is>
  static auto get(std::tuple<Ts...> value, TransformF transform,
                  std::index_sequence<Is...>) {
    return std::tuple{transform(std::move(std::get<Is>(value)))...};
  }
};

/** Extracts first @ref tupleSize elements from the input tuple, applies the
 * @ref TransformF operation on each of them, and returns the result in a new
 * tuple.
 *
 * @tparam tupleSize Number of elements to extract from input tuple
 * @tparam TransformF Type of the transform to apply
 * @tparam Ts Types of elements in the input tuple
 * @param value Input tuple
 * @param transform Object representing the transform operation
 * @return New tuple containing extracted and transformed elements
 */
template <size_t tupleSize, class TransformF = detail::identity, class... Ts>
auto extract_tuple(std::tuple<Ts...> value, TransformF transform = {}) {
  return extract_tuple_helper<Ts...>::get(
      std::move(value), transform, std::make_index_sequence<tupleSize>());
}

/** Detail reduction object
 * @tparam isUSM True if unified shared memory is used
 * @tparam dataT Underlying data type
 * @tparam dims Number of reduction dimensions
 * @tparam BinaryOperation Type of the reduction operation
 */

template <bool isUSM, class dataT, int dims, class BinaryOperation>
class reduction_impl;

template <class... Ts>
struct extract_reduction_impl {
  static constexpr const auto tupleSize = sizeof...(Ts);
  static auto get(std::tuple<Ts...> value) {
    return detail::extract_tuple<tupleSize - 1>(std::move(value));
  }
};

template <class functorT>
struct extract_reduction_impl<functorT> {
  static std::tuple<> get(std::tuple<functorT>) { return {}; }
};

#endif  // SYCL_LANGUAGE_VERSION >= 202002

/** Calculate the maximum allowed workgroup size for reductions. Makes a
 * conservative estimate of local memory usage to try and avoid exhausting
 * resources.
 * @param queue shared_ptr to detail queue object.
 * @param localMemPerWorkItem The amount of local memory used by each work item
 * in the reduction kernel itself.
 * @return The calculated max workgroup size.
 */
COMPUTECPP_EXPORT size_t reduction_get_max_wg_size(dqueue_shptr queue,
                                                   size_t localMemPerWorkItem);

/** Overrides the workgroup size if the user has the configuration option
 * reduction_workgroup_size.
 * @param maxWgSize The existing max wg size value.
 * @returns Either maxWgSize or the configuration option value.
 */
COMPUTECPP_EXPORT size_t adjust_reduction_wg_size(size_t maxWgSize);

}  // namespace detail

#if SYCL_LANGUAGE_VERSION >= 202002

namespace property {
namespace reduction {

/** The initialize_to_identity property adds the requirement that the
 * reduction should initialize the provided user memory to some known identity
 * value before performing the reduction. Otherwise any value already in the
 * memory will be included.
 **/
class COMPUTECPP_EXPORT initialize_to_identity : public detail::property_base {
 public:
  initialize_to_identity()
      : detail::property_base(detail::property_enum::initialize_to_identity) {}
};

}  // namespace reduction
}  // namespace property

namespace detail {

/** Base object with common values and operations for all reducer objects
 * @tparam T Underlying data type
 * @tparam BinaryOperation Type of the reduction operation
 */

template <typename T, typename BinaryOperation>
class reducer_base {

 public:
  /* Only available if Dimensions == 0 */
  void combine(const T& partial) {
    m_value = BinaryOperation{}(m_value, partial);
  }

  /* Only available if identity value is known */
  constexpr T identity() const {
    return known_identity_helper<T, BinaryOperation>::value;
  }

 protected:
  /** Sets the internal value of the reducer directly. Unavailable in
   * user kernels.
   * @param value The new value to be set.
   */
  void set_value_impl(T value) { this->m_value = value; }
  /** Gets the internal value of the reducer. Unavailable in
   * user kernels.
   * @return The internal reducer value.
   */
  T get_value_impl() { return this->m_value; }

 private:
  T m_value = known_identity_helper<T, BinaryOperation>::value;
};

}  // namespace detail

/** Reducer object which performs a binary operation and accumulates
 * values.
 * @tparam T Data type to use for the reduction.
 * @tparam BinaryOperation The op to use for reduction.
 */
template <typename T, typename BinaryOperation>
class reducer : public detail::reducer_base<T, BinaryOperation> {};

/** Reducer object specialization for sycl::plus<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, plus<T>> : public detail::reducer_base<T, plus<T>> {
 public:
  reducer& operator+=(const T& partial) {
    this->combine(partial);
    return *this;
  }

  /* Only available for integral types */
  reducer& operator++() {
    this->combine(static_cast<T>(1));
    return *this;
  }

 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_add(this->get_value_impl());
  }
};

/** Reducer object specialization for sycl::multiplies<T> binary
 * operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, multiplies<T>>
    : public detail::reducer_base<T, multiplies<T>> {
 public:
  reducer& operator*=(const T& partial) {
    this->combine(partial);
    return *this;
  }
};

/** Reducer object specialization for sycl::bit_and<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, bit_and<T>> : public detail::reducer_base<T, bit_and<T>> {
 public:
  /* Only available for integral types */
  reducer& operator&=(const T& partial) {
    this->combine(partial);
    return *this;
  }

 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_and(this->get_value_impl());
  }
};

/** Reducer object specialization for sycl::bit_or<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, bit_or<T>> : public detail::reducer_base<T, bit_or<T>> {
 public:
  /* Only available for integral types */
  reducer& operator|=(const T& partial) {
    this->combine(partial);
    return *this;
  }

 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_or(this->get_value_impl());
  }
};

/** Reducer object specialization for sycl::bit_xor<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, bit_xor<T>> : public detail::reducer_base<T, bit_xor<T>> {
 public:
  /** Only available for integral types */
  reducer& operator^=(const T& partial) {
    this->combine(partial);
    return *this;
  }

 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_xor(this->get_value_impl());
  }
};

/** Reducer object specialization for sycl::minimum<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, minimum<T>> : public detail::reducer_base<T, minimum<T>> {
 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_min(this->get_value_impl());
  }
};

/** Reducer object specialization for sycl::maximum<T> binary operation.
 * @tparam T Data type to use for the reduction.
 */
template <typename T>
class reducer<T, maximum<T>> : public detail::reducer_base<T, maximum<T>> {
 protected:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space>
  void atomic_combine_impl(multi_ptr<T, space> partial) {
    atomic<T, space>(partial).fetch_max(this->get_value_impl());
  }
};
#undef COMPUTECPP_REDUCER_CONSTRUCTORS

namespace detail {

/** Wraps a reducer class and provides access to additional
 * functionality required during the internal reduction kernels. This lets
 * us do things with the reducer that are not exposed to the user in their
 * kernels, such as setting the reducer value.
 * @tparam T Data type of the reduction operation
 * @tparam BinaryOperation The operation to be used in the reduction.
 */
template <typename T, typename BinaryOperation>
class reducer_wrapper : public reducer<T, BinaryOperation> {
 public:
  /** Takes a pointer to some device memory and atomically combines it
   * with the reducer's local value with the result being stored in partial.
   * @tparam space The address space of the pointer being passed in.
   * @param partial A pointer to the value of type T in device memory.
   */
  template <access::address_space space = access::address_space::global_space>
  void atomic_combine(multi_ptr<T, space> partial) {
    this->template atomic_combine_impl<space>(partial);
  }
  /** Sets the underlying reducer value directly by calling protected
   * reducer class functions.
   * @param value The new reducer value to be set.
   */
  void set_value(T value) { this->set_value_impl(value); }

  /** Gets the underlying reducer value by calling protected
   * reducer class functions.
   * @return The internal reducer value.
   */
  T get_value() { return this->get_value_impl(); }
};

/** This class encapsulates the provided user storage for the reduction
 * results and provides some convenient types and methods for interacting with
 * reductions.
 * @tparam isUSM True if unified shared memory is used
 * @tparam dataT The data type of the reduction result/operation
 * @tparam dims The number of dimensions this reduction is operating over.
 * @tparam BinaryOperation The operation to use when reducing, must be a STL
 * type like std::plus etc.
 */
template <bool isUSM, class dataT, int dims, class BinaryOperation>
class reduction_impl {
 public:
  static constexpr const auto dimensions = dims;
  static constexpr bool isUsm = isUSM;
  static_assert(dimensions <= 1, "Multi-dimensional reductions (dims > 1) are "
                                 "not yet supported by the implementation.");
  static constexpr int bufferDims = (dimensions == 0) ? 1 : dimensions;
  static constexpr const bool hasAtomics =
      is_atomic_fetch_available<dataT, BinaryOperation>::value;

  using data_t = dataT;
  using reducer_t = reducer<dataT, BinaryOperation>;
  using reducer_wrapper_t = reducer_wrapper<dataT, BinaryOperation>;
  using out_acc_t = sycl::accessor<dataT, bufferDims, access::mode::read_write,
                                   access::target::global_buffer,
                                   access::placeholder::false_t>;

  template <typename T>
  using rw_acc_t = sycl::accessor<T, bufferDims, access::mode::read_write,
                                  access::target::global_buffer>;
  template <typename T>
  using local_acc_t = sycl::accessor<T, bufferDims, access::mode::read_write,
                                     access::target::local>;

  using value_type = dataT;
  using binary_operation = BinaryOperation;

  template <typename AllocatorT>
  reduction_impl(buffer<dataT, bufferDims, AllocatorT>& buf, handler& cgh,
                 BinaryOperation, const property_list& propList = {})
      : m_userStorage{std::make_shared<out_acc_t>(buf, cgh)},
        m_initializeToIdentity(
            propList
                .has_property<property::reduction::initialize_to_identity>()) {}

  reduction_impl(dataT* ptr, BinaryOperation,
                 const property_list& propList = {})
      : m_userStorageUSM(ptr),
        m_initializeToIdentity(
            propList
                .has_property<property::reduction::initialize_to_identity>()) {}

  /** Gets the storage provided by the user for storing the result of the
   * reduction.  */
  auto get_user_storage() {
    if constexpr (isUSM) {
      return m_userStorageUSM;
    } else {
      return *m_userStorage;
    }
  }
  /** Creates buffer for and returns an accessor to the memory to store
   * the partial sums from each workgroup. If size == 1 this will be the same
   * memory that the user passed for the final result.
   * @param size The number of elements requested.
   * @param cgh SYCL command group handler.
   * @returns A read/write accessor to the memory.
   */
  rw_acc_t<dataT> get_partial_sums_acc(size_t size, handler& cgh) {
    // Only 1D reductions are supported so this is fine, will break for dims > 1
    // though.
    m_partialSumBuffer = buffer<dataT, bufferDims>(range<1>(size));
    return rw_acc_t<dataT>{m_partialSumBuffer, cgh};
  }

  /** Creates a buffer for and returns an accessor to memory to hold the
   * global counter for finished workgroups. Used only in reductions that use
   * the tree-reduction algorithm. ALways creates a 1D buffer of ints with size
   * == 1.
   * @param cgh SYCL command group handler.
   * @returns A read/write accessor to the memory.
   */
  rw_acc_t<int> get_workgroup_finished_acc(handler& cgh) {
    m_workGroupFinishedCounter = buffer<int, 1>(m_workGroupFinishedCounterData);
    m_workGroupFinishedCounter.set_write_back(false);
    return {m_workGroupFinishedCounter, cgh};
  }

  /** Retrieves the internal pointer for an output accessor.
   */
  static inline auto get_out_pointer(out_acc_t acc) {
    return acc.get_pointer().get();
  }

  /** USM pointer, simply return input
   */
  static inline auto get_out_pointer(dataT* ptr) { return ptr; }

  /** Get a read/write accessor for local memory of the specified size.
   * @tparam T The data type of the local accessor, defaults to the reduction
   * type.
   * @param size The number of elements of local memory required.
   * @param cgh SYCL command group handler.
   * @return Local memory read/write accessor
   */
  template <typename T = dataT>
  local_acc_t<T> get_rw_local_acc(size_t size, handler& cgh) {
    return {size, cgh};
  }

  bool initialize_to_identity() const { return m_initializeToIdentity; }

 private:
  /// The storage provided by the user for the result of the reduction.
  std::shared_ptr<out_acc_t> m_userStorage;

  /// The USM pointer provided by the user for the result of the reduction.
  dataT* m_userStorageUSM;

  /// Buffer storage used by the tree-reduction algorithms. Only used
  /// when the number of workgroups > 1
  buffer<dataT, bufferDims> m_partialSumBuffer;

  /// Buffer storage for an atomic counter used by the tree-reduction
  /// algorithms for synchronizing across workgroups.
  buffer<int, 1> m_workGroupFinishedCounter;

  /// Data used for initializing the workgroup sync counter.
  const std::array<int, 1> m_workGroupFinishedCounterData{0};

  /// Controls whether we intialize the user storage to 0 or use the data in
  /// there as part of the reduction. Set via property::initialize_to_identity.
  bool m_initializeToIdentity;
};

/** Calculates the range which the reduction kernel will operate over. Note that
 * workgroup size may be overriden by a configuration option.
 * @tparam dimensions Number of dimensions of the kernel.
 * @param inRange The range of the user's kernel.
 * @param maxWgSize The max work group size to allow for the reduction.
 * @return An nd_range<dimensions> of the reduction range.
 */
template <int dimensions>
nd_range<dimensions> get_reduction_range(const range<dimensions>& inRange,
                                         size_t maxWgSize) {
  size_t workItems = inRange.size();
  size_t workGroupSize = std::min(workItems, maxWgSize);
  // If the user has set the reduction_workgroup_size config option this will
  // override workgroupSize.
  workGroupSize = adjust_reduction_wg_size(workGroupSize);
  size_t numWorkGroups = workItems / workGroupSize;
  if (workItems % workGroupSize) {
    numWorkGroups++;
  }
  return {sycl::range<1>{numWorkGroups * workGroupSize},
          sycl::range<1>{workGroupSize}};
}

/** Loops over the user kernel function and executes it.
 * @tparam userFunctT Type of the user's kernel function.
 * @tparam dimensions Number of dimensions of the user's kernel.
 * @tparam reducerT Type of the reducer object.
 * @param range Range of the user's kernel.
 * @param reducer The reducer object to be passed to the user kernel.
 * @param ndItem nd_item from the wrapping reduction kernel.
 * @param userFunc The user's kernel to be run.
 */
template <typename reducerT, typename userFuncT, int dimensions>
void reduction_loop(const range<dimensions>& range, reducerT& reducer,
                    const nd_item<1>& ndItem, userFuncT& userFunc) {
  size_t start = ndItem.get_global_id(0);
  size_t end = range.size();
  size_t stride = ndItem.get_global_range(0);

  for (size_t i = start; i < end; i += stride) {
    userFunc(get_delinearized_id(range, i), reducer);
  }
}

/** Returns a lambda which wraps the kernel function provided by the user.
 * This new kernel runs the user function then performs the reduction. This
 * version uses local memory and is only available if atomic operations are
 * available for the provided binary operation.
 * @tparam dimensions Number of dimensions of the reduction.
 * @tparam userFuncT The type of the user's kernel.
 * @tparam reductionFuncT The reduction object type.
 * @tparam handlerT sycl::handler type.
 * @param cgh Ref to sycl::handler object.
 * @param userFunc The user's kernel.
 * @param reductionFunc The reduction object.
 * @param range The range of the user's kernel.
 * @param ndRange The nd_range of the reduction operation.
 * @returns A lambda function type.
 */
template <int dimensions, typename userFuncT, typename reductionFuncT>
auto get_reduction_kernel_atomics(handler& cgh, userFuncT userFunc,
                                  reductionFuncT reductionFunc,
                                  const range<dimensions>& range,
                                  const nd_range<dimensions>& ndRange) {
  size_t wgSize = ndRange.get_local_range().size();
  size_t numWorkgroups = ndRange.get_group_range().size();

  const bool initializeToIdentity = reductionFunc.initialize_to_identity();
  auto groupSum = reductionFunc.get_rw_local_acc(1, cgh);
  auto out = reductionFunc.get_user_storage();

  auto partialSums = reductionFunc.get_partial_sums_acc(numWorkgroups, cgh);
  auto numWorkgroupsFinished = reductionFunc.get_workgroup_finished_acc(cgh);
  auto doFinalWriteInLastWG =
      reductionFunc.template get_rw_local_acc<int>(1, cgh);

  auto binaryOP = typename reductionFuncT::binary_operation{};

  return [userFunc, range, groupSum, out, initializeToIdentity, partialSums,
          numWorkgroupsFinished, doFinalWriteInLastWG, binaryOP, numWorkgroups,
          wgSize](nd_item<1> id) {
    typename reductionFuncT::reducer_wrapper_t redu;

    reduction_loop<typename reductionFuncT::reducer_t>(range, redu, id,
                                                       userFunc);
    size_t linearID = id.get_local_linear_id();
    if (linearID == 0) {
      groupSum[0] = redu.identity();
    }
    group_barrier(id.get_group(), memory_scope::work_group);
    redu.template atomic_combine<access::address_space::local_space>(
        &groupSum[0]);
    group_barrier(id.get_group(), memory_scope::work_group);
    if (linearID == 0) {
      partialSums[id.get_group_linear_id()] = groupSum[0];

      auto numFinished = atomic<int, access::address_space::global_space>(
          numWorkgroupsFinished.get_pointer());
      numFinished.fetch_add(1);
      doFinalWriteInLastWG[0] =
          numFinished.load() == static_cast<int>(numWorkgroups);
      groupSum[0] = redu.identity();
    }
    group_barrier(id.get_group(), memory_scope::work_group);
    if (doFinalWriteInLastWG[0]) {
      for (size_t i = linearID; i < numWorkgroups; i += wgSize) {
        redu.set_value(partialSums[i]);
        redu.template atomic_combine<access::address_space::local_space>(
            &groupSum[0]);
      }
      group_barrier(id.get_group(), memory_scope::work_group);
      if (linearID == 0) {
        if (!initializeToIdentity) {
          groupSum[0] =
              binaryOP(groupSum[0], reductionFuncT::get_out_pointer(out)[0]);
        }
        reductionFuncT::get_out_pointer(out)[0] = groupSum[0];
      }
    }
  };
}

/** Returns a lambda which wraps the kernel function provided by the user.
 * This new kernel runs the user function then performs the reduction. This
 * version is only used if atomic operations are unavailable for the provided
 * binary operation.
 * @tparam dimensions Number of dimensions of the reduction.
 * @tparam userFuncT The type of the user's kernel.
 * @tparam reductionFuncT The reduction object type.
 * @tparam handlerT sycl::handler type.
 * @param cgh Ref to sycl::handler object.
 * @param userFunc The user's kernel.
 * @param reductionFunc The reduction object.
 * @param range The range of the user's kernel.
 * @param ndRange The nd_range of the reduction operation.
 * @returns A lambda function type.
 */
template <int dimensions, typename userFuncT, typename reductionFuncT>
auto get_reduction_kernel_no_atomics(handler& cgh, userFuncT userFunc,
                                     reductionFuncT reductionFunc,
                                     const range<dimensions>& range,
                                     const nd_range<dimensions>& ndRange) {
  size_t wgSize = ndRange.get_local_range().size();
  size_t numWorkgroups = ndRange.get_group_range().size();

  bool initializeToIdentity = reductionFunc.initialize_to_identity();
  auto localSums = reductionFunc.get_rw_local_acc(wgSize + 1, cgh);
  auto partialSums = reductionFunc.get_partial_sums_acc(numWorkgroups, cgh);
  auto out = reductionFunc.get_user_storage();
  auto numWorkgroupsFinished = reductionFunc.get_workgroup_finished_acc(cgh);
  auto doPartialSumInLastWG =
      reductionFunc.template get_rw_local_acc<int>(1, cgh);
  auto binaryOP = typename reductionFuncT::binary_operation{};

  return [userFunc, range, wgSize, numWorkgroups, localSums, partialSums, out,
          numWorkgroupsFinished, doPartialSumInLastWG, binaryOP,
          initializeToIdentity](nd_item<1> id) {
    typename reductionFuncT::reducer_wrapper_t redu;

    reduction_loop<typename reductionFuncT::reducer_t>(range, redu, id,
                                                       userFunc);
    size_t linearID = id.get_local_linear_id();
    localSums[linearID] = redu.get_value();
    if (linearID == 0) {
      localSums[wgSize] = redu.identity();
    }
    group_barrier(id.get_group(), memory_scope::work_group);

    /* Performs a tree-reduction inside each workgroup to reduce down to a
     * partial sum for each group. localSums[wgSize] reduces odd elements
     * when the step size is odd.
     */
    size_t stepSize = wgSize;
    for (size_t currentStep = stepSize / 2; currentStep > 0; currentStep /= 2) {
      if (linearID < currentStep) {
        localSums[linearID] =
            binaryOP(localSums[linearID], localSums[linearID + currentStep]);
      } else if (linearID == currentStep && (stepSize & 0x1)) {
        localSums[wgSize] =
            binaryOP(localSums[wgSize], localSums[stepSize - 1]);
      }
      group_barrier(id.get_group(), memory_scope::work_group);
      stepSize = currentStep;
    }

    /* Writes the partial sum to global memory. Uses atomic to sync between
     * workgroups.
     */
    if (linearID == 0) {
      auto value = binaryOP(localSums[0], localSums[wgSize]);

      partialSums[id.get_group_linear_id()] = value;

      auto numFinished = atomic<int, access::address_space::global_space>(
          numWorkgroupsFinished.get_pointer());
      numFinished.fetch_add(1);
      doPartialSumInLastWG[0] =
          numFinished.load() == static_cast<int>(numWorkgroups);
    }
    group_barrier(id.get_group(), memory_scope::work_group);

    /* Do a final reduction in the last workgroup, summing the
     * partial sums from other workgroups and writing out to the user memory.
     */
    if (doPartialSumInLastWG[0]) {
      auto localSum = redu.identity();
      for (size_t i = linearID; i < numWorkgroups; i += wgSize) {
        localSum = binaryOP(localSum, partialSums[i]);
      }
      localSums[linearID] = localSum;
      if (linearID == 0) {
        localSums[wgSize] = redu.identity();
      }
      group_barrier(id.get_group(), memory_scope::work_group);

      stepSize = wgSize;
      for (size_t currentStep = stepSize / 2; currentStep > 0;
           currentStep /= 2) {
        if (linearID < currentStep) {
          localSums[linearID] =
              binaryOP(localSums[linearID], localSums[linearID + currentStep]);
        } else if (linearID == currentStep && (stepSize & 0x1)) {
          localSums[wgSize] =
              binaryOP(localSums[wgSize], localSums[stepSize - 1]);
        }
        group_barrier(id.get_group(), memory_scope::work_group);
        stepSize = currentStep;
      }
      if (linearID == 0) {
        auto value = binaryOP(localSums[0], localSums[wgSize]);
        if (!initializeToIdentity) {
          value = binaryOP(value, reductionFuncT::get_out_pointer(out)[0]);
        }
        reductionFuncT::get_out_pointer(out)[0] = value;
      }
    }
  };
}

/** Returns a lambda which wraps the kernel function provided by the user.
 * This new kernel runs the user function then performs the reduction. Different
 * kernels are returned depending on whether atomics are supported for the
 * current operation/type.
 * @tparam dimensions Number of dimensions of the reduction.
 * @tparam userFuncT The type of the user's kernel.
 * @tparam reductionFuncT The reduction object type.
 * @tparam handlerT sycl::handler type.
 * @param cgh Ref to sycl::handler object.
 * @param userFunc The user's kernel.
 * @param reductionFunc The reduction object.
 * @param range The range of the user's kernel.
 * @returns A lambda function type.
 */
template <int dimensions, typename userFuncT, typename reductionFuncT>
auto get_reduction_kernel(handler& cgh, userFuncT userFunc,
                          reductionFuncT reductionFunc,
                          const range<dimensions>& range,
                          const nd_range<dimensions>& ndRange) {
  if constexpr (reductionFuncT::hasAtomics) {
    return get_reduction_kernel_atomics(cgh, userFunc, reductionFunc, range,
                                        ndRange);
  } else {
    return get_reduction_kernel_no_atomics(cgh, userFunc, reductionFunc, range,
                                           ndRange);
  }
}
}  // namespace detail

/** Constructs and returns a reduction object.
 * @tparam BufferT The sycl::buffer type.
 * @tparam BinaryOperation The binary operation type, such as std/sycl::plus.
 * @param vars The sycl buffer object which will hold the output of the
 * reduction.
 * @param cgh Reference to a SYCL handler object.
 * @param combiner The binary operation object.
 * @param propList optional property list.
 * @return A reduction object
 */
template <typename BufferT, typename BinaryOperation>
detail::reduction_impl<false, typename BufferT::value_type, BufferT::dimensions,
                       BinaryOperation>
reduction(BufferT vars, handler& cgh, BinaryOperation combiner,
          const property_list& propList = {}) {
  return {vars, cgh, combiner, propList};
}

/** Constructs and returns a reduction object using a USM pointer.
 * @tparam T The data type of the USM pointer.
 * @tparam BinaryOperation The binary operation type, such as std/sycl::plus.
 * @param ptr The USM pointer to memory which will hold the output of the
 * reduction.
 * @param cgh Reference to a SYCL handler object.
 * @param combiner The binary operation object.
 * @param propList optional property list.
 * @return A reduction object
 */
template <typename T, typename BinaryOperation>
detail::reduction_impl<true, T, 0, BinaryOperation> reduction(
    T* ptr, BinaryOperation combiner, const property_list& propList = {}) {
  return {ptr, combiner, propList};
}

/** Trait class which defines a value for the identity of a binary operation /
 * accumulator type combination when known.
 * @tparam BinaryOperation The binary operation to be performed.
 * @tparam AccumulatorT The data type of the accumulation. */
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity
    : public detail::known_identity_helper<AccumulatorT, BinaryOperation> {};

/** Helper shortcut for known_identity.  */
template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

/** Trait class which defines a boolean value which is true if the identity of
 * the given binary operation / accumulator type combination is known.
 * @tparam BinaryOperation The binary operation to be performed.
 * @tparam AccumulatorT The data type of the accumulation. */
template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity {
  static constexpr bool value =
      detail::has_known_identity_impl<BinaryOperation, AccumulatorT>::value;
};

/** Helper shortcut for has_known_identity.  */
template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

#endif  // SYCL_LANGUAGE_VERSION >= 202002

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_REDUCTION_H_
