/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_mem_container_storage_impl.h

 @brief This file contains the implementation of the base class for @ref
 cl::sycl::vec class.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_MEM_CONTAINER_STORAGE_IMPL_H_
#define RUNTIME_INCLUDE_SYCL_VEC_MEM_CONTAINER_STORAGE_IMPL_H_

#include "SYCL/vec_common.h"

#include <array>

////////////////////////////////////////////////////////////////////////////////

/* Vec
 ***
 */

namespace cl {
namespace sycl {

namespace detail {

#ifndef __SYCL_DEVICE_ONLY__

/** Base class for the vec and swizzled_vec classes and all intermediate
 * classes. Contains the vector data field. Host Only.
 * Alignment is used to match device side alignment.
 * @tparam dataT The data type.
 * @tparam kElems The number of elements.
 */
template <typename dataT, int kElems>
class alignas(sizeof(dataT) *
              (kElems == 3 ? 4 : kElems)) mem_container_storage {
 public:
  /**
  @brief Returns a const pointer to the vector data.
  @return A const pointer to the vector data.
  */
  inline const dataT* get_data() const;

  /**
  @brief Returns a pointer to the vector data.
  @return A pointer to the vector data.
  */
  inline dataT* get_data();

  /** Setter method for m_data.
   * @param rhs The vec reference.
   */
  inline void set_data(const vec<dataT, kElems>& rhs);

  /**
  @brief Returns the value at an index of the vector data.
  @param index the index of the vector to return.
  @return the value in the vector data at the index provided.
  */
  inline dataT get_value(int index) const;
  inline dataT get_value(int index, std::true_type) const;
  inline dataT get_value(int index, std::false_type) const;

  /**
  @brief Assigns a value to an index of the vector data.
  @param index the index of the vector to assign to.
  @param value the value to assign to the vector data at the index provided.
  */
  inline void set_value(int index, const dataT& value);
  inline void set_value(int index, const dataT& value, std::true_type);
  inline void set_value(int index, const dataT& value, std::false_type);

 protected:
  /**< Data storage, represented by a standard array on the host side. If the
   * number of elements is 3 the array needs to be padded on the host side due
   * to
   * implicit padding for the ext_vector type used by the __sycl_vector type.
   */
  static constexpr int sizeWithPadding = (kElems == 3) ? 4 : kElems;
  std::array<dataT, sizeWithPadding> m_data;
};

#else

template <typename dataT>
struct single_element_vec_storage {
  dataT x;
  operator dataT() { return this->x; }
  operator dataT() const { return this->x; }
  single_element_vec_storage& operator=(dataT rhs) {
    this->x = rhs;
    return *this;
  }
  inline single_element_vec_storage operator+(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x + rhs;
    return res;
  }
  inline single_element_vec_storage operator-(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x - rhs;
    return res;
  }
  inline single_element_vec_storage operator*(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x * rhs;
    return res;
  }
  inline single_element_vec_storage operator/(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x / rhs;
    return res;
  }
  inline single_element_vec_storage operator<<(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x << rhs;
    return res;
  }
  inline single_element_vec_storage operator>>(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x >> rhs;
    return res;
  }
  inline single_element_vec_storage operator&(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x & rhs;
    return res;
  }
  inline single_element_vec_storage operator|(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x | rhs;
    return res;
  }
  inline single_element_vec_storage operator&&(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x && rhs;
    return res;
  }
  inline single_element_vec_storage operator||(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x || rhs;
    return res;
  }
  inline single_element_vec_storage operator==(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x == rhs;
    return res;
  }
  inline single_element_vec_storage operator!=(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x != rhs;
    return res;
  }
  inline single_element_vec_storage operator>(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x > rhs;
    return res;
  }
  inline single_element_vec_storage operator<(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x < rhs;
    return res;
  }
  inline single_element_vec_storage operator<=(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x <= rhs;
    return res;
  }
  inline single_element_vec_storage operator>=(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x >= rhs;
    return res;
  }
  inline single_element_vec_storage operator^(const dataT rhs) const {
    single_element_vec_storage<dataT> res{};
    res.x = this->x ^ rhs;
    return res;
  }
  inline single_element_vec_storage operator-() {
    single_element_vec_storage<dataT> res{};
    res.x = -x;
    return res;
  }
  inline single_element_vec_storage operator~() {
    single_element_vec_storage<dataT> res{};
    res.x = ~x;
    return res;
  }
  inline single_element_vec_storage operator!() {
    single_element_vec_storage<dataT> res{};
    res.x = !x;
    return res;
  }
  single_element_vec_storage& operator+=(dataT rhs) {
    this->x += rhs;
    return *this;
  }
  single_element_vec_storage& operator-=(dataT rhs) {
    this->x -= rhs;
    return *this;
  }
  single_element_vec_storage& operator*=(dataT rhs) {
    this->x *= rhs;
    return *this;
  }
  single_element_vec_storage& operator/=(dataT rhs) {
    this->x /= rhs;
    return *this;
  }
  single_element_vec_storage& operator%=(dataT rhs) {
    this->x %= rhs;
    return *this;
  }
  single_element_vec_storage& operator&=(dataT rhs) {
    this->x &= rhs;
    return *this;
  }
  single_element_vec_storage& operator|=(dataT rhs) {
    this->x |= rhs;
    return *this;
  }
  single_element_vec_storage& operator^=(dataT rhs) {
    this->x ^= rhs;
    return *this;
  }
  single_element_vec_storage& operator>>=(dataT rhs) {
    this->x >>= rhs;
    return *this;
  }
  single_element_vec_storage& operator<<=(dataT rhs) {
    this->x <<= rhs;
    return *this;
  }
};

template <typename dataT, int kElems>
class mem_container_storage;

#define COMPUTECPP_MEM_CONTAINER_MIRROR_CONVERT(dataT, kElems)                 \
  [[computecpp::opencl_mirror_convert(mem_container_storage<dataT, kElems>)]]

/** Base class for the vec and swizzled_vec classes and all intermediate
 * classes. Contains the vector data field. Device Only.
 * @tparam dataT The data type.
 * @tparam kElems The number of elements.
 */
template <typename dataT, int kElems>
class COMPUTECPP_MEM_CONTAINER_MIRROR_CONVERT(dataT,
                                              kElems) mem_container_storage {
 public:
  /** Setter method for m_data.
   * @param rhs The vec reference.
   */
  inline void set_data(const vec<dataT, kElems>& rhs);

  /** Getter method for m_data. Device Only.
   * @return The m_data field.
   */
  inline detail::__sycl_vector<dataT, kElems> get_data() const;

  /** Setter method for m_data. Device Only.
   * @param rhs The __sycl_vector object.
   */
  inline void set_data(detail::__sycl_vector<dataT, kElems> rhs);

  /**
  @brief Returns the value at an index of the vector data.
  @param index the index of the vector to return.
  @return the value in the vector data at the index provided.
  */
  inline dataT get_value(int index) const;
  inline dataT get_value(int index, std::true_type) const;
  inline dataT get_value(int index, std::false_type) const;

  /**
  @brief Returns the address of the data.
  */
  detail::__sycl_vector<dataT, kElems>* get_data_ptr() noexcept {
    return &m_data;
  }

  /**
  @brief Returns the address of the data.
  */
  const detail::__sycl_vector<dataT, kElems>* get_data_ptr() const noexcept {
    return &m_data;
  }

  /**
  @brief Assigns a value to an index of the vector data.
  @param index the index of the vector to assign to.
  @param value the value to assign to the vector data at the index provided.
  */
  inline void set_value(int index, const dataT& value);
  inline void set_value(int index, const dataT& value, std::true_type);
  inline void set_value(int index, const dataT& value, std::false_type);

  /**< Data storage, represented by a __sycl_vector on the device side. */
  detail::__sycl_vector<dataT, kElems> m_data;
};

template <typename dataT>
class COMPUTECPP_MEM_CONTAINER_MIRROR_CONVERT(dataT, 1)
    mem_container_storage<dataT, 1> {
 public:
  /** Setter method for m_data.
   * @param rhs The vec reference.
   */
  inline void set_data(const vec<dataT, 1>& rhs);

  /** Getter method for m_data. Device Only.
   * @return The m_data field.
   */
  inline dataT get_data() const;

  /** Setter method for m_data. Device Only.
   * @param rhs The __sycl_vector object.
   */
  inline void set_data(dataT rhs);

  /**
  @brief Returns the value at an index of the vector data.
  @param index the index of the vector to return.
  @return the value in the vector data at the index provided.
  */
  inline dataT get_value(int index) const;
  inline dataT get_value(int index, std::true_type) const;
  inline dataT get_value(int index, std::false_type) const;

  /**
  @brief Assigns a value to an index of the vector data.
  @param index the index of the vector to assign to.
  @param value the value to assign to the vector data at the index provided.
  */
  inline void set_value(int index, const dataT& value);
  inline void set_value(int index, const dataT& value, std::true_type);
  inline void set_value(int index, const dataT& value, std::false_type);

  /**< Data storage, represented by a __sycl_vector on the device side. */
  single_element_vec_storage<dataT> m_data;
};

#undef COMPUTECPP_MEM_CONTAINER_MIRROR_CONVERT

#endif
}  // namespace detail
}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_MEM_CONTAINER_STORAGE_IMPL_H_

////////////////////////////////////////////////////////////////////////////////
