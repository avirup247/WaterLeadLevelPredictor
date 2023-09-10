/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
 @file vec_load_store.h

 @brief This file contains the definitions of the function templates load and
for the @ref cl::sycl::vec class as defined by the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_VEC_LOAD_STORE_H_
#define RUNTIME_INCLUDE_SYCL_VEC_LOAD_STORE_H_

#ifdef __SYCL_DEVICE_ONLY__
#include "SYCL/builtins/device_builtins.h"
#endif  // __SYCL_DEVICE_ONLY__

#include "SYCL/common.h"

////////////////////////////////////////////////////////////////////////////////

namespace cl {
namespace sycl {

namespace detail {

#ifdef __SYCL_DEVICE_ONLY__

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for one-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 1>& inputVec, size_t /*offset*/,
           multi_ptr<const dataT, addressSpace> ptr) {
  inputVec.x() = *ptr;
}

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for two-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 2>& inputVec, size_t offset,
           multi_ptr<const dataT, addressSpace> ptr) {
  using vec_type = vec<dataT, 2>;
  // Using COMPUTECPP_BUILTIN_INVOKE2 to reduce the burden of ensuring the
  // correct casts are applied in the correct places.
  inputVec.set_data(COMPUTECPP_BUILTIN_INVOKE2(vload2, vec_type, offset, ptr));
}

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for three-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 3>& inputVec, size_t offset,
           multi_ptr<const dataT, addressSpace> ptr) {
  using vec_type = vec<dataT, 3>;
  // Using COMPUTECPP_BUILTIN_INVOKE2 to reduce the burden of ensuring the
  // correct casts are applied in the correct places.
  inputVec.set_data(COMPUTECPP_BUILTIN_INVOKE2(vload3, vec_type, offset, ptr));
}

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for four-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 4>& inputVec, size_t offset,
           multi_ptr<const dataT, addressSpace> ptr) {
  using vec_type = vec<dataT, 4>;
  // Using COMPUTECPP_BUILTIN_INVOKE2 to reduce the burden of ensuring the
  // correct casts are applied in the correct places.
  inputVec.set_data(COMPUTECPP_BUILTIN_INVOKE2(vload4, vec_type, offset, ptr));
}

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for eight-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 8>& inputVec, size_t offset,
           multi_ptr<const dataT, addressSpace> ptr) {
  using vec_type = vec<dataT, 8>;
  // Using COMPUTECPP_BUILTIN_INVOKE2 to reduce the burden of ensuring the
  // correct casts are applied in the correct places.
  inputVec.set_data(COMPUTECPP_BUILTIN_INVOKE2(vload8, vec_type, offset, ptr));
}

/**
@brief Function template for performing a vector load operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for sixteen-element vectors.
@tparam dataT The data type of the vec instance that is loaded to.
@tparam addressSpace The address space of the multi_ptr instance being loaded
from.
@param inputVec The vec instance being loaded to.
@param offset The offset in elements of dataT that is being loaded.
@param ptr The multi_ptr instance being loaded from.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vload(vec<dataT, 16>& inputVec, size_t offset,
           multi_ptr<const dataT, addressSpace> ptr) {
  using vec_type = vec<dataT, 16>;
  // Using COMPUTECPP_BUILTIN_INVOKE2 to reduce the burden of ensuring the
  // correct casts are applied in the correct places.
  inputVec.set_data(COMPUTECPP_BUILTIN_INVOKE2(vload16, vec_type, offset, ptr));
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for one-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 1>& inputVec, size_t /*offset*/,
            multi_ptr<dataT, addressSpace> ptr) {
  *ptr = inputVec.x();
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for two-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 2>& inputVec, size_t offset,
            multi_ptr<dataT, addressSpace> ptr) {
  ::cl::sycl::detail::vstore2(::cl::sycl::detail::cpp_to_cl_cast(inputVec),
                              offset, ::cl::sycl::detail::cpp_to_cl_cast(ptr));
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for three-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 3>& inputVec, size_t offset,
            multi_ptr<dataT, addressSpace> ptr) {
  ::cl::sycl::detail::vstore3(::cl::sycl::detail::cpp_to_cl_cast(inputVec),
                              offset, ::cl::sycl::detail::cpp_to_cl_cast(ptr));
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for four-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 4>& inputVec, size_t offset,
            multi_ptr<dataT, addressSpace> ptr) {
  ::cl::sycl::detail::vstore4(::cl::sycl::detail::cpp_to_cl_cast(inputVec),
                              offset, ::cl::sycl::detail::cpp_to_cl_cast(ptr));
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for eight-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 8>& inputVec, size_t offset,
            multi_ptr<dataT, addressSpace> ptr) {
  ::cl::sycl::detail::vstore8(::cl::sycl::detail::cpp_to_cl_cast(inputVec),
                              offset, ::cl::sycl::detail::cpp_to_cl_cast(ptr));
}

/**
@brief Function template for performing a vector store operation on an OpenCL
device using the OpenCL builtin function. This function template is overloaded
for sixteen-element vectors.
@tparam dataT The data type of the vec instance that is stored from.
@tparam addressSpace The address space of the multi_ptr instance being stored
to.
@param inputVec The vec instance being stored from.
@param offset The offset in elements of dataT that is being stored.
@param ptr The multi_ptr instance being stored to.
*/
template <typename dataT, cl::sycl::access::address_space addressSpace>
void vstore(const vec<dataT, 16>& inputVec, size_t offset,
            multi_ptr<dataT, addressSpace> ptr) {
  ::cl::sycl::detail::vstore16(::cl::sycl::detail::cpp_to_cl_cast(inputVec),
                               offset, ::cl::sycl::detail::cpp_to_cl_cast(ptr));
}

#endif

}  // namespace detail
}  // namespace sycl
}  // namespace cl

namespace cl {
namespace sycl {

/**
@brief Definition of vec::load member function template, declared in vec.h.
*/
template <typename dataT, int kElems>
template <access::address_space addressSpace>
void vec<dataT, kElems>::load(size_t offset,
                              multi_ptr<const dataT, addressSpace> ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::vload(*this, offset, ptr);
#else
  size_t ptrOffset = offset * kElems;
  for (int i = 0; i < kElems; i++) {
    this->m_data[i] = *(ptr + ptrOffset + i);
  }
#endif
}

/**
@brief Definition of vec::load member function template, declared in vec.h.
*/
template <typename dataT, int kElems>
template <int kDims, access::mode accessMode, access::target accessTarget>
void vec<dataT, kElems>::load(
    size_t offset, accessor<dataT, kDims, accessMode, accessTarget> acc) {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::vload(
      *this, offset,
      static_cast<multi_ptr<const dataT,
                            static_cast<access::address_space>(accessTarget)>>(
          acc.get_pointer()));
#else
  auto ptr = acc.get_pointer();
  size_t ptrOffset = offset * kElems;
  for (int i = 0; i < kElems; i++) {
    this->m_data[i] = *(ptr + ptrOffset + i);
  }
#endif
}

/**
@brief Definition of vec::load member function template, declared in vec.h.
*/
template <typename dataT, int kElems>
template <access::address_space addressSpace>
void vec<dataT, kElems>::store(size_t offset,
                               multi_ptr<dataT, addressSpace> ptr) const {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::vstore(*this, offset, ptr);
#else
  size_t ptrOffset = offset * kElems;
  for (int i = 0; i < kElems; i++) {
    *(ptr + ptrOffset + i) = this->m_data[i];
  }
#endif
}

/**
@brief Definition of vec::store member function template, declared in vec.h.
*/
template <typename dataT, int kElems>
template <int kDims, access::mode accessMode, access::target accessTarget>
void vec<dataT, kElems>::store(
    size_t offset, accessor<dataT, kDims, accessMode, accessTarget> acc) const {
#ifdef __SYCL_DEVICE_ONLY__
  ::cl::sycl::detail::vstore(
      *this, offset,
      static_cast<
          multi_ptr<dataT, static_cast<access::address_space>(accessTarget)>>(
          acc.get_pointer()));
#else
  auto ptr = acc.get_pointer();
  size_t ptrOffset = offset * kElems;
  for (int i = 0; i < kElems; i++) {
    *(ptr + ptrOffset + i) = this->m_data[i];
  }
#endif
}

}  // namespace sycl
}  // namespace cl

////////////////////////////////////////////////////////////////////////////////

#endif  // RUNTIME_INCLUDE_SYCL_VEC_LOAD_STORE_H_

////////////////////////////////////////////////////////////////////////////////
