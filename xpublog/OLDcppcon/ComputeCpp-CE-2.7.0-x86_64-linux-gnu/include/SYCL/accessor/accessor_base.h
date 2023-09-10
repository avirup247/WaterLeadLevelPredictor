/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file accessor_base.h
  @brief Internal file used by the @ref cl::sycl::accessor class
*/

#ifndef RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_BASE_H_
#define RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_BASE_H_

#include "SYCL/accessor/accessor_args.h"
#include "SYCL/accessor/accessor_host_args.h"
#include "SYCL/common.h"
#include "SYCL/vec_types_defines.h"

namespace cl {
namespace sycl {

class handler;

/*******************************************************************************
    accessor_base (host side)
*******************************************************************************/

/** @cond COMPUTECPP_DEV */
/**

This file contains:
- @ref accessor_base

The accessor_base class has a completely different implementation for host side
and device side. For host side it contains the public interface functions and
the internal functionality of the accessor class that is required on the host
side as well as functions for reading and writing to images. It contains a
shared_ptr to the detail::accessor and a raw pointer to the host side that is
used in order to allow the accessor to point to it's data for the purposes of
setting kernel arguments. The implementation of the accessor_base is located in
accessor.cpp

See the host side accessor class structure diagram in the accessor.h page

On the device side it consists of dummy method declarations with no definitions
as the methods are never called.

The device side accessor class contains a device_arg_container struct, which
itself contains a device_arg struct (and an accessor_base struct), depending on
the specialization of the accessor template class.

See the device side accessor class structure diagram in the accessor.h page

*/

#ifndef __SYCL_DEVICE_ONLY__

/**
@brief The base class of accessor, contains the implementation of the host side
interface functions and the image read write functions that call into the image
library.
*/
class COMPUTECPP_EXPORT accessor_base {
 public:
  /**
    @brief Constructs an accessor_base for placeholder types.
    @param accessMode The access mode that the accessor is requesting.
    @param accessTarget The access target that the accessor is requesting.
  */
  accessor_base(access::mode accessMode, access::target accessTarget);

  /**
    @brief Constructs an accessor_base with a storage_mem reference. Base class
    constructor for buffer and image accessor specializations. Constructs the
    internal detail accessor with the arguments passed down from the accessor
    constructor.
    @param store The storage mem that the accessor is requesting access to.
    @param accessMode The access mode that the accessor is requesting.
    @param accessTarget The access target that the accessor is requesting.
    @param elementSize The size of an element of the accessor.
    @param commandHandler The command handler that the accessor is requesting
    access for.
  */
  accessor_base(storage_mem& store, access::mode accessMode,
                access::target accessTarget, size_t elementSize,
                handler& commandHandler);

  /**
  @brief Constructor that takes a storage_mem, access mode, access target,
  element size, command handler and an access range. It constructs an
  implementation accessor object that will give access to the storage_mem object
  provided, in the transaction associated with the command group with the access
  mode, target and range provided.
  @param store The storage mem that the accessor is requesting access to.
  @param accessMode The access mode that the accessor is requesting.
  @param accessTarget The access target that the accessor is requesting.
  @param elementSize The size of an element of the accessor.
  @param commandHandler The command handler that the accessor is requesting
  access for.
  @param accessRange The range of access that the accessor is requesting.
  */
  accessor_base(storage_mem& store, access::mode accessMode,
                access::target accessTarget, size_t elementSize,
                handler& commandHandler, detail::access_range accessRange);

  /**
  @brief Constructs an accessor_base with a storage_mem reference. Base class
  constructor for local accessor specializations. Constructs the internal detail
  accessor with the arguments passed down from the accessor constructor.
  */
  accessor_base(dim_t numDims, const detail::index_array& numElements,
                access::mode accessMode, access::target accessTarget,
                size_t elementSize, handler& commandHandler);

  /**
    @brief Constructs an accessor_base with a storage_mem reference. Base class
    constructor for host buffer and image accessor specializations. Constructs
    the internal detail accessor with the arguments passed down from the
    accessor constructor. Also used for placeholder accessors.
    @param store The storage mem that the accessor is requesting access to.
    @param accessMode The access mode that the accessor is requesting.
    @param accessTarget The access target that the accessor is requesting.
    @param elementSize The size of an element of the accessor.
  */
  accessor_base(storage_mem& store, access::mode accessMode,
                access::target accessTarget, size_t elementSize);

  /**
    @brief Constructs an accessor_base with a storage_mem reference. Base class
    constructor for host buffer and image accessor specializations. Constructs
    the internal detail accessor with the arguments passed down from the
    accessor constructor. Also used for placeholder accessors.
    @param store The storage mem that the accessor is requesting access to.
    @param accessMode The access mode that the accessor is requesting.
    @param accessTarget The access target that the accessor is requesting.
    @param elementSize The size of an element of the accessor.
    @param accessRange The access range used with accessor arithmetic's
  */
  accessor_base(storage_mem& store, access::mode accessMode,
                access::target accessTarget, size_t elementSize,
                detail::access_range accessRange);

  /**
  @brief Copies one accessor_base to another, creating a new reference to the
  internal detail accessor.
  @param rhs Reference to the accessor_base being copied.
  */
  accessor_base(const accessor_base& rhs);

  /**
  @brief Moves one accessor_base to another, moving the reference to the
  internal detail accessor to the new accessor_base, invalidating the old
  reference.
  @param rhs Reference to the accessor_base being moved.
  */
  accessor_base(accessor_base&& rhs) noexcept;

  /**
  @brief Copy Assigns one accessor_base to another, creating a new reference to
  the internal detail accessor.
  @param rhs Reference to the accessor_base being copied.
  @return Reference to the new accessor_base.
  */
  accessor_base& operator=(const accessor_base& rhs);

  /**
  @brief Copy assigns one accessor_base to another, moving the reference to the
  internal detail accessor to the new accessor_base, invalidating the old
  reference.
  @param rhs Reference to the accessor_base being moved.
  @return Reference to the new accessor_base.
  */
  accessor_base& operator=(accessor_base&& rhs) noexcept;

  /**
  @brief Default destructor.
  */
  ~accessor_base() = default;

 private:
  size_t size_impl() const;

  size_t byte_size_impl() const;

 public:
  /**
  @brief Gets the number of elements that the accessor can access. Host only.
  @return The number of elements that the accessor can access.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use accessor::size() instead.")
  size_t get_count() const;

  /**
  @brief Gets the number of elements the accessor can access. Host only.
  @return The number of elements the accessor can access.
  */
  COMPUTECPP_DEPRECATED_BY_SYCL_202001("Use accessor::byte_size() instead.")
  size_t get_size() const;

#if SYCL_LANGUAGE_VERSION >= 202001
  /**
  @brief Gets the number of elements that the accessor can access. Host only.
  @return The number of elements that the accessor can access.
  */
  size_t size() const noexcept { return size_impl(); }

  /**
  @brief Gets the number of elements the accessor can access. Host only.
  @return The number of elements the accessor can access.
  */
  size_t byte_size() const noexcept { return byte_size_impl(); }
#endif  // SYCL_LANGUAGE_VERSION >= 202001

  /**
  @brief Gets the range of the memory the accessor can access. Host only.
  @return The the range of the memory the accessor can access.
  */
  detail::index_array get_range() const;

  /**
  @brief Gets the offset of the memory the accessor can access. Host only.
  @return The the offset of the memory the accessor can access.
  */
  detail::index_array get_offset() const;

  /** @brief This function returns if the accessor is bound to a
   * memory object
   */
  bool is_null() const;

 protected:
  /**
  @brief Initializes m_hostDataPtr with the host data from the implementation
         object. Required in constructors.
  */
  void initialize_host_data();

  /**
  @brief Returns the storage associated with the accessor.
  @return Storage associated with the accessor.
  */
  const dmem_shptr& get_store() const noexcept;

  /** @brief Retrieves the full range of the associated storage.
   * @return Full storage range
   */
  inline detail::index_array get_store_range() const noexcept {
    return m_host_args.m_storeRange;
  }

  /**
  @brief Returns the access range of the accessor.
  @return Access range of the accessor.
  */
  detail::access_range get_access_range() const;

 public:
  /**
  @brief Reads a float4 from a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int coords) const;

  /**
  @brief Reads a float4 from a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int2 coords) const;

  /**
  @brief Reads a float4 from a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int4 coords) const;

  /**
  @brief Reads a half4 from a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int coords) const;

  /**
  @brief Reads a half4 from a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int2 coords) const;

  /**
  @brief Reads a half4 from a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int4 coords) const;

  /**
  @brief Reads a float4 from a 1 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float coords) const;

  /**
  @brief Reads a float4 from a 2 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float2 coords) const;

  /**
  @brief Reads a float4 from a 3 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float4 coords) const;

  /**
  @brief Reads a int4 from a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int coords) const;

  /**
  @brief Reads a int4 from a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int2 coords) const;

  /**
  @brief Reads a int4 from a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int4 coords) const;

  /**
  @brief Reads a int4 from a 1 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float coords) const;

  /**
  @brief Reads a int4 from a 2 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float2 coords) const;

  /**
  @brief Reads a int4 from a 3 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float4 coords) const;

  /**
  @brief Reads a uint4 from a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int coords) const;

  /**
  @brief Reads a uint4 from a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int2 coords) const;

  /**
  @brief Reads a uint4 from a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int4 coords) const;

  /**
  @brief Reads a uint4 from a 1 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float coords) const;

  /**
  @brief Reads a uint4 from a 2 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float2 coords) const;

  /**
  @brief Reads a uint4 from a 3 dimensional image using floating point
  coordinates. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float4 coords) const;

  /**
  @brief Reads a float4 from a 1 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int coords, sampler smpl) const;

  /**
  @brief Reads a float4 from a 2 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int2 coords, sampler smpl) const;

  /**
  @brief Reads a float4 from a 3 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_int4 coords, sampler smpl) const;

  /**
  @brief Reads a float4 from a 1 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float coords, sampler smpl) const;

  /**
  @brief Reads a float4 from a 2 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float2 coords, sampler smpl) const;

  /**
  @brief Reads a float4 from a 3 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_float4 readf(cl::sycl::cl_float4 coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 1 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 2 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int2 coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 3 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_int4 coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 1 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_float coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 2 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The float4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_float2 coords, sampler smpl) const;

  /**
  @brief Reads a half4 from a 3 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The half4 value read from the image.
  */
  cl::sycl::cl_half4 readh(cl::sycl::cl_float4 coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 1 dimensional image using integer coordinates and a
  sampler. Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 2 dimensional image using integer coordinates and a
  sampler. Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int2 coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 3 dimensional image using integer coordinates and a
  sampler. Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_int4 coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 1 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 2 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float2 coords, sampler smpl) const;

  /**
  @brief Reads a int4 from a 3 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The int4 value read from the image.
  */
  cl::sycl::cl_int4 readi(cl::sycl::cl_float4 coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 1 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 2 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int2 coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 3 dimensional image using integer coordinates and
  a sampler. Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_int4 coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 1 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 1 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 2 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 2 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float2 coords, sampler smpl) const;

  /**
  @brief Reads a uint4 from a 3 dimensional image using floating point
  coordinates and a sampler. Image accessors only.
  @param coords 4 dimensional floating point coordinates for reading the image.
  @param smpl The sampler being used for reading the image.
  @return The uint4 value read from the image.
  */
  cl::sycl::cl_uint4 readui(cl::sycl::cl_float4 coords, sampler smpl) const;

  /**
  @brief Writes a float4 to a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param value The float4 value to be written to the image.
  */
  void writef(cl::sycl::cl_int coords, cl::sycl::cl_float4 value) const;

  /**
  @brief Writes a float4 to a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param value The float4 value to be written to the image.
  */
  void writef(cl::sycl::cl_int2 coords, cl::sycl::cl_float4 value) const;

  /**
  @brief Writes a float4 to a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param value The float4 value to be written to the image.
  */
  void writef(cl::sycl::cl_int4 coords, cl::sycl::cl_float4 value) const;

  /**
  @brief Writes a half4 to a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param value The half4 value to be written to the image.
  */
  void writeh(cl::sycl::cl_int coords, cl::sycl::cl_half4 value) const;

  /**
  @brief Writes a half4 to a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param value The half4 value to be written to the image.
  */
  void writeh(cl::sycl::cl_int2 coords, cl::sycl::cl_half4 value) const;

  /**
  @brief Writes a half4 to a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param value The half4 value to be written to the image.
  */
  void writeh(cl::sycl::cl_int4 coords, cl::sycl::cl_half4 value) const;

  /**
  @brief Writes a int4 to a 1 dimensional image using integer coordinates. Image
  accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param value The int4 value to be written to the image.
  */
  void writei(cl::sycl::cl_int coords, cl::sycl::cl_int4 value) const;

  /**
  @brief Writes a int4 to a 2 dimensional image using integer coordinates. Image
  accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param value The int4 value to be written to the image.
  */
  void writei(cl::sycl::cl_int2 coords, cl::sycl::cl_int4 value) const;

  /**
  @brief Writes a int4 to a 3 dimensional image using integer coordinates. Image
  accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param value The int4 value to be written to the image.
  */
  void writei(cl::sycl::cl_int4 coords, cl::sycl::cl_int4 value) const;

  /**
  @brief Writes a uint4 to a 1 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 1 dimensional integer coordinates for reading the image.
  @param value The uint4 value to be written to the image.
  */
  void writeui(cl::sycl::cl_int coords, cl::sycl::cl_uint4 value) const;

  /**
  @brief Writes a uint4 to a 2 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 2 dimensional integer coordinates for reading the image.
  @param value The uint4 value to be written to the image.
  */
  void writeui(cl::sycl::cl_int2 coords, cl::sycl::cl_uint4 value) const;

  /**
  @brief Writes a uint4 to a 3 dimensional image using integer coordinates.
  Image accessors only.
  @param coords 4 dimensional integer coordinates for reading the image.
  @param value The uint4 value to be written to the image.
  */
  void writeui(cl::sycl::cl_int4 coords, cl::sycl::cl_uint4 value) const;

  /**
  @brief Gets a shared_ptr to internal detail accessor. Host only.
  Implementation Detail.
  @return A shared_ptr to internal detail accessor.
  */
  inline const daccessor_shptr& get_impl() const { return m_host_args.m_impl; }

  /// @cond COMPUTECPP_DEV

  /** @brief Constructs an implementation accessor with no storage
   * @param mode Access mode
   * @param target Access target
   * @internal
   */
  void make_impl(access::mode mode, access::target target) const;

  /// COMPUTECPP_DEV @endcond

 protected:
  /**
  @brief Gets a raw pointer to the host memory of the accessor. Host only.
  Implementation Detail.
  @return A raw pointer to the host memory.
  */
  inline void* get_host_data() const { return m_host_args.m_hostDataPtr; }

  /**
  @brief Host side argument container.
  */
  mutable detail::host_arg_container m_host_args;
};

#endif

/*******************************************************************************
    accessor_base (device side)
*******************************************************************************/

#ifdef __SYCL_DEVICE_ONLY__

/**
@brief Device view of the accessor_base class. Because the device arg container
need to be templated, the device accessor_base needs to be as well. This allow
the host and device common accessor template class and the host side interface
to compile with the device compiler. There are no definitions for the
constructors or methods in this class, as they are never intended to be called.
*/
template <typename deviceArgsT>
class accessor_device_base {
 public:
  /** @brief Dummy placeholder constructor.
   */
  accessor_device_base(access::mode, access::target) {}

  /**
  @brief Dummy constructor for the buffer and image specializations of the
  accessor template class.
  */
  template <typename constructorT>
  accessor_device_base(constructorT, access::mode, access::target, size_t,
                       handler&) {}

  /**
  @brief Dummy constructor for the buffer and image specializations of the
  accessor template class.
  */
  template <typename constructorT>
  accessor_device_base(constructorT, access::mode, access::target, size_t,
                       handler&, detail::access_range) {}

  /**
  @brief Dummy constructor for the local specializations of the accessor
  template class.
  */
  accessor_device_base(dim_t, const detail::index_array&, access::mode,
                       access::target, size_t, handler&) {}

  /**
  @brief Dummy constructor for the host buffer and image specializations of the
  accessor template class.
  */
  template <typename constructorT>
  accessor_device_base(constructorT, access::mode, access::target, size_t) {}

  /**
  @brief Dummy constructor for the host buffer and image specializations of the
  accessor template class.
  */
  template <typename constructorT>
  accessor_device_base(constructorT, access::mode, access::target, size_t,
                       detail::access_range) {}

  /**
  @brief Dummy get_impl() method.
  */
  const daccessor_shptr& get_impl() const;

  /**
  @brief Dummy get_store() method.
  */
  dmem_shptr get_store() const;

  /**
  @brief get_store_range() method for the device.
  */
  detail::index_array get_store_range() const noexcept {
    return detail::index_array(m_deviceArgs.m_range);
  }

  /**
  @brief Dummy get_access_range() method.
  */
  detail::access_range get_access_range() const;

  /**@brief is_null() method for the device
   */
  inline bool is_null() const {
    return m_deviceArgs.m_deviceArg.get_ptr() == nullptr;
  }

 protected:
  /**
  @brief Device side argument container.
  */
  deviceArgsT m_deviceArgs;
};

// Make the device view looking like the host view.
// For practical reasons, the host side of accessor_base need to be non
// templated. So accessor_device_base act as accessor_base on device, and we
// alias accessor_base to some alias of accessor_device_base.
using accessor_base = accessor_device_base<void*>;

#endif

/******************************************************************************/

}  // namespace sycl
}  // namespace cl

/** COMPUTECPP_DEV @endcond */

/******************************************************************************/

#endif  // RUNTIME_INCLUDE_SYCL_ACCESSOR_ACCESSOR_BASE_H_

/******************************************************************************/
