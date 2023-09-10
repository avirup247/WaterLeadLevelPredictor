/*****************************************************************************

    Copyright (C) 2002-2021 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/**
  @file info.h

  \brief This file implements the get info mechanism that is used for all SYCL
  class get_info methods.

  This is implemented as a generic function that can be applied to any OpenCL
  get info function with any OpenCL return type and SYCL return type. In order
  for this to work it does require some conversion functions to be implemented
  for the cases where the OpenCL type and the SYCL type do not match. The info
  parameters and host info definitions are all defined in the appropriate header
  file using macros that are defined here. The conversion functions are defined
  here for the common cases and in the appropriate header files for SYCL object
  specific info parameters.
 */

#ifndef RUNTIME_INCLUDE_SYCL_INFO_H_
#define RUNTIME_INCLUDE_SYCL_INFO_H_

#include "SYCL/cl_types.h"
#include "SYCL/common.h"
#include "SYCL/error_log.h"
#include "SYCL/include_opencl.h"
#include "SYCL/predefines.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>  // for istream_iterator
#include <limits>
#include <sstream>  // for stringstream
#include <string>
#include <type_traits>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

namespace info {
/**
 * @brief provides the class templates whose specializations will contain the
 * return_type field containing the type of the value returned by the
 * specializations of get_info().
 */
template <typename T, T param>
struct param_traits;
}  // namespace info

/** @cond COMPUTECPP_DEV */

/*******************************************************************************
    opencl_info_base
*******************************************************************************/

/**
@brief Specialized template struct that contains using declarations for the
OpenCL object and OpenCL function and a static OpenCL function pointer of the
type defined by the using declaration. Specializations of this class are defined
per SYCL get info type (info::device, info::platform, etc) using the
COMPUTECPP_DEFINE_SYCL_INFO_HANDLER macro.
@tparam syclInfo Specifies the SYCL info type.
*/
template <typename syclInfo>
struct opencl_info_base;

/*******************************************************************************
    sycl_host_info
*******************************************************************************/

/**
@brief Specialized template struct that contains a get method for retrieving the
host info definition for a specific OpenCL info parameter. Specializations of
this class are defined per OpenCL get info parameter using the
COMPUTECPP_DEFINE_SYCL_INFO_HOST macro.
@tparam syclType Specifies the SYCL return type.
@tparam clParam Specifies the OpenCL info parameter.
*/
template <typename syclType, cl_uint clParam>
struct sycl_host_info {
  /**
  @brief Static method that returns a host info parameter as a SYCL type.
  @return Host info parameter.
  */
  static syclType get() = delete;
};

/*******************************************************************************
    get_opencl_info
*******************************************************************************/

/**
@brief Specialized template function that wraps the OpenCL get info API
functions. The function is specialized based on the SYCL info type
(info::device, info::platform, etc) as they each require different OpenCL
objects and OpenCL function pointers, this information is specified in the
opencl_info_base struct. The specializations are explicitly instantiated in
info.cpp as they require internal functionality that cannot be included in the
header file. The parameters for this function match that of the OpenCL get info
APIS, so it can be used for both retrieving info sizes and the info parameters
them selves.
@tparam syclInfo Specifies the SYCL info type.
@param object The OpenCL object (cl_device_id, cl_context, etc).
@param param The OpenCL info parameter.
@param buffer A pointer to a buffer that the parameter can be returned in.
@param size The size of the parameter being returned.
@param returnSize A pointer to a size_t that can be used to return the size of a
parameter.
*/
template <typename syclInfo>
COMPUTECPP_EXPORT void get_opencl_info(
    const typename opencl_info_base<syclInfo>::cl_object& object,
    const cl_int& param, void* buffer, const size_t size, size_t* returnSize);

/*******************************************************************************
    COMPUTECPP_DEFINE_SYCL_INFO_HANDLER macro
*******************************************************************************/

/**
@brief Macro for defining the required template struct specialization for a
specific SYCL info type (info::device, info::platform, etc) as well as a SYCL
info type specific template (named opencl_##syclInfo##_info using the preprocess
stringification). This struct is intended to inherit from the opencl_info_base
class and is instantiated once per SYCL info parameter using the
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER macro. A specialization of the
param_traits is also defined in order to provide the type returned by the
get_info, in conformance with SYCL 1.2.1
@param syclInfo Specifies the SYCL info type.
@param clInfo Specifies the OpenCL info type.
@param clObject Specifies the OpenCL object type.
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(syclInfo, clInfo, clObject)        \
  template <>                                                                  \
  struct opencl_info_base<info::syclInfo> {                                    \
    using cl_object = clObject;                                                \
    using cl_function_type = cl_int(COMPUTECPP_CL_API_CALL*)(clObject, clInfo, \
                                                             size_t, void*,    \
                                                             size_t*);         \
    static cl_function_type cl_function;                                       \
  };                                                                           \
                                                                               \
  template <info::syclInfo syclParam>                                          \
  struct opencl_##syclInfo##_info;                                             \
                                                                               \
  namespace info {                                                             \
  template <info::syclInfo syclParam>                                          \
  struct param_traits<info::syclInfo, syclParam> {                             \
   public:                                                                     \
    using return_type =                                                        \
        typename opencl_##syclInfo##_info<syclParam>::sycl_type;               \
  };                                                                           \
  }

/*******************************************************************************
    COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER macro
    *******************************************************************************/

/**
@brief Macro for defining a specialization of one of the SYCL info type specific
structs for a specific SYCL info parameter. The struct is specialized to contain
using declarations for the SYCL return type and the OpenCL return type and a
static const unsigned int specifying the OpenCL info parameter.
@param syclInfo Specifies the SYCL info type.
@param syclParam Specifies the SYCL info parameter.
@param clParam Specifies the OpenCL info parameter.
@param syclType Specifies the SYCL return type.
@param clType Specifies the OpenCL return type.
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(syclInfo, syclParam, clParam,    \
                                              syclType, clType)                \
  template <>                                                                  \
  struct opencl_##syclInfo##_info<info::syclInfo::syclParam>                   \
      : public opencl_info_base<info::syclInfo> {                              \
    static const cl_uint cl_param = clParam;                                   \
    using sycl_type = syclType;                                                \
    using cl_type = clType;                                                    \
    static const cl_bitfield andValue =                                        \
        std::numeric_limits<cl_bitfield>::max();                               \
  };

/**
@brief Macro for defining a specialization of one of the SYCL info type specific
structs for a specific SYCL info parameter. The struct is specialized to contain
using declarations for the SYCL return type and the OpenCL return type and a
static const unsigned int specifying the OpenCL info parameter. _WITH_ANDVAL
variation includes a value with which the cl return value should be anded before
conversion to the SYCL type.
@param syclInfo Specifies the SYCL info type.
@param syclParam Specifies the SYCL info parameter.
@param clParam Specifies the OpenCL info parameter.
@param syclType Specifies the SYCL return type.
@param clType Specifies the OpenCL return type.
@param clBitfieldAndVal The value the clType should be anded with. Requires
clType is compatible with this.
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER_WITH_ANDVAL(                     \
    syclInfo, syclParam, clParam, syclType, clType, clBitfieldAndVal)          \
  template <>                                                                  \
  struct opencl_##syclInfo##_info<info::syclInfo::syclParam>                   \
      : public opencl_info_base<info::syclInfo> {                              \
    static const cl_uint cl_param = clParam;                                   \
    using sycl_type = syclType;                                                \
    using cl_type = clType;                                                    \
    static const cl_bitfield andValue = clBitfieldAndVal;                      \
  };

/*******************************************************************************
    COMPUTECPP_DEFINE_SYCL_INFO_HOST macro
    *******************************************************************************/

/**
@brief Macro for defining the host info for a specific SYCL info parameter
@param syclInfo Specifies the SYCL info type.
@param syclParam Specifies the SYCL info parameter.
@param returnValue the value to be returned when on host
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_HOST(syclInfo, syclParam, returnValue)     \
  template <>                                                                  \
  struct sycl_host_info<                                                       \
      typename opencl_##syclInfo##_info<info::syclInfo::syclParam>::sycl_type, \
      opencl_##syclInfo##_info<info::syclInfo::syclParam>::cl_param> {         \
    static opencl_##syclInfo##_info<info::syclInfo::syclParam>::sycl_type      \
    get() {                                                                    \
      return returnValue;                                                      \
    }                                                                          \
  };

/**
@brief Macro for declaring the host info for a specific SYCL info parameter.
Only the declaration will be present here, as we are unable to add the return
value, returned by the definition due to the way we include the header files..
it must be revisited after Bug #10470 is fixed.
@param syclInfo Specifies the SYCL info type.
@param syclParam Specifies the SYCL info parameter.
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_HOST_DECL(syclInfo, syclParam)             \
  template <>                                                                  \
  struct sycl_host_info<                                                       \
      typename opencl_##syclInfo##_info<info::syclInfo::syclParam>::sycl_type, \
      opencl_##syclInfo##_info<info::syclInfo::syclParam>::cl_param> {         \
    static COMPUTECPP_EXPORT                                                   \
        opencl_##syclInfo##_info<info::syclInfo::syclParam>::sycl_type         \
        get();                                                                 \
  };

/**
@brief Macro for defining the host info result for a specific SYCL info
parameter. Only the declaration will be present here, as we are unable to add
the return value, returned by the definition due to the way we include the
header files.. it must be revisited after Bug #10470 is fixed.
@param syclInfo Specifies the SYCL info type.
@param syclParam Specifies the SYCL info parameter.
@param returnValue The actual value returned
*/
#define COMPUTECPP_DEFINE_SYCL_INFO_HOST_DEF(syclInfo, syclParam, returnValue) \
  COMPUTECPP_EXPORT cl::sycl::opencl_##syclInfo##_info<                        \
      cl::sycl::info::syclInfo::syclParam>::sycl_type                          \
  cl::sycl::sycl_host_info<                                                    \
      cl::sycl::opencl_##syclInfo##_info<                                      \
          cl::sycl::info::syclInfo::syclParam>::sycl_type,                     \
      cl::sycl::opencl_##syclInfo##_info<                                      \
          cl::sycl::info::syclInfo::syclParam>::cl_param>::get() {             \
    return returnValue;                                                        \
  }

/**
 @internal
 @brief Macro for defining the host info directly for a specific SYCL info
 parameter
 @param syclType Specifies the SYCL return type.
 @param clParam Specifies the OpenCL info parameter.
 @param returnValue the value to be returned when on host
 */
#define COMPUTECPP_DEFINE_SYCL_DETAIL_INFO_HOST(syclType, clParam,             \
                                                returnValue)                   \
  template <>                                                                  \
  struct sycl_host_info<syclType, clParam> {                                   \
    static syclType get() { return returnValue; }                              \
  };

/** @internal
 * @brief Macro for declaring an explicit instantiation of the get_info function
 * @param className Class containing the get_info function
 * @param paramName Name of the parameter to instantiate get_info for
 */
#define COMPUTECPP_GET_INFO_INSTANTIATION(className, paramName)                \
  template COMPUTECPP_EXPORT typename cl::sycl::info::param_traits<            \
      cl::sycl::info::className,                                               \
      cl::sycl::info::className::paramName>::return_type                       \
  cl::sycl::className::get_info<cl::sycl::info::className::paramName>() const;

/** @internal
 * @brief Macro for declaring a specialization of the get_info function
 * @param className Class containing the get_info function
 * @param paramName Name of the parameter to specialize get_info for
 */
#define COMPUTECPP_GET_INFO_SPECIALIZATION_DECL(className, paramName)          \
  template <>                                                                  \
  COMPUTECPP_EXPORT typename cl::sycl::info::param_traits<                     \
      cl::sycl::info::className,                                               \
      cl::sycl::info::className::paramName>::return_type                       \
  cl::sycl::className::get_info<cl::sycl::info::className::paramName>() const;

/** @internal
 * @brief Macro for defining a specialization of the get_info function
 * @param className Class containing the get_info function
 * @param paramName Name of the parameter to specialize get_info for
 * @param getterBody Body of the function specialization
 */
#define COMPUTECPP_GET_INFO_SPECIALIZATION_DEF(className, paramName,           \
                                               getterBody)                     \
  template <>                                                                  \
  COMPUTECPP_EXPORT typename cl::sycl::info::param_traits<                     \
      cl::sycl::info::className,                                               \
      cl::sycl::info::className::paramName>::return_type                       \
  cl::sycl::className::get_info<cl::sycl::info::className::paramName>()        \
      const getterBody

/*******************************************************************************
    info_convert
*******************************************************************************/

/**
@brief Specialized template struct containing a function that performs a
conversion from an OpenCL return type to a corresponding SYCL return type. The
non-specialized function performs a conversion for when the OpenCL return type
and the SYCL return type is the same and contains a static_assert which triggers
an error if they are not, this allows errors when a conversion function is not
available. The function always takes a number of elements, however for scalar
conversions this is not used. The OpenCL type is always passed as a pointer,
this is in order to allow a generic mechanism, as some OpenCL get info API calls
return pointers.
@tparam clType Specifies the OpenCL return type.
@tparam syclType Specifies the SYCL return type.
*/
template <typename clType, typename syclType>
struct info_convert {
  static syclType cl_to_sycl(clType clPtr, size_t /*numElems*/,
                             cl_uint /*clParam*/) {
    static_assert(
        (std::is_same<clType, syclType*>::value),
        "SYCL type does not match OpenCL type, a conversion is required.");
    return *clPtr;
  }
};

/**
@brief Specialization of info_convert for converting a const char * type to a
vector_class<string_class> type.
@ref cl::sycl::info_convert
*/
template <>
struct COMPUTECPP_EXPORT info_convert<char*, vector_class<string_class>> {
  static vector_class<string_class> cl_to_sycl(const char* clPtr,
                                               size_t numElems,
                                               cl_uint clParam);
};

/**
@brief Specialization of info_convert for converting a pointer type to a
vector_class type.
@ref cl::sycl::info_convert
*/
template <typename clType, typename syclType>
struct info_convert<clType*, vector_class<syclType>> {
  static vector_class<syclType> cl_to_sycl(clType* clPtr, size_t numElems,
                                           cl_uint clParam) {
    vector_class<syclType> syclVector;
    for (size_t i = 0; i < numElems; i++) {
      syclVector.push_back(
          info_convert<clType*, syclType>::cl_to_sycl(&(clPtr[i]), 1, clParam));
    }
    return syclVector;
  }
};

/**
@brief Specialization of info_convert for converting a pointer type to a
bitset type
@ref cl::sycl::info_convert
*/
template <typename clType, size_t Size>
struct info_convert<clType*, bitset_class<Size>> {
  static bitset_class<Size> cl_to_sycl(clType* clPtr, size_t /*numElems*/,
                                       cl_uint /*clParam*/) {
    bitset_class<Size> retVal(*clPtr);
    return retVal;
  }
};

/** @brief Specialization of info_convert for converting a const char * type to
 *a string_class type.
 *@ref cl::sycl::info_convert
 */
template <>
struct COMPUTECPP_EXPORT info_convert<const char*, string_class> {
  static string_class cl_to_sycl(const char* clPtr, size_t numElems,
                                 cl_uint clParam);
};

/**
@brief Specialization of info_convert for converting a char * type to a
string_class type.
@ref cl::sycl::info_convert
*/
template <>
struct COMPUTECPP_EXPORT info_convert<char*, string_class> {
  static string_class cl_to_sycl(char* clPtr, size_t numElems, cl_uint clParam);
};

/** @brief Specialization of info_convert to converting size_t to uint32_t.
 * @ref cl::sycl::info_convert
 */
template <>
struct COMPUTECPP_EXPORT info_convert<size_t*, uint32_t> {
  static uint32_t cl_to_sycl(size_t* clPtr, size_t /*numElems*/,
                             cl_uint /*clParam*/) {
    return static_cast<uint32_t>(*clPtr);
  }
};

template <>
struct COMPUTECPP_EXPORT info_convert<cl_bitfield*, bool> {
  static bool cl_to_sycl(cl_bitfield* clValue, size_t numElems,
                         cl_uint clParam);
};

/*******************************************************************************
    use_host_info_definitions
*******************************************************************************/

/**
@brief Function which returns whether or not the host info definitions should be
used. This returns true in two cases; either if the isHost boolean parameter
specifying whether or not the calling SYCL object is in host mode is true or if
unit testing is enabled.
@param isHost Boolean specifying whether or not the calling SYCL object is in
host mode.
*/
COMPUTECPP_EXPORT bool use_host_info_definitions(bool isHost);

namespace detail {

/**
@brief Class to take the AND of two cl_bitfields, or do nothing for other types.
@tparam T the type of the runtime variable. Nothing happens if T is not
cl_bitfield.
@tparam andBits bits known at compile time to AND a cl_bitfield with. If T is
not of type cl_bitfield, it is expected that this has value
std::numeric_limits<cl_bitfield>::max() - ie. it would do nothing.
*/
template <typename T, cl_bitfield andBits>
struct extract_bit {
  static_assert(andBits == std::numeric_limits<cl_bitfield>::max(),
                "Template parameter andBits was set for an incompatible type.");
  /**
  @brief Take the AND of andBits and argument if argument is of type
  cl_bitfield. Does nothing for other types.
  @param clPtr a pointer to the data to be operated on. Operation is in-place.
  */
  static void extract(T* clPtr) {
    // Do nothing with the data
    (void)clPtr;
  }
};

template <cl_bitfield andBits>
struct extract_bit<cl_bitfield, andBits> {
  static void extract(cl_bitfield* clPtr) { *clPtr = *clPtr & andBits; }
};

/*******************************************************************************
    get_opencl_info_as_sycl
*******************************************************************************/

/** @brief Retrieves the OpenCL info value for the parameter specified by
 * clParam as the OpenCL type specified by clType and returns it as the
 * corresponding SYCL type specified by syclType.
 * @tparam syclInfo Specifies the SYCL enum class for which the info is being
 *                  requested.
 * @tparam syclType Specifies the SYCL type that the info is being returned as.
 * @tparam clType Specifies the intermediate OpenCL type that is returned from
 *                the OpenCL API, before being converted to the SYCL type.
 * @tparam andBits Specifies the bits to anded with value (of type clType)
 * returned from opencl with query clParam if clType==cl_bitfield
 * @param clObject The underlying OpenCL object on which the OpenCL info API is
 *                 being called.
 * @param clParam Specifies the OpenCL info parameter that is being requested.
 * @return The SYCL type corresponding to the requested OpenCL information
 */
template <typename syclInfo, typename syclType, typename clType,
          cl_bitfield andBits>
syclType get_opencl_info_as_sycl(
    const typename opencl_info_base<syclInfo>::cl_object& clObject,
    cl_uint clParam) {
  /* Create an instance of the SYCL type to be returned. */
  auto syclValue = syclType();

  /* Create a variable for the size of the value that will be returned. */
  size_t size = 0;

  /* Retrieve the size of the OpenCL info value that will be returned. */
  get_opencl_info<syclInfo>(clObject, clParam, nullptr, 0, &size);

  /* If zero size, there is nothing else to be done. */
  if (size == 0) {
    return syclValue;
  }

  /* Allocate memory for the char* based on the size of the value that will be
   * returned. */
  vector_class<char> charVector(size);

  /* Retrieve the value for the OpenCL info parameter specified by clParam. */
  get_opencl_info<syclInfo>(clObject, clParam, charVector.data(), size,
                            nullptr);

  /* Calculate the number of elements that were returned. */
  size_t numElems = size / sizeof(clType);

  /* Cast the char* to a pointer to the corresponding OpenCL type. */
  clType* clPtr = reinterpret_cast<clType*>(charVector.data());

  /* Do we need to extract any bits from the clValue before casting to the SYCL
  type? */
  extract_bit<clType, andBits>::extract(clPtr);

  /* Convert the pointer to the OpenCL type to the corresponding SYCL type. */
  syclValue =
      info_convert<clType*, syclType>::cl_to_sycl(clPtr, numElems, clParam);

  /* Return the SYCL value. */
  return syclValue;
}

/** @brief Retrieves the OpenCL info value for the parameter specified by
 * clParam as the OpenCL type specified by clType and returns it as the
 * corresponding SYCL type specified by syclType.
 * @tparam syclInfo Specifies the SYCL enum class for which the info is being
 *                  requested.
 * @tparam syclType Specifies the SYCL type that the info is being returned as.
 * @tparam clType Specifies the intermediate OpenCL type that is returned from
 *                the OpenCL API, before being converted to the SYCL type.
 * @param clObject The underlying OpenCL object on which the OpenCL info API is
 *                 being called.
 * @param clParam Specifies the OpenCL info parameter that is being requested.
 * @return The SYCL type corresponding to the requested OpenCL information
 */
template <typename syclInfo, typename syclType, typename clType>
syclType get_opencl_info_as_sycl(
    const typename opencl_info_base<syclInfo>::cl_object& clObject,
    cl_uint clParam) {
  return get_opencl_info_as_sycl<syclInfo, syclType, clType,
                                 std::numeric_limits<cl_bitfield>::max()>(
      clObject, clParam);
}

}  // namespace detail

/*******************************************************************************
    get_sycl_info
*******************************************************************************/

/**
@brief Retrieves the OpenCL info value for the parameter specified by clParam as
the OpenCL type specified by clType and returns it as the corresponding SYCL
type specified by syclType. If the requesting SYCL object is in host mode then
instead the SYCL value returned will be retrieved from the SYCL host info
definitions.
@tparam syclInfo Specifies the SYCL enum class for which the info is being
requested.
@tparam syclType Specifies the SYCL type that the info is being returned as.
@tparam clType Specifies the intermediate OpenCL type that is returned from the
OpenCL API, before being converted to the SYCL type.
@tparam clParam Specifies the OpenCL info parameter that is being requested.
@tparam andBits Specifies the bits to anded with value (of type clType)
returned from opencl with query clParam if clType==cl_bitfield
@param clObject The underlying OpenCL object on which the OpenCL info API is
being called.
@param isHost Whether or not the SYCL object that is calling this in host mode.
*/
template <typename syclInfo, typename syclType, typename clType,
          cl_uint clParam, cl_bitfield andBits>
syclType get_sycl_info(
    const typename opencl_info_base<syclInfo>::cl_object& clObject,
    bool isHost) {
  /* If the requesting object is in host mode retrieve the info through the host
   * info definitions. */
  if (use_host_info_definitions(isHost)) {
    /* Retrieve the SYCL info host info specified by clParam. */
    return sycl_host_info<syclType, clParam>::get();
  }
  /* If the requesting object is not in host mode retrieve the info through the
     OpenCL info API and convert it to the corresponding SYCL type. */
  else {
    return detail::get_opencl_info_as_sycl<syclInfo, syclType, clType, andBits>(
        clObject, clParam);
  }
}

template <typename syclInfo, typename syclType, typename clType,
          cl_uint clParam>
syclType get_sycl_info(
    const typename opencl_info_base<syclInfo>::cl_object& clObject,
    bool isHost) {
  return get_sycl_info<syclInfo, syclType, clType, clParam,
                       std::numeric_limits<cl_bitfield>::max()>(clObject,
                                                                isHost);
}

namespace detail {
/** @brief Detects and removes a terminating byte from a string_class object.
 * @param clPtr The c-string returned from OpenCL.
 * @param numElems The length of the c-string returned from OpenCL.
 */
inline string_class make_valid_string(const char* clPtr,
                                      const size_t numElems) {
  const auto validString = clPtr && numElems != 0;

  // return empty string object if the argument is empty
  if (!validString) {
    return {};
  }

  // We can't be sure the clPtr argument is null terminated, so the object
  // is created, and if a terminating byte is detected, it is removed.

  string_class result(clPtr, numElems);
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}
}  // namespace detail

inline std::vector<std::string>
info_convert<char*, std::vector<std::string>>::cl_to_sycl(const char* clPtr,
                                                          size_t /*numElems*/,
                                                          cl_uint /*clParam*/) {
  auto inputString = std::string{clPtr};

  // Replace semi-colons with spaces as some implementations use semi-colons
  // and some use spaces.
  std::replace(std::begin(inputString), std::end(inputString), ';', ' ');

  // Copy each space delimited substring from the input string into the
  // result vector.
  std::istringstream inputStream{inputString};

  std::vector<std::string> result(
      std::istream_iterator<std::string>{inputStream},
      std::istream_iterator<std::string>{});

  if (!inputStream.eof()) {
    COMPUTECPP_CL_ERROR_CODE_MSG(
        CL_SUCCESS, detail::cpp_error_code::GET_INFO_ERROR, nullptr,
        "Error separating extensions into individual strings.");
  }

  return result;
}

inline string_class info_convert<const char*, string_class>::cl_to_sycl(
    const char* clPtr, const size_t numElems, cl_uint /*clParam*/) {
  return detail::make_valid_string(clPtr, numElems);
}

inline string_class info_convert<char*, string_class>::cl_to_sycl(
    char* clPtr, const size_t numElems, cl_uint clParam) {
  return info_convert<const char*, string_class>::cl_to_sycl(clPtr, numElems,
                                                             clParam);
}

/** COMPUTECPP_DEV @endcond */

}  // namespace sycl
}  // namespace cl

#endif  // RUNTIME_INCLUDE_SYCL_INFO_H_

////////////////////////////////////////////////////////////////////////////////
