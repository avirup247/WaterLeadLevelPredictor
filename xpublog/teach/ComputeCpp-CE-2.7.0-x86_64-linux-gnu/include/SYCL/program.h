/*****************************************************************************

    Copyright (C) 2002-2018 Codeplay Software Limited
    All Rights Reserved.

    Codeplay's ComputeCpp

*******************************************************************************/

/** @file program.h
 *
 * @brief This file implements the \ref cl::sycl::program class as defined by
 * the SYCL 1.2 specification.
 */

#ifndef RUNTIME_INCLUDE_SYCL_PROGRAM_H_
#define RUNTIME_INCLUDE_SYCL_PROGRAM_H_

#include "SYCL/base.h"
#include "SYCL/cl_types.h"
#include "SYCL/common.h"
#include "SYCL/context.h"
#include "SYCL/device.h"
#include "SYCL/error_log.h"
#include "SYCL/include_opencl.h"
#include "SYCL/info.h"
#include "SYCL/kernel.h"
#include "SYCL/predefines.h"
#include "SYCL/property.h"  // IWYU pragma: keep

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "computecpp_export.h"

namespace cl {
namespace sycl {

namespace info {

/** @brief Program descriptor to query information about a program object
 */
enum class program : int {
  reference_count, /**< Query the reference count of the program */
  context,         /**< Query the cl_context associate with the program */
  devices          /**< Query the set of devices the program is built against */
};

}  // namespace info

/// @cond COMPUTECPP_DEV

/// Program info definition
COMPUTECPP_DEFINE_SYCL_INFO_HANDLER(program, cl_program_info, cl_program)

/// Program info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(program, reference_count,
                                      CL_PROGRAM_REFERENCE_COUNT, cl_uint,
                                      cl_uint)
/// Program info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(program, context, CL_PROGRAM_CONTEXT,
                                      cl::sycl::context, cl_context)
/// Program info definition
COMPUTECPP_DEFINE_SYCL_INFO_PARAMETER(program, devices, CL_PROGRAM_DEVICES,
                                      vector_class<cl::sycl::device>,
                                      cl_device_id)

/// Host program info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(program, reference_count, 0)
/// Host program info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(program, context, cl::sycl::context())
/// Host program info definition
COMPUTECPP_DEFINE_SYCL_INFO_HOST(program, devices, vector_class<device>())

/// COMPUTECPP_DEV @endcond

/** @brief Enum describing the build state of the program
 */
enum class program_state { none, compiled, linked };

/**
@brief Public facing program class that provides an interface for abstracting
the construction and building of a cl_program object, See section 3.5.5 of the
SYCL 1.2 specification.
*/
class COMPUTECPP_EXPORT program {
  /**
  @brief Friend class declaration of kernel as the program class requires access
  to the kernel classes protected constructor.
  */
  friend class kernel;

 public:
  /** Constructs a program for a context
   * @param context The context that will be associated with the program
   * @param propList Additional properties
   */
  explicit program(const context& context, const property_list& propList = {});

  /** Constructs a program for a list of devices on a context
   * @param context The context that will be associated with the program
   * @param deviceList List of devices that will be associated with the program
   * @param propList Additional properties
   */
  program(const context& context, vector_class<device> deviceList,
          const property_list& propList = {});

  /**
  @brief Inter-op constructor that takes a context and a cl_program. Note that
  the clProgram param must have previously been created from the underlying
  cl_context of the context parameter and the underlying cl_devices from the
  list of devices parameter.
  @param context A reference to the context that the cl_program will be
  associated with.
  @param clProgram The cl_program that the program will be assigned to.
  */
  program(const context& context, cl_program clProgram);

  /** Linker constructor from a list of programs
   *
   * All of the provided programs must be in the compiled program state
   * and associated with the same context.
   * The constructed program will be in the linked program state.
   *
   * @param programList List of programs to link together into a new program
   * @param propList Additional properties
   * @throw invalid_object_error if provided programs not associated
   *        with the same context
   *        or if not all of them are in the compiled program state
   */
  program(vector_class<program> programList, const property_list& propList = {})
      : program{std::move(programList), "", propList} {}

  /** Linker constructor from a list of programs
   *
   * All of the provided programs must be in the compiled program state
   * and associated with the same context.
   * The constructed program will be in the linked program state.
   *
   * @param programList List of programs to link together into a new program
   * @param linkOptions String of options used when linking programs
   * @param propList Additional properties
   * @throw invalid_object_error if provided programs not associated
   *        with the same context
   *        or if not all of them are in the compiled program state
   */
  program(vector_class<program> programList, string_class linkOptions,
          const property_list& propList = {})
      : program{std::move(programList), linkOptions.c_str(), propList} {}

  /// @copydoc program(vector_class<program>, string_class,
  ///                  const property_list&)
  program(vector_class<program> programList, const char* linkOptions,
          const property_list& propList = {});

  /**
  @brief Copy constructor that initialises a copy of the program with the same
  underlying cl_program, associated context and list of associated devices.
  @param rhs The program being copied from.
  */
  program(const program& rhs) = default;

  /**
  @brief Assignment operator that initialises a copy of the program with the
  same underlying cl_program, associated context and list of associated devices.
  @param rhs The program being assigned from.
  */
  program& operator=(const program& rhs) = default;

  /**
  @brief Copy constructor that initialises a copy of the program with the same
  underlying cl_program, associated context and list of associated devices.
  @param rhs The program being copied from.
  */
  program(program&& rhs) = default;

  /**
  @brief Assignment operator that initialises a copy of the program with the
  same underlying cl_program, associated context and list of associated devices.
  @param rhs The program being assigned from.
  */
  program& operator=(program&& rhs) = default;

  /** @brief Determines if lhs and rhs are equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if same underlying object
   */
  friend inline bool operator==(const program& lhs, const program& rhs) {
    return (lhs.get_impl() == rhs.get_impl());
  }

  /** @brief Determines if lhs and rhs are not equal
   * @param lhs Left-hand-side object in comparison
   * @param rhs Right-hand-side object in comparison
   * @return True if different underlying objects
   */
  friend inline bool operator!=(const program& lhs, const program& rhs) {
    return !(lhs == rhs);
  }

  /**
  @brief Destructor that releases the cl_program.
  */
  COMPUTECPP_TEST_VIRTUAL ~program() = default;

  /**
   * @brief Compile a SYCL kernel using name and optional custom compile
   * options. This function creates a ready-to-link program.
   *
   * Note that calling this member function is invalid if the program has
   * already been successfully compiled, built or linked via either
   * link(string_class), compile_with_kernel_type(string_class),
   * build_with_kernel_type(string_class) or program(vector_class<program>,
   * string_class).
   * @tparam kernelT Typename specifying the name of the kernel to be compiled.
   * @param compileOptions String of compile options that will be passed to the
   * OpenCL driver.
   */
  template <typename kernelT>
  void compile_with_kernel_type(string_class compileOptions = "") {
    if (detail::kernel_info<kernelT>::name == nullptr) {
      COMPUTECPP_CL_ERROR_CODE(
          0, detail::cpp_error_code::KERNEL_NOT_FOUND_ERROR, nullptr);
    }
    const auto ctx = this->get_context();
    auto hostBinInfo = detail::make_host_binary_info();
    const auto& binInfo = program::select_kernel_binary_info(
        hostBinInfo, detail::kernel_info<kernelT>::bin_info,
        detail::kernel_info<kernelT>::bin_count, ctx);
    this->compile_with_kernel_type_impl(binInfo.data, binInfo.data_size,
                                        binInfo.used_extensions,
                                        compileOptions.c_str(), binInfo.target);
  }

  /**
   * Compiles a program from the given OpenCL C kernel source.
   * Note that calling this member function is invalid if the program has
   * already been successfully compiled, built or linked via either
   * link(string_class), compile_with_kernel_type(string_class),
   * build_with_kernel_type(string_class) or program(vector_class<program>,
   * string_class).
   * @param kernelSource to compile
   * @param compilation options for the source
   */
  void compile_with_source(string_class kernelSource,
                           string_class compileOptions = "") {
    compile_with_source(kernelSource.c_str(), compileOptions.c_str());
  }

  /// @copydoc compile_with_source(string_class, string_class)
  void compile_with_source(const char* kernelSrc, const char* compileOptions);

  /**
   * Creates a valid cl_program from a pre-built kernel provided by the
   * underlying OpenCL implementation.
   * @param kernel The name of the built-in kernel
   */
  void create_from_built_in_kernel(string_class kernel) {
    return create_from_built_in_kernel(kernel.c_str());
  }

  /// @copydoc create_from_built_in_kernel(string_class)
  void create_from_built_in_kernel(const char* kernel);

  /**
  @brief Build a SYCL kernel using its name and optional custom build options.
  This function produces a ready-to-run program.
  Note that calling this member function is invalid if the program has
  already been successfully compiled, built or linked via either
  link(string_class), compile_with_kernel_type(string_class),
  build_with_kernel_type(string_class) or program(vector_class<program>,
  string_class).
  @tparam kernelT Typename specifying the name of the kernel to be built.
  @param buildOptions The string specifying the build options to provide to the
  underlying OpenCL API.
  */
  template <typename kernelT>
  void build_with_kernel_type(string_class buildOptions = "") {
    if (detail::kernel_info<kernelT>::name == nullptr) {
      COMPUTECPP_CL_ERROR_CODE(
          0, detail::cpp_error_code::KERNEL_NOT_FOUND_ERROR, nullptr);
    }
    const auto ctx = this->get_context();
    auto hostBinInfo = detail::make_host_binary_info();
    const auto& binInfo = program::select_kernel_binary_info(
        hostBinInfo, detail::kernel_info<kernelT>::bin_info,
        detail::kernel_info<kernelT>::bin_count, ctx);
    this->build_with_kernel_type_impl(binInfo.data, binInfo.data_size,
                                      binInfo.used_extensions,
                                      buildOptions.c_str(), binInfo.target);
  }

  /**
  @brief Creates and builds a program from OpenCL C kernel source and optional
         build options. This function produces a ready-to-run program.
  @param kernelSource Source of the OpenCL kernel.
  @param buildOptions The string specifying the build options to provide to the
         underlying OpenCL API.
  */
  void build_with_source(string_class kernelSource,
                         string_class buildOptions = "") {
    this->build_with_source(kernelSource.c_str(), buildOptions.c_str());
  }

  /// @copydoc build_with_source(string_class, string_class)
  void build_with_source(const char* kernelSrc, const char* compileOptions);

  /**
   * @brief Link all compiled programs using the (optional) link options.
   * This function produce a ready-to-run program using a compiled program.
   *
   * Note that calling this member function is invalid if the cl_program has
   * already been successfully built or linked via either link(string_class),
   * build_with_kernel_type(string_class) or program(vector_class<program>,
   * string_class).
   * @param linkOptions String specifying the link options to provide to the
   * underlying OpenCL API.
   */
  void link(string_class linkOptions = "") { this->link(linkOptions.c_str()); }

  /// @copydoc link(string_class)
  void link(const char* linkOptions);

  /**
  @brief Checks whether the program contains a kernel specified by the type.
  @tparam kernelT Typename specifying the name of the kernel to be returned.
  @return True if the SYCL kernel function defined by the type kernelT
          is an available kernel, either within the encapsulated cl_program
          (if this SYCL program is an OpenCL program), or on the host,
          otherwise false.
  */
  template <typename kernelT>
  bool has_kernel() const {
    const char* kernelName = detail::kernel_info<kernelT>::name;
    if (kernelName) {
      return has_kernel(kernelName);
    } else {
      /* Host mode or name not found */
      return has_kernel("");
    }
  }

  /**
   * @brief Checks whether the program contains a kernel specified by the name.
   * @return True if the OpenCL C kernel function defined by kernelName is an
   *         available kernel within the encapsulated cl_program and this SYCL
   *         program is not a host program, otherwise false.
   */
  bool has_kernel(string_class kernelName) const {
    return has_kernel(kernelName.c_str());
  }

  /// @copydoc has_kernel(string_class)
  bool has_kernel(const char* kernelName) const;

  /**
   * @brief Retrieve a SYCL \ref kernel object described by the typename
   *        kernelT.
   * @tparam kernelT Typename specifying the name of the kernel to be returned.
   * @return The kernel that has been created from the kernel name parameter.
   */
  template <typename kernelT>
  kernel get_kernel() const {
    const char* kernelName = detail::kernel_info<kernelT>::name;
    if (kernelName) {
      return get_kernel(kernelName);
    } else {
      /* Host mode or name not found */
      return get_kernel("");
    }
  }

  /**
   * @brief Retrieve a SYCL \ref kernel object described by the kernel name.
   * @param kernelName The string specifying the kernel name.
   * @return The kernel that has been created form the kernel name parameter.
   */
  kernel get_kernel(string_class kernelName) const {
    return get_kernel(kernelName.c_str());
  }

  /// @copydoc get_kernel(string_class)
  kernel get_kernel(const char* kernelName) const;

  /**
  @brief Retrieves information about the program. The runtime query the OpenCL
  API and then converts the result into the SYCL representation before returning
  it.
  @tparam param Information to retrieve.
  @return The information in the SYCL format.
  */
  template <info::program param>
  COMPUTECPP_EXPORT
      typename info::param_traits<info::program, param>::return_type
      get_info() const;

  /**
  @brief Return the list of binaries that were used to compile and link the
  program.
  @return The list of binaries that were used to compile and link the program.
  */
  vector_class<vector_class<char>> get_binaries() const;

  /**
  @brief Retrieves the context associated with the program.
  @return Associated context.
  */
  context get_context() const;

  /**
  @brief Return the list of devices associated with the program.
  @return The list of associated devices.
  */
  vector_class<cl::sycl::device> get_devices() const;

  /**
   * @brief Return the compile options used when compiling the program.
   * @return A string specifying the compile options used when compiling the
   *         program.
   */
  string_class get_compile_options() const {
    return string_class{get_compile_options_impl()};
  }

  /**
   * @brief Return the link options used when linking the program.
   * @return A string specifying the link options used when linking the program.
   */
  string_class get_link_options() const {
    return string_class{get_link_options_impl()};
  }

  /**
   * @brief Return the build options used when building the program.
   * @return A string specifying the options used when building the program.
   */
  string_class get_build_options() const {
    return string_class{get_build_options_impl()};
  }

  /**
  @brief Inter-op member function that returns the underlying cl_program.
  @return The underlying cl_program usable by the OpenCL API.
  */
  cl_program get() const;

  /**
  @brief Return a bool specifying whether the program has been linked.
  @return True if the program has been linked.
  */
  bool is_linked() const;

  /**
  @brief Returns whether the program was constructed from a host context
  @return True if program constructed from a host context
  */
  bool is_host() const;

  /**
  @brief Retrieves the current build state of the program
  @return The build state of the program
  */
  program_state get_state() const;

  /// @cond COMPUTECPP_DEV
  /**
  @brief Implementation-defined member function that returns the runtime's
  program implementation object.
  @return Opaque pointer to runtime's program implementation object.
  */
  dprogram_shptr get_impl() const;

  /**
  @brief Returns a program for a kernel from a context.
  @tparam kernelT The kernel to build the program for.
  @param c The context that the kernel is to be created on.
  @return The created program.
  */
  template <typename kernelT>
  static program create_program_for_kernel(cl::sycl::context c) {
    if (detail::kernel_info<kernelT>::name == nullptr && !c.is_host()) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          CL_SUCCESS, detail::cpp_error_code::KERNEL_NOT_FOUND_ERROR,
          c.get_impl().get(),
          "Unable to retrieve kernel function, is integration header included?")
    } else if (c.is_host()) {
      return program(c);
    }
    // Otherwise, create a normal Program
    std::string kernelName = detail::kernel_info<kernelT>::name;
    if (kernelName.empty()) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          0, detail::cpp_error_code::KERNEL_NOT_FOUND_ERROR, nullptr,
          kernelName.c_str());
    }
    auto hostBinInfo = detail::make_host_binary_info();
    const auto& binInfo = program::select_kernel_binary_info(
        hostBinInfo, detail::kernel_info<kernelT>::bin_info,
        detail::kernel_info<kernelT>::bin_count, c);
    return program(create_program_for_kernel_impl(
        kernelName.c_str(), binInfo.data, binInfo.data_size,
        binInfo.used_extensions, c.get_impl(), binInfo.target));
  }
  /// COMPUTECPP_DEV @endcond

 protected:
  /// @cond COMPUTECPP_DEV
  /**
  @brief Implementation-defined constructor that takes a detail program that
  initialises a copy of the program with the same underlying cl_program,
  associated context and list of associated devices.
  @param impl The detail program being initialized from.
  */
  explicit program(dprogram_shptr impl) : m_impl(std::move(impl)) {}
  /// COMPUTECPP_DEV @endcond

  /**
  @brief Inter-op member function that returns the underlying cl_program.
  @return The underlying cl_program usable by the OpenCL API.
  */
  cl_program get_no_retain() const;

 private:
  /**
  @brief Implementation-defined member function that performs the implementation
  of the build_with_kernel_type() template member function. Assigns the
  cl_program to the result of the building; with the build options parameter, of
  the binary data and size parameters.
  @ref program::build_with_kernel_type
  @brief binaryData The data of the binary to be built.
  @brief binarySize The size of the binary to be built.
  @param requiredExtensions A list of extensions as strings.
  @param buildOptions The string specifying the build options the cl_program
  will be built with.
  @param target target string
  */
  void build_with_kernel_type_impl(detail::binary_address binaryData,
                                   size_t binarySize,
                                   const char* const* const requiredExtensions,
                                   const char* buildOptions,
                                   const char* target);

  /**
   * @brief Implementation of the @ref program::compile_with_kernel_type()
   * template member function.
   *
   * Assigns the cl_program to the result of the compilation with the
   * compile options parameter.
   * @brief binaryData The data of the binary to be compiled.
   * @brief binarySize The size of the binary to be compiled.
   * @param requiredExtensions A list of extensions as strings.
   * @param buildOptions Compile options the cl_program will be built with.
   * @param target target string
   */
  void compile_with_kernel_type_impl(
      detail::binary_address binaryData, size_t binarySize,
      const char* const* const requiredExtensions, const char* compileOptions,
      const char* target);

  /**
   * @brief ABI-safe implementation of get_compile_options(). Callers must copy
   * returned string.
   * @return Pointer to compile options.
   */
  const char* get_compile_options_impl() const;

  /**
   * @brief ABI-safe implementation of get_link_options(). Callers must copy
   * returned string.
   * @return Pointer to link options.
   */
  const char* get_link_options_impl() const;

  /**
   * @brief ABI-safe implementation of get_build_options(). Callers must copy
   * returned string.
   * @return Pointer to build options.
   */
  const char* get_build_options_impl() const;

  /**
  @brief Implementation-defined static member function that performs the
  implementation of create_program_for_kernel implementation-defined template
  member function. Returns a program for a kernel invoke API, either creating
  and caching the program or retrieving a cached program from a  previous call.
  @ref program::create_program_for_kernel
  @param kernelName The string specifying the name of the kernel to be built.
  @param binaryData The data of the binary to be built.
  @param dataSize The size of the binary to be built.
  @param requiredExtensions A list of extensions as strings.
  @param context The detail context that the program is to be associated with.
  @param target target string
  */
  static program create_program_for_kernel_impl(
      const char* kernelName, detail::binary_address binaryData,
      size_t dataSize, const char* const* const requiredExtensions,
      dcontext_shptr context, const char* target);

  /*!
  @brief Friend class declaration of handler, as it needs to call this method
  to get the binary information to query the arguments.
  */
  friend class handler;

 protected:
  /**
  @brief Retrieve an appropriate binary to build a program.

  If a sycl stub file has more than one binary in it (e.g. spir64 and spirv64)
  the environment variable COMPUTECPP_TARGET_BITCODE can be used to select which
  binary this function will return. If the binary for the specified value is not
  found, the first one available will be returned.

  @param binList The list available binaries for this program.
  @param binListSize The length of `binList`.
  @param dev Device where the binary will execute on.
  @return A reference to the selected binary.
  */
  template <typename BIType>
  static const BIType& select_kernel_binary_info_helper(const BIType* binList,
                                                        size_t binListSize,
                                                        ddevice_wkptr dev) {
    if ((binList == nullptr) || (binListSize == 0)) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          0, detail::cpp_error_code::BINARY_NOT_FOUND_ERROR, nullptr,
          "Unable to retrieve a binary, is integration header included?");
    }

    // Find most suitable binary
    const auto endOfList = binList + binListSize;
    auto bestMatch = endOfList;
    int bestMatchValue = -1;
    for (auto it = binList; it < endOfList; ++it) {
      if (it->binary_info == nullptr) {
        // This can happen when -fsycl-split-modules is used
        break;
      }
      const int currentMatchValue =
          rank_binary_info_impl(*it->binary_info, dev);
      if (currentMatchValue > bestMatchValue) {
        bestMatchValue = currentMatchValue;
        bestMatch = it;
      }
    }

    if (bestMatchValue >= 0) {
      // Only select positive matches
      return *bestMatch;
    }

    // The default case is to return the first kernel binary available.
    return binList[0];
  }

  /** Retrieves binary info to build a program
   * @tparam BIType
   * @param hostBinaryInfo Host binary info.
   *        Will be selected if the context is a host context
   * @param binList The list available binaries for this program
   * @param binListSize The length of `binList`
   * @param ctx Context to retrieve binary info for
   * @return Reference to kernel binary info
   */
  template <typename BIType>
  static const detail::kernel_binary_info& select_kernel_binary_info(
      const detail::kernel_binary_info& hostBinaryInfo, const BIType* binList,
      size_t binListSize, const context& ctx) {
    if (ctx.is_host()) {
      return hostBinaryInfo;
    }
    // Retrieve suitable binary for first available device
    const ddevice_wkptr dev = ctx.get_devices()[0].get_impl();
    const auto& binInfo =
        *select_kernel_binary_info_helper(binList, binListSize, dev)
             .binary_info;
    if (binInfo.target == nullptr) {
      COMPUTECPP_CL_ERROR_CODE_MSG(
          0, detail::cpp_error_code::TARGET_NOT_FOUND_ERROR, nullptr,
          binInfo.target);
    }
    return binInfo;
  }

 private:
  /*!
  @brief Returns true iff the provided binary info should be used.

  This will return true if this is a PE ComputeCpp edition, the
  `COMPUTECPP_TARGET_BITCODE` is set, and the provided binary info has this
  bitcode type.

  Note that this function may return false for all binary infos, in that case
  no preference is indicated.

  @param binInfo The binary info to check.
  @return true iff the provided binary info should be used.
  @deprecated A valid non-host device must be provided as well
  */
  COMPUTECPP_DEPRECATED_API("Please provide a valid non-host device")
  static bool should_use_binary_info_impl(
      const detail::kernel_binary_info& binInfo);

  /** Ranks the kernel binary on how suitable it is to be used by the device.
   *
   * @param binInfo Kernel binary to rank
   * @param dev Valid non-host device
   * @return Non-negative result if the kernel binary is suitable
   *  for the device. Higher rank indicates a better match.
   *  A negative rank indicates that the device most likely cannot execute
   *  the binary so it shouldn't be selected.
   */
  static int rank_binary_info_impl(const detail::kernel_binary_info& binInfo,
                                   const ddevice_wkptr& dev);

  /** Shared pointer to the implementation-defined program object. */
  dprogram_shptr m_impl;
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
template <>
struct hash<cl::sycl::program> {
 public:
  /**
  @brief enables calling an std::hash object as a function with the object to be
  hashed as a parameter
  @param object the object to be hashed
  @tparam std the std namespace where this specialization resides
  */
  size_t operator()(const cl::sycl::program& object) const {
    hash<cl::sycl::dprogram_shptr> hasher;
    return hasher(object.get_impl());
  }
};
}  // namespace std
#endif  // RUNTIME_INCLUDE_SYCL_PROGRAM_H_
