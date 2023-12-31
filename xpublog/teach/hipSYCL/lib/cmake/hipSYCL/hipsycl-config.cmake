
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was hipsycl-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(HIPSYCL_SYCLCC "${PACKAGE_PREFIX_DIR}/bin/syclcc-clang")
set(HIPSYCL_SYCLCC_LAUNCHER "${PACKAGE_PREFIX_DIR}/lib/cmake/hipSYCL/syclcc-launcher")
set(HIPSYCL_SYCLCC_LAUNCH_RULE_IN_FILE "${PACKAGE_PREFIX_DIR}/lib/cmake/hipSYCL/syclcc-launch.rule.in")
set(HIPSYCL_OMP_BACKEND_AVAILABLE "true")
set(HIPSYCL_CUDA_BACKEND_AVAILABLE "FALSE")
set(HIPSYCL_HIP_BACKEND_AVAILABLE "false")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(hipsycl-targets)

set(HIPSYCL_PLATFORM_OPTIONS "")

macro(check_backend backend)
  string(TOUPPER ${backend} backend_uc)
  if(HIPSYCL_${backend_uc}_BACKEND_AVAILABLE)
    list(APPEND HIPSYCL_PLATFORM_OPTIONS ${backend})
    if(NOT HIPSYCL_DEFAULT_PLATFORM)
      set(HIPSYCL_DEFAULT_PLATFORM ${backend})
    endif()
  endif()
endmacro()

check_backend("omp")
check_backend("cuda")
check_backend("hip")

list(JOIN HIPSYCL_PLATFORM_OPTIONS " | " HIPSYCL_PLATFORMS_STRING)

set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS}" CACHE STRING "Arguments to pass through directly to syclcc.")

set(HIPSYCL_TARGETS_DESC "in the format HIPSYCL_TARGETS=<platform1>[:arch1[,arch2][..,archN]][;<platform2>[:arch1][...]][..;<platformN>] where platforms are one of ${HIPSYCL_PLATFORMS_STRING}")
set(HIPSYCL_TARGETS "${HIPSYCL_TARGETS}" CACHE STRING "The platforms and architectures hipSYCL should target, ${HIPSYCL_TARGETS_DESC}.")
# IF HIPSYCL_TARGETS has not been explicitly set by the user, first tro to find the corresponding environment variable.
# If found, takes precedence over deprecated HIPSYCL_PLATFORM and HIPSYCL_GPU_ARCH
# IF not found, fallback to deprecated HIPSYCL_PLATFORM and HIPSYCL_GPU_ARCH logic
if(NOT HIPSYCL_TARGETS)
  set(HIPSYCL_TARGETS_ENV $ENV{HIPSYCL_TARGETS})
  if(HIPSYCL_TARGETS_ENV)
    message("Found HIPSYCL_TARGETS from environment: ${HIPSYCL_TARGETS_ENV}")
    set(HIPSYCL_TARGETS "${HIPSYCL_TARGETS_ENV}")
  endif()
endif()
if(HIPSYCL_TARGETS)
  set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-targets=\"${HIPSYCL_TARGETS}\"")
endif()

set(HIPSYCL_PLATFORM "${HIPSYCL_PLATFORM}" CACHE STRING "(DEPRECATED. Use HIPSYCL_TARGETS) The platform that hipSYCL should target. One of ${HIPSYCL_PLATFORMS_STRING}.")

set_property(CACHE HIPSYCL_PLATFORM PROPERTY STRINGS ${HIPSYCL_PLATFORM_OPTIONS})

# If HIPSYCL_PLATFORM has not been explicitly set by the user, first try to find
# the corresponding environment variable. If that isn't set either, and only
# a single platform is available, default to it. Otherwise throw an error.
if(HIPSYCL_TARGETS AND HIPSYCL_PLATFORM)
  message("Both HIPSYCL_TARGETS=${HIPSYCL_TARGETS} and (deprecated) HIPSYCL_PLATFORM=${HIPSYCL_PLATFORM} set, using HIPSYCL_TARGETS.")
elseif(NOT HIPSYCL_TARGETS)
  if(NOT HIPSYCL_PLATFORM)
    set(HIPSYCL_PLATFORM_ENV $ENV{HIPSYCL_PLATFORM})
    list(LENGTH HIPSYCL_PLATFORM_OPTIONS num_backends_available)
    if(HIPSYCL_PLATFORM_ENV)
      message("Found HIPSYCL_PLATFORM from environment: ${HIPSYCL_PLATFORM_ENV}")
      set(HIPSYCL_DEFAULT_PLATFORM ${HIPSYCL_PLATFORM_ENV})
    elseif(num_backends_available GREATER 1)
      message(SEND_ERROR "More than one hipSYCL target platform is available.\n"
        "Must specify desired target(s) [and associated device(s)] via HIPSYCL_TARGETS, ${HIPSYCL_TARGETS_DESC}.")
    endif()
    set(HIPSYCL_PLATFORM ${HIPSYCL_DEFAULT_PLATFORM})
    unset(HIPSYCL_PLATFORM_ENV)
  endif()

  # Determine canonical platform from aliases
  if(HIPSYCL_PLATFORM MATCHES "cpu|host|hipcpu|omp")
    set(HIPSYCL_PLATFORM_CANONICAL "omp")
  elseif(HIPSYCL_PLATFORM MATCHES "cuda|nvidia")
    set(HIPSYCL_PLATFORM_CANONICAL "cuda")
  elseif(HIPSYCL_PLATFORM MATCHES "rocm|amd|hip|hcc")
    set(HIPSYCL_PLATFORM_CANONICAL "hip")
  else()
    message(SEND_ERROR "Unknown hipSYCL platform '${HIPSYCL_PLATFORM}'")
  endif()

  unset(HIPSYCL_PLATFORMS_STRING)
  unset(HIPSYCL_DEFAULT_PLATFORM)

  set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-platform=${HIPSYCL_PLATFORM}")
endif()

set(HIPSYCL_CLANG "" CACHE STRING "Clang compiler executable used for compilation.")
if(HIPSYCL_CLANG)
  set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-clang=${HIPSYCL_CLANG}")
endif()

set(HIPSYCL_CUDA_PATH "" CACHE STRING "The path to the CUDA toolkit installation directory.")
if(HIPSYCL_CUDA_PATH)
  if((HIPSYCL_PLATFORM_CANONICAL STREQUAL "cuda") OR (HIPSYCL_TARGETS MATCHES "cuda"))
    set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-cuda-path=${HIPSYCL_CUDA_PATH}")
  else()
    message(WARNING "HIPSYCL_CUDA_PATH (${HIPSYCL_CUDA_PATH}) is ignored for current platform (${HIPSYCL_PLATFORM})")
  endif()
endif()

set(HIPSYCL_ROCM_PATH "" CACHE STRING "The path to the ROCm installation directory.")
if(HIPSYCL_ROCM_PATH)
  if((HIPSYCL_PLATFORM_CANONICAL STREQUAL "hip") OR (HIPSYCL_TARGETS MATCHES "hip"))
    set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-rocm-path=${HIPSYCL_ROCM_PATH}")
  else()
    message(WARNING "HIPSYCL_ROCM_PATH (${HIPSYCL_ROCM_PATH}) is ignored for current platform (${HIPSYCL_PLATFORM})")
  endif()
endif()

set(HIPSYCL_GPU_ARCH "" CACHE STRING "(DEPRECATED. Use HIPSYCL_TARGETS) GPU architecture used by ROCm / CUDA.")
if(HIPSYCL_GPU_ARCH AND NOT HIPSYCL_TARGETS)
  if(HIPSYCL_PLATFORM_CANONICAL STREQUAL "cuda" OR HIPSYCL_PLATFORM_CANONICAL STREQUAL "hip")
    set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-gpu-arch=${HIPSYCL_GPU_ARCH}")
  else()
    message(WARNING "HIPSYCL_GPU_ARCH (${HIPSYCL_GPU_ARCH}) is ignored for current platform (${HIPSYCL_PLATFORM})")
  endif()
else()
  set(_TMP $ENV{HIPSYCL_GPU_ARCH})
  if((HIPSYCL_PLATFORM_CANONICAL STREQUAL "cuda" OR HIPSYCL_PLATFORM_CANONICAL STREQUAL "hip") AND NOT _TMP)
    message(SEND_ERROR "Please specify HIPSYCL_GPU_ARCH")
  endif()
  unset(_TMP)
endif()

set(HIPSYCL_CPU_CXX "" CACHE STRING "The compiler that should be used when targeting only CPUs.")
if(HIPSYCL_CPU_CXX)
  if((HIPSYCL_PLATFORM_CANONICAL STREQUAL "cpu") OR (HIPSYCL_TARGETS MATCHES "omp"))
    set(HIPSYCL_SYCLCC_EXTRA_ARGS "${HIPSYCL_SYCLCC_EXTRA_ARGS} --hipsycl-cpu-cxx=${HIPSYCL_CPU_CXX}")
  else()
    message(WARNING "HIPSYCL_CPU_CXX (${HIPSYCL_CPU_CXX}) is ignored for current platform (${HIPSYCL_PLATFORM})")
  endif()
endif()

# To invoke syclcc, the add_sycl_to_target function sets a compiler and linker launch rule on the target that will pass
# the entire GCC or Clang command line to lib/cmake/hipSYCL/syclcc-launcher. The launcher will prepend syclcc-specific
# arguments from HIPSYCL_SYCLCC_EXTRA_ARGS and replace GCC or Clang with syclcc in the command line.
# This is done to keep COMPILE_FLAGS free from Clang-incompatible command line arguments, allowing it to be reused
# by clang(d)-based tooling and IDEs.
if(WIN32)
  set(HIPSYCL_SYCLCC_LAUNCH_RULE "python ${HIPSYCL_SYCLCC_LAUNCHER} --launcher-cxx-compiler=${CMAKE_CXX_COMPILER} --launcher-syclcc=\"python*${HIPSYCL_SYCLCC}\" ${HIPSYCL_SYCLCC_EXTRA_ARGS}")
else()
  set(HIPSYCL_SYCLCC_LAUNCH_RULE "${HIPSYCL_SYCLCC_LAUNCHER} --launcher-cxx-compiler=${CMAKE_CXX_COMPILER} --launcher-syclcc=${HIPSYCL_SYCLCC} ${HIPSYCL_SYCLCC_EXTRA_ARGS}")
endif()

# All SYCL targets must be rebuilt when syclcc arguments change, e.g. by changing the target platform. Since the
# contents of HIPSYCL_SYCLCC_LAUNCH_RULE are invisible to CMake's dependency tracking, we configure() a file with
# the variables's content and have every object file within a SYCL target depend on it.
set(HIPSYCL_SYCLCC_LAUNCH_RULE_FILE "${CMAKE_BINARY_DIR}/CMakeFiles/hipsycl-syclcc-launch.rule")
configure_file("${HIPSYCL_SYCLCC_LAUNCH_RULE_IN_FILE}" "${HIPSYCL_SYCLCC_LAUNCH_RULE_FILE}" @ONLY)
set(HIPSYCL_SYCLCC_EXTRA_OBJECT_DEPENDS "${HIPSYCL_SYCLCC_LAUNCHER};${HIPSYCL_SYCLCC_LAUNCH_RULE_FILE}")

# Do not call target_sources after add_sycl_to_target or dependency tracking on compiler flags will break in subtle ways
function(add_sycl_to_target)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(ADD_SYCL
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  # The SOURCES argument to add_sycl_to_target is ignored and exists only for compatibility with ComputeCpp, since
  # the compiler launcher can only be set with per-target granularity. Dependencies on the launcher args are therefore
  # also set for all files in the list.
  get_target_property(ADD_SYCL_SOURCES "${ADD_SYCL_TARGET}" SOURCES)

  foreach(ADD_SYCL_SOURCE_ITER IN LISTS ADD_SYCL_SOURCES)
    get_source_file_property(ADD_SYCL_OBJECT_DEPENDS "${ADD_SYCL_SOURCE_ITER}" OBJECT_DEPENDS)
    if(ADD_SYCL_OBJECT_DEPENDS)
      set(ADD_SYCL_OBJECT_DEPENDS "${ADD_SYCL_OBJECT_DEPENDS};${HIPSYCL_SYCLCC_EXTRA_OBJECT_DEPENDS}")
    else()
      set(ADD_SYCL_OBJECT_DEPENDS "${HIPSYCL_SYCLCC_EXTRA_OBJECT_DEPENDS}")
    endif()
    set_source_files_properties("${ADD_SYCL_SOURCE_ITER}" PROPERTIES OBJECT_DEPENDS "${ADD_SYCL_OBJECT_DEPENDS}")
  endforeach()

  set_target_properties("${ADD_SYCL_TARGET}" PROPERTIES RULE_LAUNCH_COMPILE "${HIPSYCL_SYCLCC_LAUNCH_RULE}")
  set_target_properties("${ADD_SYCL_TARGET}" PROPERTIES RULE_LAUNCH_LINK "${HIPSYCL_SYCLCC_LAUNCH_RULE}")

  target_link_libraries(${ADD_SYCL_TARGET} PUBLIC hipSYCL::hipSYCL-rt)
endfunction()
