#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hipSYCL::hipSYCL-rt" for configuration "Release"
set_property(TARGET hipSYCL::hipSYCL-rt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hipSYCL::hipSYCL-rt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhipSYCL-rt.so"
  IMPORTED_SONAME_RELEASE "libhipSYCL-rt.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS hipSYCL::hipSYCL-rt )
list(APPEND _IMPORT_CHECK_FILES_FOR_hipSYCL::hipSYCL-rt "${_IMPORT_PREFIX}/lib/libhipSYCL-rt.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
