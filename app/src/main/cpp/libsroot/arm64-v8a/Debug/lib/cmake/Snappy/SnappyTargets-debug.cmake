#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Snappy::snappy" for configuration "Debug"
set_property(TARGET Snappy::snappy APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Snappy::snappy PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libsnappy.so"
  IMPORTED_SONAME_DEBUG "libsnappy.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS Snappy::snappy )
list(APPEND _IMPORT_CHECK_FILES_FOR_Snappy::snappy "${_IMPORT_PREFIX}/lib/libsnappy.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)