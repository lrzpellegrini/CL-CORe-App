#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)
IF(NOT DEFINED _CAFFE_INSTALL_PREFIX)
    get_filename_component (_CAFFE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
ENDIF()

# Import target "caffe" for configuration "Debug"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "caffeproto;${_CAFFE_INSTALL_PREFIX}/lib/libboost_system.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_thread.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_filesystem.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_date_time.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_atomic.so;${_CAFFE_INSTALL_PREFIX}/lib/libglogd.so;gflags_shared;${_CAFFE_INSTALL_PREFIX}/lib/libprotobufd.so;${_CAFFE_INSTALL_PREFIX}/lib/libleveldb.so;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;${_CAFFE_INSTALL_PREFIX}/lib/libopenblas_d.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcaffe-d.so"
  IMPORTED_SONAME_DEBUG "libcaffe-d.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe-d.so" )

# Import target "caffeproto" for configuration "Debug"
set_property(TARGET caffeproto APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(caffeproto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "${_CAFFE_INSTALL_PREFIX}/lib/libprotobufd.so"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcaffeproto-d.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffeproto )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffeproto "${_IMPORT_PREFIX}/lib/libcaffeproto-d.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
