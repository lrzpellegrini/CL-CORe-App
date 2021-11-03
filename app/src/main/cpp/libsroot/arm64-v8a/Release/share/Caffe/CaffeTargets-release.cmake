#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)
IF(NOT DEFINED _CAFFE_INSTALL_PREFIX)
    get_filename_component (_CAFFE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
ENDIF()

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "caffeproto;${_CAFFE_INSTALL_PREFIX}/lib/libboost_system.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_thread.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_filesystem.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_date_time.so;${_CAFFE_INSTALL_PREFIX}/lib/libboost_atomic.so;${_CAFFE_INSTALL_PREFIX}/lib/libglog.so;gflags_shared;${_CAFFE_INSTALL_PREFIX}/lib/libprotobuf.so;${_CAFFE_INSTALL_PREFIX}/lib/libleveldb.so;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;${_CAFFE_INSTALL_PREFIX}/lib/libopenblas.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.so"
  IMPORTED_SONAME_RELEASE "libcaffe.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.so" )

# Import target "caffeproto" for configuration "Release"
set_property(TARGET caffeproto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffeproto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "${_CAFFE_INSTALL_PREFIX}/lib/libprotobuf.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffeproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffeproto )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffeproto "${_IMPORT_PREFIX}/lib/libcaffeproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
