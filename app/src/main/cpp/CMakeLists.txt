cmake_minimum_required(VERSION 3.4.1)
include(ExternalProject)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Cmake prefix path" FORCE)
set(CMAKE_FIND_ROOT_PATH "${CMAKE_PREFIX_PATH}" CACHE INTERNAL "Cmake find root path" FORCE)

set(BUILD_SHARED_LIBS ON)

set(LMDB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/lmdb/libraries/liblmdb)

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED)
set(LMDB_LIBRARY lmdb)

target_sources(native-lib
        PUBLIC
        # Provides a relative path to your source file(s).
        native-lib.cpp
        ${CMAKE_SOURCE_DIR}/lmdb/libraries/liblmdb/lmdb.h
        ${CMAKE_SOURCE_DIR}/lmdb/libraries/liblmdb/mdb.c
        ${CMAKE_SOURCE_DIR}/lmdb/libraries/liblmdb/midl.h
        ${CMAKE_SOURCE_DIR}/lmdb/libraries/liblmdb/midl.c
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/cwr_app.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/cwr_app.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/cwr.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/cwr.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/visualization.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/visualization.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/train_utils.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/train_utils.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/model_status_reporter.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/model_status_reporter.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/rehearsal_memory.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/rehearsal_memory.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/training_status_reporter.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/myapp_training_reporter.h
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/src/myapp_training_reporter.cpp
        ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include/picosha2.h)

#set(gflags_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/lib/cmake/gflags)
#set(glog_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/lib/cmake/glog)
#set(leveldb_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/lib/cmake/leveldb)
#set(Snappy_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/lib/cmake/Snappy)
# OpenCV install directory structure is strange and requires an explicit *_DIR set to the OpenCVConfig.cmake location
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/sdk/native/jni)
#set(OpenBLAS_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/share/cmake/OpenBLAS)
#set(Caffe_DIR ${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/share/Caffe)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(leveldb REQUIRED)
find_package(Snappy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(Caffe REQUIRED)
find_package(Threads)
find_package(Boost REQUIRED)

file(GLOB allLibs "${CMAKE_SOURCE_DIR}/libsroot/${ANDROID_ABI}/${CMAKE_BUILD_TYPE}/**/*.so")

# Copies shared libraries to output directory (so that they will be added to the final APK)
foreach(lib_path ${allLibs})
    add_custom_command(TARGET native-lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${lib_path} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
endforeach()

find_library(log-lib log)

target_include_directories(native-lib PUBLIC ${CMAKE_SOURCE_DIR}/native-lib/CwrApp/include ${LMDB_INCLUDE_DIR})
target_link_libraries(
       native-lib

        ${log-lib}
        Snappy::snappy
        leveldb::leveldb
        ${Boost_LIBRARIES}
        glog::glog
        gflags
        OpenBLAS::OpenBLAS
        ${OpenCV_LIBS}
        caffe)