cmake_minimum_required(VERSION 3.18)
project(medusa_camera LANGUAGES C)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "QNX")
    message(FATAL_ERROR "Build on the QNX Pi or configure with a QNX toolchain.")
endif()

find_package(Threads REQUIRED)
find_path(CAMERA_INCLUDE_DIR camera/camera_api.h REQUIRED)
find_library(CAMERA_LIBRARY camapi REQUIRED)

add_library(medusa_camera SHARED camera_capture.c)
target_compile_features(medusa_camera PRIVATE c_std_99)
target_compile_definitions(medusa_camera PRIVATE _QNX_SOURCE)
target_include_directories(medusa_camera PRIVATE ${CAMERA_INCLUDE_DIR})
target_link_libraries(medusa_camera PRIVATE ${CAMERA_LIBRARY} Threads::Threads)
