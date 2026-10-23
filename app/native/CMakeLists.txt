cmake_minimum_required(VERSION 3.24)

# --- toolchain discovery (must happen before project()) ------------------------------------
# CUDA toolkit: -DCUDAToolkit_ROOT / env CUDA_PATH (layout: bin/nvcc.exe, include, lib/x64).
if(NOT CMAKE_CUDA_COMPILER)
  set(_fb_cuda_root "")
  foreach(_c "${CUDAToolkit_ROOT}" "$ENV{CUDAToolkit_ROOT}" "$ENV{CUDA_PATH}" "$ENV{CUDA_HOME}")
    if(_c AND EXISTS "${_c}/bin/nvcc.exe" AND NOT _fb_cuda_root)
      set(_fb_cuda_root "${_c}")
    elseif(_c AND EXISTS "${_c}/bin/nvcc" AND NOT _fb_cuda_root)
      set(_fb_cuda_root "${_c}")
    endif()
  endforeach()
  if(_fb_cuda_root)
    file(TO_CMAKE_PATH "${_fb_cuda_root}" _fb_cuda_root)
    set(CUDAToolkit_ROOT "${_fb_cuda_root}" CACHE PATH "CUDA toolkit root" FORCE)
    if(WIN32)
      set(CMAKE_CUDA_COMPILER "${_fb_cuda_root}/bin/nvcc.exe")
    else()
      set(CMAKE_CUDA_COMPILER "${_fb_cuda_root}/bin/nvcc")
    endif()
  endif()
endif()

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Native SASS for Turing/Ampere/Ada, SASS+PTX for Blackwell (PTX lets future GPUs JIT).
set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

project(fb_native VERSION 1.0.0 LANGUAGES CXX CUDA)

option(FB_BUILD_TESTS "Build the fb_native test executable" OFF)
option(FB_STATIC_CRT "Link the MSVC C/C++ runtime statically (no vcruntime140.dll dependency)" ON)
set(TENSORRT_INCLUDE_DIR "$ENV{TENSORRT_INCLUDE}" CACHE PATH "Directory containing NvInferRuntime.h")

if(MSVC AND FB_STATIC_CRT)
  # Always the release static CRT (also for Debug builds: keeps the DLL free of ucrtbased/vcruntime deps).
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
  # The Rust cmake crate injects -MD/-MDd through CMAKE_CXX_FLAGS; drop them (CMake adds -MT itself).
  string(REGEX REPLACE "[-/]M[DT]d?( |$)" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "[-/]M[DT]d?( |$)" " " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TENSORRT_INCLUDE_DIR OR NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h")
  message(FATAL_ERROR "TensorRT headers not found. Set env TENSORRT_INCLUDE or -DTENSORRT_INCLUDE_DIR=<dir with NvInferRuntime.h> (current: '${TENSORRT_INCLUDE_DIR}')")
endif()
file(TO_CMAKE_PATH "${TENSORRT_INCLUDE_DIR}" TENSORRT_INCLUDE_DIR)

find_package(CUDAToolkit REQUIRED)

# Predictable output layout:  <build>/bin/fb_native.dll  <build>/lib/fb_native.lib
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}/lib")
endforeach()

add_library(fb_native SHARED
  src/api_device.cpp
  src/api_kernels.cpp
  src/error.cpp
  cuda/kernels.cu
  trt/trt_runtime.cpp
  trt/api_engine.cpp
)
target_include_directories(fb_native
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/cuda" "${CMAKE_CURRENT_SOURCE_DIR}/trt"
          "${TENSORRT_INCLUDE_DIR}")
target_compile_definitions(fb_native PRIVATE FB_NATIVE_BUILD NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
set_target_properties(fb_native PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  WINDOWS_EXPORT_ALL_SYMBOLS OFF)
target_compile_options(fb_native PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/EHsc /W4 /wd4100 /wd4324 /utf-8>>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcompiler=/utf-8>)

target_link_libraries(fb_native PRIVATE CUDA::cudart_static)

# --- install (what the cmake crate picks up) ----------------------------------------------
install(TARGETS fb_native RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION bin)
install(FILES include/fb_native.h DESTINATION include)
if(MSVC)
  install(FILES $<TARGET_PDB_FILE:fb_native> DESTINATION bin OPTIONAL)
endif()

# --- tests ----------------------------------------------------------------------------------
if(FB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
