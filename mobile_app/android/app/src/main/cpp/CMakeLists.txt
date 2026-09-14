cmake_minimum_required(VERSION 3.22.1)

# Project name must be appmodules for React Native New Architecture (Fabric / TurboModules)
project(appmodules LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 1. Build React Native appmodules (provides PlatformConstants, DefaultTurboModuleManagerDelegate, etc.)
if(EXISTS "${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake")
    include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)
endif()

# 2. Build BSmart VLM JNI bridge (bsmart_llama_vlm)
option(BSMART_LLAMA_WITH_LLAMA_CPP "Build the BSmart VLM JNI bridge against llama.cpp/libmtmd" OFF)
set(BSMART_LLAMA_CPP_DIR "" CACHE PATH "Path to a local ggml-org/llama.cpp checkout")

add_library(bsmart_llama_vlm SHARED
        llama/llama_vlm_jni.cpp)

target_link_libraries(bsmart_llama_vlm
        android
        log)

if(BSMART_LLAMA_WITH_LLAMA_CPP)
    if(NOT EXISTS "${BSMART_LLAMA_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "BSMART_LLAMA_CPP_DIR must point to a llama.cpp checkout when BSMART_LLAMA_WITH_LLAMA_CPP=ON")
    endif()
    if(NOT EXISTS "${BSMART_LLAMA_CPP_DIR}/tools/mtmd/mtmd.h")
        message(FATAL_ERROR "llama.cpp checkout does not contain tools/mtmd/mtmd.h: ${BSMART_LLAMA_CPP_DIR}")
    endif()

    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)

    add_subdirectory("${BSMART_LLAMA_CPP_DIR}" llama_cpp_build)

    target_compile_definitions(bsmart_llama_vlm PRIVATE BSMART_LLAMA_WITH_LLAMA_CPP=1)
    target_include_directories(bsmart_llama_vlm PRIVATE
            "${BSMART_LLAMA_CPP_DIR}/include"
            "${BSMART_LLAMA_CPP_DIR}/ggml/include"
            "${BSMART_LLAMA_CPP_DIR}/tools/mtmd")
    target_link_libraries(bsmart_llama_vlm
            llama
            mtmd)
endif()
