cmake_minimum_required(VERSION 3.18)

project(yolo_cpp_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# 构建选项
# ============================================================

# 默认关闭 TensorRT。
# 当前电脑没有 TensorRT 开发环境时，
# 仍然可以正常编译 ORT 版本。
option(ENABLE_TENSORRT
    "Build TensorRT backend and engine builder"
    OFF)

# ONNX Runtime 安装目录。
# 实际使用时通过：
#
# cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
#
# 指定。
set(ONNXRUNTIME_ROOT
    ""
    CACHE PATH
    "ONNX Runtime root directory")

# TensorRT 安装目录。
# 只有 ENABLE_TENSORRT=ON 时需要。
set(TENSORRT_ROOT
    ""
    CACHE PATH
    "TensorRT root directory")


# ============================================================
# 基础依赖
# ============================================================

find_package(OpenCV REQUIRED)

find_package(CUDAToolkit REQUIRED)


# ============================================================
# ONNX Runtime
# ============================================================

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set.\n"
        "Example:\n"
        "cmake .. -DONNXRUNTIME_ROOT=/home/xct/onnxruntime-linux-x64-gpu-1.18.0")
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
    REQUIRED
)

find_library(
    ONNXRUNTIME_LIBRARY
    onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
    REQUIRED
)


# ============================================================
# 主程序：YOLO Detector
# ============================================================

set(APP_SOURCES

    src/main.cpp

    src/detector.cpp

    src/pipeline.cpp

    src/ort_backend.cpp

    src/benchmark.cpp
)


# ============================================================
# TensorRT 后端
# ============================================================

if(ENABLE_TENSORRT)

    if(NOT TENSORRT_ROOT)
        message(FATAL_ERROR
            "ENABLE_TENSORRT=ON but TENSORRT_ROOT is not set.")
    endif()

    find_path(
        TENSORRT_INCLUDE_DIR
        NvInfer.h
        PATHS
            ${TENSORRT_ROOT}/include
        REQUIRED
    )

    find_library(
        TENSORRT_LIBRARY
        nvinfer
        PATHS
            ${TENSORRT_ROOT}/lib
        REQUIRED
    )

    list(APPEND APP_SOURCES
        src/trt_backend.cpp
    )

endif()


# ============================================================
# YOLO Detector 可执行文件
# ============================================================

add_executable(
    yolo_detector
    ${APP_SOURCES}
)


target_include_directories(
    yolo_detector
    PRIVATE

    ${PROJECT_SOURCE_DIR}/include

    ${OpenCV_INCLUDE_DIRS}

    ${ONNXRUNTIME_INCLUDE_DIR}
)


target_link_libraries(
    yolo_detector
    PRIVATE

    ${OpenCV_LIBS}

    ${ONNXRUNTIME_LIBRARY}

    CUDA::cudart
)


# ============================================================
# TensorRT 编译配置
# ============================================================

if(ENABLE_TENSORRT)

    target_include_directories(
        yolo_detector
        PRIVATE

        ${TENSORRT_INCLUDE_DIR}
    )

    target_link_libraries(
        yolo_detector
        PRIVATE

        ${TENSORRT_LIBRARY}
    )

    target_compile_definitions(
        yolo_detector
        PRIVATE

        ENABLE_TENSORRT
    )

endif()


# ============================================================
# TensorRT Engine Builder
# ============================================================
#
# 只有启用 TensorRT 时才构建。
#
# build_engine：
#
# ONNX
#   ↓
# Parser（模型翻译员）
#   ↓
# Network（生产线设计图）
#   ↓
# Builder（建厂工程师）
#   ↓
# Engine（已经建好的生产线）

if(ENABLE_TENSORRT)

    find_path(
        NVONNXPARSER_INCLUDE_DIR
        NvOnnxParser.h
        PATHS
            ${TENSORRT_ROOT}/include
        REQUIRED
    )

    find_library(
        NVONNXPARSER_LIBRARY
        nvonnxparser
        PATHS
            ${TENSORRT_ROOT}/lib
        REQUIRED
    )


    add_executable(
        build_engine

        tools/build_engine.cpp

        tools/int8_calibrator.cpp
    )


    target_include_directories(
        build_engine
        PRIVATE

        ${PROJECT_SOURCE_DIR}/tools

        ${TENSORRT_INCLUDE_DIR}

        ${NVONNXPARSER_INCLUDE_DIR}

        ${CUDAToolkit_INCLUDE_DIRS}

        ${OpenCV_INCLUDE_DIRS}
    )


    target_link_libraries(
        build_engine
        PRIVATE

        ${TENSORRT_LIBRARY}

        ${NVONNXPARSER_LIBRARY}

        CUDA::cudart

        ${OpenCV_LIBS}
    )

endif()


# ============================================================
# 编译信息
# ============================================================

message(STATUS "")
message(STATUS "========== YOLO C++ Detector ==========")
message(STATUS "OpenCV            : ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime      : ${ONNXRUNTIME_LIBRARY}")
message(STATUS "TensorRT Enabled  : ${ENABLE_TENSORRT}")

if(ENABLE_TENSORRT)
    message(STATUS "TensorRT Root     : ${TENSORRT_ROOT}")
endif()

message(STATUS "========================================")
message(STATUS "")
