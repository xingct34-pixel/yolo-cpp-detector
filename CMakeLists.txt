cmake_minimum_required(VERSION 3.18)

project(yolo_cpp_detector)

# ==================== C++ 标准 ====================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==================== 构建选项 ====================

# ON  ：编译 TensorRT 后端
# OFF ：只编译 ONNX Runtime 后端
#
# 当前 GTX 1050 环境没有 TensorRT 开发库，
# 因此以后租用合适 GPU 后再开启。
option(ENABLE_TENSORRT "Build TensorRT backend" OFF)

# ==================== OpenCV ====================

find_package(OpenCV REQUIRED)

# ==================== ONNX Runtime ====================

# 根据本机实际安装路径修改
set(ONNXRUNTIME_ROOT
    "/home/xct/onnxruntime-linux-x64-gpu-1.21.1"
)

set(ONNXRUNTIME_INCLUDE_DIR
    "${ONNXRUNTIME_ROOT}/include"
)

set(ONNXRUNTIME_LIBRARY_DIR
    "${ONNXRUNTIME_ROOT}/lib"
)

# ==================== 头文件 ====================

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

# ==================== 库搜索路径 ====================

link_directories(
    ${ONNXRUNTIME_LIBRARY_DIR}
)

# ==================== 核心源文件 ====================

set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/pipeline.cpp
    src/ort_backend.cpp
    src/benchmark.cpp
)

# ==================== TensorRT ====================

if(ENABLE_TENSORRT)

    find_path(TENSORRT_INCLUDE_DIR
        NvInfer.h
        PATHS
            /usr/include
            /usr/local/include
            /usr/include/x86_64-linux-gnu
            /usr/local/TensorRT/include
    )

    find_library(TENSORRT_LIBRARY
        nvinfer
        PATHS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            /usr/local/TensorRT/lib
    )

    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR
            "TensorRT was requested but not found."
        )
    endif()

    message(STATUS
        "TensorRT enabled: ${TENSORRT_INCLUDE_DIR}"
    )

    include_directories(
        ${TENSORRT_INCLUDE_DIR}
    )

    list(APPEND SOURCES
        src/trt_backend.cpp
        tools/build_engine.cpp
    )

endif()

# ==================== 主程序 ====================

add_executable(yolo_detector ${SOURCES})

# ==================== 链接库 ====================

target_link_libraries(
    yolo_detector
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

if(ENABLE_TENSORRT)

    target_link_libraries(
        yolo_detector
        PRIVATE
        ${TENSORRT_LIBRARY}
    )

endif()

# ==================== 编译选项 ====================

target_compile_options(
    yolo_detector
    PRIVATE
    -Wall
    -Wextra
    -O2
)

# ==================== 输出信息 ====================

message(STATUS "====================================")
message(STATUS "YOLO C++ Inference Deployment")
message(STATUS "C++ Standard : C++17")
message(STATUS "OpenCV       : ${OpenCV_VERSION}")
message(STATUS "TensorRT     : ${ENABLE_TENSORRT}")
message(STATUS "====================================")
