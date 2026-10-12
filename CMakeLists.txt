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

/*
1. 为什么有 ENABLE_TENSORRT？

TensorRT 在这个项目里是可选后端，ORT 才是基础后端。打开开关后，下面这些事会一起发生：

多编译一个 trt_backend.cpp
多一组头文件目录和 nvinfer 库
给代码传入 ENABLE_TENSORRT 宏
多构建一个 build_engine 工具

一个开关控制整条链路，不用手动改多处。

2. 为什么默认 OFF？

TensorRT 需要单独安装，版本要和 CUDA、驱动严格匹配，并且只能用于 NVIDIA GPU。如果默认 ON，没装 TensorRT 的机器在 find_path(... REQUIRED) 这一步就会直接配置失败。默认 OFF 能保证"任何装了 OpenCV、ORT、CUDA 的机器都能先编译出 ORT 版本"。

3. 为什么 trt_backend.cpp 是条件编译？

这个文件里 #include <NvInfer.h>，没有 TensorRT 头文件就会编译报错。所以它只在开关打开时才加入 APP_SOURCES。

同时 target_compile_definitions(... ENABLE_TENSORRT) 会让 C++ 代码里可以写：

cpp
#ifdef ENABLE_TENSORRT
    // 创建 TensorRT 后端
#endif

这样 main.cpp 或 pipeline.cpp 在没有 TensorRT 时也能正常编译，只是少了那个分支。

4. 为什么 build_engine 是单独的 executable？

两者的职责和生命周期不同：

	build_engine	yolo_detector
作用	ONNX → .engine（离线转换）	加载模型做推理（在线运行）
运行频率	模型变一次跑一次，耗时可能几分钟	每次检测都运行
额外依赖	nvonnxparser、INT8 校准器	不需要 parser

把 parser 和校准器这类只在建 engine 时才用的东西，塞进推理程序里没有意义，还会增加依赖和体积。

5. 为什么 build_engine.cpp 不能和 main.cpp 一起编译？

最直接的原因是两个文件各有一个 main()。一个可执行文件只能有一个程序入口，放在一起链接时会报 multiple definition of main。

另外，它们依赖的库也不同（见上表），分开才能各取所需。

6. include directory 是干什么的？

target_include_directories 告诉编译器去哪里找头文件，作用于编译阶段。比如代码里写 #include <onnxruntime_cxx_api.h>，编译器就会到 ONNXRUNTIME_INCLUDE_DIR 里找。头文件只提供"声明"，相当于菜单：告诉你有哪些函数和类，但里面没有实现。

PRIVATE 表示这些路径只用于该目标自己，不会传递给依赖它的其他目标。

7. link library 是干什么的？

target_link_libraries 告诉链接器去哪里找函数的真正实现，作用于链接阶段。相当于后厨：菜单上有这道菜，但真正做出来要靠库里的二进制代码。

${OpenCV_LIBS}：图像读写、预处理
onnxruntime：ORT 推理
CUDA::cudart：CUDA 运行时（显存、stream 等）
nvinfer：TensorRT 推理（开关打开时）
nvonnxparser：只给 build_engine，用来解析 ONNX
*/
