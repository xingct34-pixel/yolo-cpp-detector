# YOLO 检测部署 | CMakeLists.txt
# CMake 用本文件把源文件、头文件和推理库连成可执行程序及测试目标。
# 先看 ONNXRUNTIME_ROOT 怎样定位头文件与动态库，再看 yolo_detector 组合了哪些 .cpp；
# TensorRT 工具和部分 Python 验收由相应构建选项决定是否加入。


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

find_package(Threads REQUIRED)
if(ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
endif()


# ============================================================
# ONNX Runtime
# ============================================================

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set.\n"
        "Example:\n"
        "cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
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


# 模块：核心源文件
# 图片与视频共用这组实现，选中 TensorRT 时才追加对应源文件。


set(APP_SOURCES

    src/main.cpp

    src/detector.cpp

    src/pipeline.cpp

    src/ort_backend.cpp

    src/benchmark.cpp
    src/yolo_processing.cpp
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

    Threads::Threads
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
        CUDA::cudart
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


# ============================================================
# 自动化测试
# ============================================================

include(CTest)


# 模块：可选 Python 测试
# 把安装了量化依赖的解释器显式交给 CTest，才能实际导入 ORT 等库。


set(YOLO_QUANT_PYTHON "" CACHE FILEPATH "Optional Python interpreter with requirements-quantization.txt")
if(BUILD_TESTING AND YOLO_QUANT_PYTHON)
    add_test(NAME quantization_preprocessing
        COMMAND "${YOLO_QUANT_PYTHON}" "${PROJECT_SOURCE_DIR}/tests/test_quantization.py")
    set_tests_properties(quantization_preprocessing PROPERTIES TIMEOUT 30)
endif()

if(BUILD_TESTING)
    add_executable(
        thread_safe_queue_test
        tests/test_thread_safe_queue.cpp
    )

    target_include_directories(
        thread_safe_queue_test
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME thread_safe_queue_test
        COMMAND thread_safe_queue_test
    )

    set_tests_properties(
        thread_safe_queue_test
        PROPERTIES TIMEOUT 5
    )

    get_filename_component(
        ONNXRUNTIME_LIBRARY_DIR
        "${ONNXRUNTIME_LIBRARY}"
        DIRECTORY
    )

    function(add_invalid_queue_size_test test_name queue_value)
        add_test(
            NAME "${test_name}"
            COMMAND "${CMAKE_COMMAND}"
                "-DYOLO_EXECUTABLE=$<TARGET_FILE:yolo_detector>"
                "-DORT_LIBRARY_DIR=${ONNXRUNTIME_LIBRARY_DIR}"
                "-DQUEUE_VALUE=${queue_value}"
                "-P"
                "${PROJECT_SOURCE_DIR}/tests/test_invalid_queue_size.cmake"
        )
        set_tests_properties("${test_name}" PROPERTIES TIMEOUT 25)
    endfunction()

    add_invalid_queue_size_test(cli_reject_queue_zero "0")
    add_invalid_queue_size_test(cli_reject_queue_text "abc")
    add_invalid_queue_size_test(cli_reject_queue_suffix "3junk")
    add_invalid_queue_size_test(cli_reject_queue_negative "-1")

    add_test(
        NAME cli_reject_missing_video
        COMMAND "${CMAKE_COMMAND}"
            "-DYOLO_EXECUTABLE=$<TARGET_FILE:yolo_detector>"
            "-DORT_LIBRARY_DIR=${ONNXRUNTIME_LIBRARY_DIR}"
            "-DPROJECT_ROOT=${PROJECT_SOURCE_DIR}"
            "-P"
            "${PROJECT_SOURCE_DIR}/tests/test_invalid_video.cmake"
    )

    set_tests_properties(
        cli_reject_missing_video
        PROPERTIES TIMEOUT 20
    )
endif()

if(BUILD_TESTING)
    add_executable(yolo_processing_test tests/test_processing.cpp src/yolo_processing.cpp)
    target_include_directories(yolo_processing_test PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(yolo_processing_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME yolo_processing_test COMMAND yolo_processing_test)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME cli_validation COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test_cli.py $<TARGET_FILE:yolo_detector>)
    set_tests_properties(cli_validation PROPERTIES TIMEOUT 20)
endif()
option(ENABLE_SANITIZERS "Enable application ASan and UBSan" OFF)
if(ENABLE_SANITIZERS)
    foreach(target yolo_detector thread_safe_queue_test yolo_processing_test)
        if(TARGET ${target})
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(benchmark_test tests/test_benchmark.cpp src/benchmark.cpp)
    target_include_directories(benchmark_test PRIVATE include)
    add_test(NAME benchmark_test COMMAND benchmark_test)
    if(ENABLE_SANITIZERS)
        target_compile_options(benchmark_test PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(benchmark_test PRIVATE -fsanitize=address,undefined)
    endif()
endif()


# 模块：精度与算子诊断：标准库评估测试和复用现有CPU后端的COCO预测导出器


add_executable(yolo_coco_predict tools/coco_predict.cpp src/ort_backend.cpp src/yolo_processing.cpp)
target_include_directories(yolo_coco_predict PRIVATE include ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(yolo_coco_predict PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY} Threads::Threads)
if(ENABLE_SANITIZERS)
    target_compile_options(yolo_coco_predict PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(yolo_coco_predict PRIVATE -fsanitize=address,undefined)
endif()
if(BUILD_TESTING)
    add_test(NAME detection_evaluation COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test_evaluation.py)
    set_tests_properties(detection_evaluation PROPERTIES TIMEOUT 20)
    add_test(NAME ort_profile_summary COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test_profile.py)
    set_tests_properties(ort_profile_summary PROPERTIES TIMEOUT 20)
endif()


# 模块：官方评估交叉验证：可选的项目Python环境，不触发下载或系统安装


if(BUILD_TESTING AND YOLO_QUANT_PYTHON)
    add_test(NAME coco_reference COMMAND ${YOLO_QUANT_PYTHON} ${PROJECT_SOURCE_DIR}/tests/test_coco_reference.py)
    set_tests_properties(coco_reference PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
endif()
