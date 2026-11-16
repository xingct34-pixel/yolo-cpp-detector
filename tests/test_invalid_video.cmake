# YOLO 检测部署 | tests/test_invalid_video.cmake
# 功能：给检测程序一个明确不存在的视频路径，检查退出码和报错里是否含有这个路径。
# 这让 CTest 可以区分“成功报告读视频失败”和“程序在更早的地方意外退出”。
# 目的：验证错误信息能指出打不开的那段视频，使用者才知道要改哪个输入。


if(NOT DEFINED YOLO_EXECUTABLE OR
   NOT DEFINED ORT_LIBRARY_DIR OR
   NOT DEFINED PROJECT_ROOT)
    message(FATAL_ERROR "缺少测试所需参数")
endif()

set(missing_video "/tmp/yolo_cpp_detector_missing_input_test.mp4")

if(EXISTS "${missing_video}")
    message(FATAL_ERROR "测试视频路径意外存在：${missing_video}")
endif()


# 模块：准备其他有效输入
# 模型和类别表先确认存在，让接下来这一次失败能归因到视频路径。
if(EXISTS "${PROJECT_ROOT}/models/yolo11n.onnx")
    set(test_model "${PROJECT_ROOT}/models/yolo11n.onnx")
elseif(EXISTS "${PROJECT_ROOT}/yolo11n.onnx")
    set(test_model "${PROJECT_ROOT}/yolo11n.onnx")
else()
    message(FATAL_ERROR
        "缺少模型：models/yolo11n.onnx 或 yolo11n.onnx")
endif()

if(NOT EXISTS "${PROJECT_ROOT}/data/coco.txt")
    message(FATAL_ERROR "缺少 data/coco.txt")
endif()

set(runtime_library_path "${ORT_LIBRARY_DIR}")
if(DEFINED ENV{LD_LIBRARY_PATH} AND
   NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    string(APPEND runtime_library_path ":$ENV{LD_LIBRARY_PATH}")
endif()


# 模块：触发读视频错误
# 模型和类别表准备好之后才传入缺失的视频，避免被更早的错误挡住。
# 模型和类别文件得先能打开，程序才会尝试读视频。所以这条用例准备有效的前置输入，只把视频路径指向不存在的位置。
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "LD_LIBRARY_PATH=${runtime_library_path}"
        "${YOLO_EXECUTABLE}"
        --backend ort
        --ort-device cpu
        --video "${missing_video}"
        --model "${test_model}"
        --classes "${PROJECT_ROOT}/data/coco.txt"
        --queue-size 3
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    TIMEOUT 20
)

set(output "${stdout}\n${stderr}")

if(NOT "${exit_code}" STREQUAL "1")
    message(FATAL_ERROR
        "预期退出码为 1，实际为 '${exit_code}'。\n${output}")
endif()


# 模块：错误定位
# 错误消息必须指出那个不存在的文件，便于使用者找到出问题的输入。
string(FIND "${output}" "${missing_video}" path_position)
if(path_position EQUAL -1)
    message(FATAL_ERROR
        "错误信息没有包含缺失的视频路径。\n${output}")
endif()

string(FIND "${output}" "程序异常" error_position)
if(error_position EQUAL -1)
    message(FATAL_ERROR
        "没有找到预期的程序异常信息。\n${output}")
endif()

message(STATUS "无效视频路径被正确拒绝。")
