if(NOT DEFINED YOLO_EXECUTABLE OR
   NOT DEFINED ORT_LIBRARY_DIR OR
   NOT DEFINED PROJECT_ROOT)
    message(FATAL_ERROR "缺少测试所需参数")
endif()

set(missing_video "/tmp/yolo_cpp_detector_missing_input_test.mp4")

if(EXISTS "${missing_video}")
    message(FATAL_ERROR "测试视频路径意外存在：${missing_video}")
endif()

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
