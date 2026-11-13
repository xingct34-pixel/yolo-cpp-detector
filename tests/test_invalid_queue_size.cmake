# YOLO 检测部署 | tests/test_invalid_queue_size.cmake
# CTest 把待测试的 --queue-size 字符串传进来。本文件启动程序并读取退出码和诊断文字，
# 确认程序因为指定参数不合法而退出，防止把缺少动态库等其他错误误当作通过。


if(NOT DEFINED YOLO_EXECUTABLE OR
   NOT DEFINED ORT_LIBRARY_DIR OR
   NOT DEFINED QUEUE_VALUE)
    message(FATAL_ERROR "缺少测试参数")
endif()

set(runtime_library_path "${ORT_LIBRARY_DIR}")
if(DEFINED ENV{LD_LIBRARY_PATH} AND
   NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    string(APPEND runtime_library_path ":$ENV{LD_LIBRARY_PATH}")
endif()


# 模块：运行被测程序
# 收集退出码和输出。仅看非零会把动态库缺失等启动故障也算成“拒绝输入”。


execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "LD_LIBRARY_PATH=${runtime_library_path}"
        "${YOLO_EXECUTABLE}"
        --queue-size "${QUEUE_VALUE}"
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    TIMEOUT 15
)

set(output "${stdout}\n${stderr}")

if("${exit_code}" STREQUAL "0")
    message(FATAL_ERROR
        "非法 queue-size='${QUEUE_VALUE}' 被错误接受。\n${output}")
endif()

string(FIND "${output}" "queue-size" message_position)
if(message_position EQUAL -1)
    message(FATAL_ERROR
        "程序退出失败，但没有找到 queue-size 错误信息。\n${output}")
endif()

message(STATUS
    "非法 queue-size='${QUEUE_VALUE}' 已正确拒绝。")
