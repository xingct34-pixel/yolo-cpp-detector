
cmake_minimum_required(VERSION 3.10)
project(yolo_v2)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/xct/onnxruntime-linux-x64-1.20.1/include)

link_directories(/home/xct/onnxruntime-linux-x64-1.20.1/lib)

add_executable(yolo_v2 main.cpp detector.cpp)
target_link_libraries(yolo_v2
    ${OpenCV_LIBS}
    onnxruntime
)
