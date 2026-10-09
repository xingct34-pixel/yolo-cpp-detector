# YOLO C++ 推理部署项目

这是一个使用 C++ 实现的 YOLO 目标检测推理部署项目，主要用于学习视频推理流水线、多线程编程、ONNX Runtime、CUDA、TensorRT 以及模型性能优化。

## 一、项目目标

本项目希望实现一套模块化的 C++ 目标检测程序，逐步掌握从模型加载、图像预处理、模型推理、结果后处理，到性能测试的完整流程。

主要学习内容：

* C++ 工程组织与 CMake 构建
* 多线程视频处理 Pipeline（流水线）
* ThreadSafeQueue（线程安全队列）
* ONNX Runtime（ONNX 模型推理运行框架）
* CUDA（GPU 计算平台）
* TensorRT（NVIDIA 推理优化框架）
* FP16、INT8 模型精度优化
* Benchmark（性能测试与统计）

注意：代码中包含 TensorRT 相关实现，但必须在匹配的 TensorRT 开发环境中编译和验证。不能因为代码已经写好，就认为所有功能都已经测试成功。

## 二、项目目录

```text
yolo-cpp-detector/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/                  # 头文件：声明类和函数
│   ├── benchmark.h
│   ├── detector.h
│   ├── inference_backend.h
│   ├── ort_backend.h
│   ├── pipeline.h
│   ├── thread_safe_queue.h
│   └── trt_backend.h
├── src/                      # 源文件：实现具体功能
│   ├── benchmark.cpp
│   ├── detector.cpp
│   ├── main.cpp
│   ├── ort_backend.cpp
│   ├── pipeline.cpp
│   └── trt_backend.cpp
├── tools/                    # 辅助工具
│   ├── build_engine.cpp
│   ├── int8_calibrator.h
│   └── int8_calibrator.cpp
├── models/                   # 存放模型文件
├── data/
│   └── coco.txt              # COCO 80 类名称
└── docs/
    └── architecture.md       # 项目架构说明
```

## 三、整体工作流程

```text
main.cpp（程序入口）
    |
    v
Pipeline（流水线调度）
    |
    +--> 读取视频
    |       |
    |       v
    |   frame_queue（原始帧队列）
    |       |
    |       v
    |   Detector（目标检测模块）
    |       |
    |       v
    |   InferenceBackend（统一推理接口）
    |       |
    |       +--> OrtBackend（ONNX Runtime 后端）
    |       |
    |       +--> TrtBackend（TensorRT 后端）
    |       |
    |       v
    |   result_queue（检测结果队列）
    |       |
    |       v
    |   显示检测结果
    |
    +--> Benchmark（性能统计）
```

TensorRT 模型构建是独立流程：

```text
ONNX 模型
    |
    v
build_engine（Engine 构建工具）
    |
    +--> FP32（32 位浮点精度）
    +--> FP16（16 位浮点精度）
    +--> INT8（8 位整数精度）
    |
    v
.engine 文件
    |
    v
TrtBackend（TensorRT 推理后端）
```

更详细的模块关系见 `docs/architecture.md`。

## 四、运行环境

项目涉及以下依赖：

* C++17 编译器
* CMake 3.18 或更高版本
* OpenCV
* ONNX Runtime
* CUDA Toolkit
* TensorRT（仅在构建 TensorRT 后端和 Engine 构建工具时需要）

实际运行还取决于显卡架构、驱动、CUDA、cuDNN、ONNX Runtime 和 TensorRT 版本之间的兼容性。

## 五、模型与测试视频

默认路径：

```text
models/yolo11n.onnx
data/coco.txt
test.mp4
```

TensorRT 模式还需要准备：

```text
models/yolo11n.engine
```

注意：

* `.onnx` 是模型交换格式文件。
* `.engine` 是 TensorRT 序列化后的推理引擎文件。
* 大型模型文件和生成的 Engine 文件默认不提交到 GitHub。
* `test.mp4` 需要自行准备。

当前 Detector 的实现假定模型输入尺寸为 `640 × 640`，输出形状为 `[1, 84, 8400]`。更换模型时，需要核对实际输入输出格式。

## 六、编译项目

先确保已经安装 OpenCV、CUDA Toolkit 和 ONNX Runtime。

在项目根目录执行：

```bash
mkdir -p build
cd build
```

这两条命令分别创建构建目录、进入构建目录。

然后配置项目：

```bash
cmake .. \
  -DONNXRUNTIME_ROOT=/你的/onnxruntime/实际路径 \
  -DENABLE_TENSORRT=OFF
```

将路径替换为实际 ONNX Runtime 安装目录。该目录下应有 `include/` 和 `lib/`。

再编译：

```bash
cmake --build . -j
```

预期生成主程序：

```text
yolo_detector
```

以上是预期构建流程，仍需结合实际环境解决编译问题并完成验证。

## 七、运行项目

查看命令行参数：

```bash
./yolo_detector --help
```

使用 ONNX Runtime CPU 推理：

```bash
./yolo_detector \
  --backend ort \
  --ort-device cpu \
  --video test.mp4 \
  --model models/yolo11n.onnx \
  --classes data/coco.txt
```

使用 ONNX Runtime CUDA 推理：

```bash
./yolo_detector \
  --backend ort \
  --ort-device cuda \
  --video test.mp4 \
  --model models/yolo11n.onnx \
  --classes data/coco.txt
```

CUDA 推理需要支持 CUDA Execution Provider（CUDA 执行提供器）的 ONNX Runtime，以及兼容的 GPU 依赖环境。

## 八、TensorRT Engine 构建

只有在安装了匹配的 TensorRT 开发库后，才可以启用 TensorRT 编译选项。

配置：

```bash
cmake .. \
  -DONNXRUNTIME_ROOT=/你的/onnxruntime/实际路径 \
  -DTENSORRT_ROOT=/你的/tensorrt/实际路径 \
  -DENABLE_TENSORRT=ON
```

编译：

```bash
cmake --build . -j
```

示例：构建 FP16 Engine。

```bash
./build_engine \
  --onnx models/yolo11n.onnx \
  --engine models/yolo11n.engine \
  --precision fp16
```

示例：构建 INT8 Engine。

```bash
./build_engine \
  --onnx models/yolo11n.onnx \
  --engine models/yolo11n.engine \
  --precision int8 \
  --calib-dir data/calibration \
  --calib-cache data/calibration.cache
```

INT8 校准需要具有代表性的图片数据集。生成的 Engine 还需要进行精度和性能验证。

使用 TensorRT 进行推理：

```bash
./yolo_detector \
  --backend trt \
  --engine models/yolo11n.engine \
  --video test.mp4 \
  --classes data/coco.txt
```

以上命令是预期使用方式，实际参数和功能仍须以最终审查、编译后的代码为准。

## 九、性能测试

项目中的 Benchmark 用于统计推理延迟：

* Average：平均延迟
* P50：中位数延迟
* P95：95 分位延迟
* Min：最小延迟
* Max：最大延迟

注意区分：

* **模型推理延迟**：一次模型执行花费的时间。
* **端到端延迟**：从输入帧到结果输出的整体耗时。
* **Pipeline FPS**：整个流水线每秒处理或输出的帧数。

这几个指标并不等价，队列丢帧、视频读取、模型初始化和显示都会影响整体表现。

## 十、项目学习路线

1. 理解 `main.cpp` 和 `Pipeline` 的调用关系。
2. 理解多线程、互斥锁、条件变量和线程安全队列。
3. 理解 Detector 的预处理、推理和后处理。
4. 理解 ONNX Runtime 的模型加载和推理过程。
5. 理解 CUDA 内存传输和 GPU 执行。
6. 理解 TensorRT Engine 构建与推理。
7. 理解 FP16、INT8 和校准流程。
8. 学习延迟、吞吐量和性能测试方法。

项目功能以实际编译、运行和测试结果为准，不以文档中的功能描述代替验证。
