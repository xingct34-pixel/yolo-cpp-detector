# 项目架构说明

## 一、项目整体架构

本项目使用 C++ 实现视频目标检测，采用模块化设计，将视频读取、推理执行、结果显示、模型后端和性能统计分开。

主要模块：

| 模块              | 文件                                | 主要职责                         |
| --------------- | --------------------------------- | ---------------------------- |
| 程序入口            | `src/main.cpp`                    | 解析参数、选择后端、启动程序               |
| 流水线             | `pipeline.h / pipeline.cpp`       | 协调视频读取、推理和显示                 |
| 线程安全队列          | `thread_safe_queue.h`             | 在不同线程之间传递数据                  |
| 检测模块            | `detector.h / detector.cpp`       | 图像预处理、模型调用、结果后处理             |
| 统一后端接口          | `inference_backend.h`             | 规定推理后端必须实现的接口                |
| ONNX Runtime 后端 | `ort_backend.h / ort_backend.cpp` | 加载 ONNX 模型并执行推理              |
| TensorRT 后端     | `trt_backend.h / trt_backend.cpp` | 加载 Engine 并执行 GPU 推理         |
| 性能统计            | `benchmark.h / benchmark.cpp`     | 统计推理延迟                       |
| INT8 校准         | `tools/int8_calibrator.*`         | 为 INT8 Engine 构建提供校准数据       |
| Engine 构建       | `tools/build_engine.cpp`          | 将 ONNX 模型转换为 TensorRT Engine |

## 二、程序执行流程

```text
main.cpp
   |
   v
创建 Pipeline
   |
   v
pipeline.run()
   |
   +--> readLoop()
   |       |
   |       v
   |   frame_queue_
   |       |
   |       v
   +--> inferenceLoop()
   |       |
   |       v
   |    Detector
   |       |
   |       v
   |  InferenceBackend
   |       |
   |       v
   |   result_queue_
   |       |
   |       v
   +--> display()
```

三个阶段各自负责不同的工作：

1. `readLoop()`：从视频源读取帧。
2. `inferenceLoop()`：取出帧并执行目标检测。
3. `display()`：取出结果并显示。

它们通过队列交换数据，而不是直接操作对方的内部数据。

## 三、Pipeline 的数据结构

### 1. FrameData

`FrameData` 保存原始视频帧及其编号。

* `frame_id`：记录帧的编号。
* `frame`：保存对应的图像。

编号可以用于日志、统计和后续多推理线程场景下的结果追踪。

### 2. ResultData

`ResultData` 保存检测处理后的图像及其编号。

它与 `FrameData` 的用途不同：前者传递待处理的数据，后者传递已经处理的数据。

### 3. 为什么需要两个队列？

`frame_queue_` 负责把视频读取线程的数据传递给推理线程。

`result_queue_` 负责把推理线程的数据传递给显示线程。

两个队列分别对应不同阶段，可以减少线程之间的直接耦合。

## 四、线程安全队列

文件：`include/thread_safe_queue.h`

线程安全队列解决多个线程同时访问共享队列时的同步问题。

| 成员           | 作用            |
| ------------ | ------------- |
| `queue_`     | 存储待处理的数据      |
| `mutex_`     | 保护队列的共享状态     |
| `not_empty_` | 通知消费者队列状态发生变化 |
| `max_size_`  | 限制队列容量        |
| `closed_`    | 标记队列是否关闭      |

### `push()`

将数据放入队列。队列关闭后不再接受新数据；队列已满时，当前设计会先丢弃最旧的数据。

### `pop()`

当队列为空时，消费者可以通过条件变量等待，而不是不断检查队列。队列有数据时取出一项；队列关闭且没有剩余数据时返回 `false`。

### `close()`

标记队列关闭，并唤醒等待中的消费者，使线程有机会正常退出。

锁应只保护必要的共享状态操作，不应在持锁期间执行耗时的模型推理。

## 五、Detector 检测流程

文件：`src/detector.cpp`

```text
输入图像
   |
   v
preprocess（预处理）
   |
   v
backend_->infer（模型推理）
   |
   v
postprocess（后处理）
   |
   v
绘制检测结果
```

### 1. 预处理

主要工作包括：

* 使用 Letterbox 调整图像尺寸，并尽量保持原始宽高比。
* 记录缩放比例和填充位置。
* 将 OpenCV 常用的 BGR 通道顺序转换为 RGB。
* 将像素值归一化到模型需要的数值范围。
* 将图像内存布局从 HWC 转为 CHW。
* 将图像转换为模型需要的浮点数输入张量。

### 2. 推理

Detector 通过 `InferenceBackend` 调用底层后端。这样，图像处理流程不需要分别为 ONNX Runtime 和 TensorRT 重写一遍。

### 3. 后处理

当前代码假设 YOLO 输出为 `[1, 84, 8400]`：

* 4 个边界框参数。
* 80 个类别分数。
* 8400 个候选预测。

后处理需要完成置信度筛选、坐标映射、NMS（非极大值抑制）和结果绘制。

更换模型时必须核对实际输出格式，不能仅凭模型名称假定形状一致。

## 六、统一推理后端

文件：`include/inference_backend.h`

`InferenceBackend` 定义所有推理后端需要实现的共同接口。

```text
             Detector
                 |
                 v
       InferenceBackend
            /       \
           v         v
      OrtBackend  TrtBackend
```

这样，Detector 只依赖统一接口，不必在检测流程中写入大量特定框架的代码。

### ONNX Runtime 后端

主要职责：

1. 初始化 Runtime 环境。
2. 加载 ONNX 模型并创建 Session。
3. 获取模型输入、输出信息。
4. 创建输入张量。
5. 调用 `session.Run()` 执行推理。
6. 读取模型输出。

CUDA Execution Provider 可以将支持的模型计算交给 GPU 执行，但前提是依赖版本和硬件兼容。

### TensorRT 后端

主要职责：

1. 加载序列化的 Engine。
2. 创建执行 Context。
3. 分配 GPU 输入、输出缓冲区。
4. 将输入数据传输到 GPU。
5. 提交推理任务。
6. 等待必要的 GPU 操作完成。
7. 将输出传回 CPU。

常见概念：

* Runtime：负责加载 Engine 的运行时环境。
* Engine：经过优化的推理执行计划。
* Context：保存一次执行所需的状态。
* Stream：规定 GPU 操作执行顺序的队列。
* H2D：从主机内存传到 GPU 内存。
* D2H：从 GPU 内存传回主机内存。

TensorRT Engine 需要与目标运行环境兼容。

## 七、Benchmark 性能统计

文件：`src/benchmark.cpp`

Benchmark 收集多次推理的耗时，计算平均值、P50、P95、最小值和最大值。

* 平均延迟用于了解总体平均表现。
* P50 反映典型样本的中位数耗时。
* P95 反映较慢的那部分样本，有助于分析延迟波动。
* 最小值和最大值用于观察性能范围。

必须区分单次模型推理耗时和整条 Pipeline 的耗时。视频读取、排队、丢帧和显示都可能影响端到端性能。

## 八、INT8 校准与 Engine 构建

### INT8 校准

文件：`tools/int8_calibrator.h / tools/int8_calibrator.cpp`

校准器向 TensorRT 提供具有代表性的输入数据，帮助构建 INT8 Engine。校准数据应尽可能接近实际部署场景。

校准结果需要通过检测精度测试验证。量化可能提升执行效率，但也可能损失精度。

### Engine 构建

文件：`tools/build_engine.cpp`

构建流程：

```text
ONNX 模型
   |
   v
ONNX Parser（模型解析）
   |
   v
Network（网络定义）
   |
   v
Builder + BuilderConfig
   |
   +--> FP32
   +--> FP16
   +--> INT8 + 校准
   |
   v
序列化 Engine
```

Engine 构建属于离线准备工作，不应该在每一帧视频推理时重复执行。

## 九、CMake 构建配置

文件：`CMakeLists.txt`

CMake 决定哪些源文件参与编译、头文件从哪里查找、程序需要链接哪些库。

当前设计通过 `ENABLE_TENSORRT` 控制 TensorRT 相关目标的构建。

* 关闭 TensorRT 时，目标是构建 ONNX Runtime 应用。
* 启用 TensorRT 时，还需要 TensorRT 开发头文件及相关链接库。
* `build_engine` 是独立可执行程序，不应与主程序的 `main.cpp` 混在同一个目标中。

需要保证条件编译、源文件列表和后端创建逻辑相互一致，才能让不同配置都正常构建。

## 十、当前需要验证的事项

代码写完不代表功能已经验证。正式运行前应检查：

1. 所有头文件与源文件的声明、定义是否一致。
2. TensorRT 关闭时，ORT 版本能否独立编译。
3. TensorRT 开启时，相关头文件、库和 API 是否匹配。
4. 模型的输入、输出形状是否与 Detector 假设一致。
5. 队列关闭、线程退出和窗口退出逻辑是否正确。
6. Benchmark 的统计结果是否符合实际测量范围。
7. FP16、INT8 的检测精度是否满足预期。
8. 不同后端的延迟和端到端性能是否经过实际测试。

项目架构应以代码实际实现和测试结果为准。
