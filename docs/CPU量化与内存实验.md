# CPU量化与内存实验

## 数据流

离线Python工具读取已有ONNX和本地视频，执行与C++一致的Letterbox、RGB、CHW、归一化，完成静态INT8 QDQ量化。校准默认使用前8帧，数值检查使用随后8帧。生成模型保留FP32输入输出合同，由现有C++ ORT后端加载。量化工具未引入到在线C++运行时。

检查记录模型/视频/输入张量SHA-256、量化节点数量、模型体积、预热后的Session.run耗时及原始输出数值差异。不同帧编号不保证内容不同，报告同时检查校准与检查张量的内容交集。

## 复现

Windows启动WSL时明确选择 `wsl.exe -d Ubuntu-24.04`，然后在项目根目录执行以下Linux命令。仅向项目虚拟环境安装依赖。

```bash
python3 -m venv .venv-quant
.venv-quant/bin/python -m pip install -r requirements-quantization.txt
.venv-quant/bin/python tests/test_quantization.py
.venv-quant/bin/python tools/cpu_quantize.py \
  --model models/yolo11n.onnx --video test.mp4 \
  --output results/quant-experiment
build-local/yolo_detector --model results/quant-experiment/model.int8.onnx \
  --video test.mp4 --headless --max-frames 16 --warmup 2 \
  --report results/int8-native.json
```

需已有模型、类别文件、视频及已构建的C++程序。工具不会下载这些素材。输出目录需选用新目录以保存前次证据。

可选将预处理测试接入CTest，配置时增加 `-DYOLO_QUANT_PYTHON="$PWD/.venv-quant/bin/python"`。基础C++构建无需Python量化依赖。

## 测量与优化

`prepare_yolo()` 原先创建OpenCV blob，再复制进vector。现在为最终vector分配内存，再让OpenCV直接写入该缓冲区。固定640输入下，省去一个3×640×640×4字节，即约4.69 MiB的额外张量缓冲及一次复制。每帧仍分配最终向量，未实现跨帧内存池。C++测试用非均匀图像逐元素对照原路径，覆盖横图、竖图、方图和尺寸取整。

`bench/native_resources.py` 从真实CLI启动独立子进程，交替测量改动前FP32、改动后FP32与可选INT8模型。每个子进程固定3个ORT线程、2帧预热，采样VmHWM/VmSwap。需自行保留改动前二进制。

```bash
python3 bench/native_resources.py --before-binary /path/to/before/yolo_detector \
  --candidate-model results/quant-experiment/model.int8.onnx \
  --output results/native-resource-experiment
```

进程内存包含模型、ORT、OpenCV和缓冲区。Python Session.run耗时与C++端到端耗时范围不同。主机其他负载和页缓存未隔离，小样本结果只能说明本轮实验。

## 验收边界

未标注视频只能验证量化、加载、有限数值输出及端到端执行。原始张量误差不是mAP，也不能证明检测精度保持。需要代表性校准集和带标注的独立验证集才能做精度验收。

本实验采用Conv/MatMul选择性量化及QDQ格式，不把整个图称为纯INT8计算。低精度的体积收益与速度收益分别报告，量化后变慢也保留结果。GPU、TensorRT、Jetson、剪枝与蒸馏不在此次验收范围。

依赖固定在requirements-quantization.txt。第三方wheel和环境不入库，安装包自带许可证保留于项目虚拟环境。量化模型和视频受gitignore排除。实测摘要见 [本轮结果](CPU扩展结果-2026-11-09.md)。

## 后续学习顺序

先读src/yolo_processing.cpp中的数据布局与缓冲区，再对照tools/cpu_quantize.py的prepare()。接着看Reader如何向量化器提供校准数据，以及QDQ模型为何保留浮点接口。最后查看数值误差、重复内容检测和bench/native_resources.py的独立进程测量。练习应围绕改变校准数据、解释变慢原因和核对计时范围逐项进行。
