#!/usr/bin/env bash
# YOLO 检测部署 | scripts/demo.sh
# 把常用的 CPU 视频演示参数集中在一个命令里。准备好本地模型和视频后，
# 运行 yolo_detector 的无窗口模式，把这次实际输出的统计写到 results/demo.json。
#
# 阅读地图（按源码顺序）
#   1 准备二进制 → 2 准备输入 → 3 运行视频


set -euo pipefail
cd "$(dirname "$0")/.."


# 模块 1：准备二进制
# 如果缺可执行文件，先走同一个 release 检查脚本，运行示例也能复用正式构建路径。


if [[ ! -x build-local/yolo_detector ]]; then bash scripts/check.sh release; fi
mkdir -p results


# 模块 2：准备输入
# 先确定本地权重与视频存在，再把路径作为参数交给检测程序。


model=models/yolo11n.onnx
[[ -f "$model" ]] || model=yolo11n.onnx
if [[ ! -f "$model" || ! -f test.mp4 ]]; then echo 'Place the YOLO11n ONNX model and test.mp4 as documented in README.' >&2; exit 2; fi


# 模块 3：运行视频
# block 让读取端等空位，headless 关闭窗口，便于核对读入的帧有没有完整走到输出。


exec build-local/yolo_detector --model "$model" --video test.mp4 --headless --queue-policy block --max-frames 150 --report results/demo.json "$@"
