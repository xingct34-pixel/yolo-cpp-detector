#!/usr/bin/env bash
# YOLO 检测部署 | scripts/demo.sh
# 把常用的 CPU 视频演示参数集中在一个命令里。准备好本地模型和视频后，
# 运行 yolo_detector 的无窗口模式，把这次实际输出的统计写到 results/demo.json。


set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x build-local/yolo_detector ]]; then bash scripts/check.sh release; fi
mkdir -p results


# 模块：准备输入
# 先确定本地权重与视频存在，再把路径作为参数交给检测程序。


model=models/yolo11n.onnx
[[ -f "$model" ]] || model=yolo11n.onnx
if [[ ! -f "$model" || ! -f test.mp4 ]]; then echo 'Place the YOLO11n ONNX model and test.mp4 as documented in README.' >&2; exit 2; fi
exec build-local/yolo_detector --model "$model" --video test.mp4 --headless --queue-policy block --max-frames 150 --report results/demo.json "$@"
