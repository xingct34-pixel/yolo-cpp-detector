#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x build-local/yolo_detector ]]; then bash scripts/check.sh release; fi
mkdir -p results
model=models/yolo11n.onnx
[[ -f "$model" ]] || model=yolo11n.onnx
if [[ ! -f "$model" || ! -f test.mp4 ]]; then echo 'Place the YOLO11n ONNX model and test.mp4 as documented in README.' >&2; exit 2; fi
exec build-local/yolo_detector --model "$model" --video test.mp4 --headless --queue-policy block --max-frames 150 --report results/demo.json "$@"
