#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-release}"
case "$mode" in release) build=build-local; kind=Release; sanitize=OFF;; sanitize) build=build-local-sanitize; kind=Debug; sanitize=ON;; *) echo 'release or sanitize required' >&2; exit 2;; esac
runtime="${ONNXRUNTIME_ROOT:-}"
if [[ -z "$runtime" ]]; then
  for candidate in "$HOME"/onnxruntime-linux-x64-*; do
    if [[ -f "$candidate/include/onnxruntime_cxx_api.h" && -f "$candidate/lib/libonnxruntime.so" ]]; then runtime="$candidate"; fi
  done
fi
if [[ -z "$runtime" ]]; then echo 'Set ONNXRUNTIME_ROOT to an existing ONNX Runtime directory.' >&2; exit 2; fi
mkdir -p results
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE="$kind" -DENABLE_TENSORRT=OFF -DENABLE_SANITIZERS="$sanitize" -DONNXRUNTIME_ROOT="$runtime" >"results/configure-$mode.log" 2>&1
cmake --build "$build" -j2 >"results/build-$mode.log" 2>&1
ctest --test-dir "$build" --output-on-failure | tee "results/test-$mode.log"
