#!/usr/bin/env bash
# YOLO 检测部署 | scripts/check.sh
# 功能：本地构建与测试入口。根据 release 或 sanitize 找到对应构建目录和现有 ORT，
# 依次配置 CMake、编译和运行 CTest，失败时可以按阶段查看 results 下的日志。
# 目的：在相同构建模式下重现编译与测试，遇到依赖问题时能定位失败阶段。


set -euo pipefail
cd "$(dirname "$0")/.."


# 模块：构建模式
# release 和 sanitize 用不同的目录；不会把调试选项误用到发布版产物上。
mode="${1:-release}"
case "$mode" in release) build=build-local; kind=Release; sanitize=OFF;; sanitize) build=build-local-sanitize; kind=Debug; sanitize=ON;; *) echo 'release or sanitize required' >&2; exit 2;; esac


# 模块：定位 ORT
# 头文件用于编译，动态库用于链接和运行。优先采用调用者指定的安装路径。
runtime="${ONNXRUNTIME_ROOT:-}"
if [[ -z "$runtime" ]]; then
  for candidate in "$HOME"/onnxruntime-linux-x64-*; do
    if [[ -f "$candidate/include/onnxruntime_cxx_api.h" && -f "$candidate/lib/libonnxruntime.so" ]]; then runtime="$candidate"; fi
  done
fi
if [[ -z "$runtime" ]]; then echo 'Set ONNXRUNTIME_ROOT to an existing ONNX Runtime directory.' >&2; exit 2; fi
mkdir -p results


# 模块：配置与验证
# 配置阶段定位头文件和库，构建阶段生成二进制，CTest 调用前面注册的测试；各阶段日志分别保留。
# 若配置阶段报缺 onnxruntime_cxx_api.h，应查上面的 ORT 根目录；只有配置和编译成功才会走到 CTest。这三种失败对应不同的排查入口。
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE="$kind" -DENABLE_TENSORRT=OFF -DENABLE_SANITIZERS="$sanitize" -DONNXRUNTIME_ROOT="$runtime" >"results/configure-$mode.log" 2>&1
cmake --build "$build" -j2 >"results/build-$mode.log" 2>&1
ctest --test-dir "$build" --output-on-failure | tee "results/test-$mode.log"
