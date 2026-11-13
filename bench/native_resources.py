# YOLO 检测部署 | bench/native_resources.py
# 通过真实命令行交替执行前后两个版本的程序或模型，同时记录耗时和进程资源。
# 哈希值标识这一轮实际测的输入与程序，方便把读到的数字和对应版本对上。


# 模块：原生资源测量：对比真实C++入口的速度及进程内存


"""Alternating before/after CPU measurements through the actual C++ CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


# 模块：文件校验：计算输入或程序摘要


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# 模块：执行入口：读取参数并组织该工具的完整流程


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-binary', type=Path, required=True)
    parser.add_argument('--after-binary', type=Path, default=ROOT / 'build-local/yolo_detector')
    parser.add_argument('--model', type=Path, default=ROOT / 'yolo11n.onnx')
    parser.add_argument('--candidate-model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000 or not 1 <= args.repeats <= 10:
        parser.error('invalid frames or repeats')
    if args.output.exists():
        parser.error('choose a new output directory')
    args.output.mkdir(parents=True)
    cases = [
        ('before', args.before_binary.resolve(), args.model.resolve()),
        ('after', args.after_binary.resolve(), args.model.resolve()),
    ]
    if args.candidate_model:
        cases.append(('int8', args.after_binary.resolve(), args.candidate_model.resolve()))
    report = {
        'passed': False,
        'threads': 3,
        'frames': args.frames,
        'sampling_interval_ms': 20,
        'runs': [],
        'note': 'RSS includes process/model/runtime. Host load and page cache are not isolated.',
    }
    try:


        # 模块：成对运行
        # 每轮都运行两个候选，保留各轮原始记录，方便查看波动。


        for repeat in range(args.repeats):
            order = cases if repeat % 2 == 0 else list(reversed(cases))
            for label, binary, model in order:
                name = f'{repeat}-{label}'
                result_path = args.output / (name + '.json')
                command = [
                    str(binary),
                    '--model',
                    str(model),
                    '--video',
                    'test.mp4',
                    '--headless',
                    '--mode',
                    'sequential',
                    '--max-frames',
                    str(args.frames),
                    '--warmup',
                    '2',
                    '--report',
                    str(result_path),
                ]
                peak = swap = 0
                with (args.output / (name + '.log')).open('w') as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=log,
                        env=dict(os.environ, YOLO_ORT_INTRA_OP_THREADS='3'),
                    )
                    started = time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic() - started > 300:
                                raise TimeoutError('native benchmark exceeded 300 seconds')
                            try:
                                lines = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                                for line in lines:
                                    if line.startswith('VmHWM:'):
                                        peak = max(peak, int(line.split()[1]))
                                    elif line.startswith('VmSwap:'):
                                        swap = max(swap, int(line.split()[1]))
                            except FileNotFoundError:
                                pass
                            time.sleep(0.02)
                    finally:
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                            try:
                                process.wait(10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    if process.returncode:
                        raise RuntimeError(f'{name} exited {process.returncode}')
                result = json.loads(result_path.read_text())
                if not all(
                    result[key] == args.frames
                    for key in ('read_frames', 'inferred_frames', 'output_frames')
                ):
                    raise RuntimeError('frame conservation failed')
                report['runs'].append(
                    {
                        'case': label,
                        'repeat': repeat,
                        'binary_sha256': digest(binary),
                        'model_sha256': digest(model),
                        'peak_rss_kib_sampled': peak,
                        'peak_swap_kib_sampled': swap,
                        'process_wall_seconds': time.monotonic() - started,
                        'pipeline_report': result,
                    }
                )
                print(name, 'passed', flush=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
