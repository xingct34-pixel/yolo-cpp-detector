# YOLO 检测部署 | bench/cpu_compare.py
# 功能：在同一段视频上交替运行顺序模式与流水线模式，读取每次 yolo_detector 输出的报告。
# 同时核对读取、推理、输出的帧数，避免因少处理帧而“变快”，被误算成吞吐提升。
# 目的：回答“同一段视频、相同处理帧数下，阶段重叠能否给这台 CPU 带来吞吐收益”。
# 注意：脚本只记录测量结果，不预设流水线一定更快。


# 模块：CPU 基线比较
# 功能：在固定输入、固定线程数下，组织顺序与流水线两种模式的重复测量
"""Alternating sequential/pipeline measurements; no claim of guaranteed speedup."""
from pathlib import Path
import hashlib, json, subprocess, time, os

# 项目根目录：本文件位于 bench/ 下，上一级即项目根
ROOT = Path(__file__).resolve().parents[1]
# 所有报告、日志、汇总结果统一放在 results/，目录已存在时不报错
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
binary = ROOT / 'build-local/yolo_detector'
# ONNX Runtime 算子内线程数固定为 3，与下方记录的 threads 保持一致，避免线程数差异干扰对比
environment = dict(os.environ, YOLO_ORT_INTRA_OP_THREADS='3')
# 汇总记录：先写入测量条件，便于事后复现和核对
# binary_sha256 用来确认所有结果来自同一个可执行文件
record = {
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'frames_per_run': 30,   # 每轮处理帧数，必须与命令行 --max-frames 一致
    'warmup': 2,            # 预热帧数，必须与命令行 --warmup 一致
    'queue_capacity': 2,    # 流水线阶段间队列容量，必须与 --queue-size 一致
    'threads': 3,           # 必须与上方环境变量中的线程数一致
    'runs': [],             # 逐次运行的报告会追加到这里
}


# 模块：交替顺序测量
# 重复 3 轮，每轮内两种模式各跑一次，并轮换谁先跑，
# 以减少固定顺序对缓存、CPU 温度和频率等因素的偏向。
for repeat in range(3):


    # 本轮的执行顺序：偶数轮先顺序后流水线，奇数轮相反
    modes = ['sequential', 'pipeline'] if repeat % 2 == 0 else ['pipeline', 'sequential']
    for mode in modes:
        # label 同时用作报告和日志的文件名，例如 compare-0-sequential
        label = f'compare-{repeat}-{mode}'
        path = OUT / (label + '.json')
        # 本次运行内存驻留高水位的采样最大值，单位 KiB
        peak = 0
        command = [
            str(binary),
            '--video',
            'test.mp4',
            '--headless',           # 不弹出显示窗口，避免渲染影响耗时
            '--mode',
            mode,
            '--queue-policy',
            'block',                # 队列满时阻塞而不是丢帧，保证处理帧数一致
            '--queue-size',
            '2',
            '--max-frames',
            '30',
            '--warmup',
            '2',
            '--report',
            str(path),              # 程序把本次测量写成 JSON 报告
        ]
        # 程序的标准输出和错误输出都写入同名日志，便于出错时排查
        with (OUT / (label + '.log')).open('w') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, env=environment)
            begin = time.monotonic()
            # 轮询进程状态，同时做超时保护和内存采样
            while p.poll() is None:
                # 单次运行超过 120 秒视为卡死，强制结束并让整个脚本失败
                if time.monotonic() - begin > 120:
                    p.kill()
                    p.wait()
                    raise RuntimeError('comparison timeout')
                try:
                    for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
                        # VmHWM 是该进程历史驻留内存的高水位，单位 KiB；
                        # 它同时包含模型、运行库和其余进程内数据，不只是推理缓冲区。
                        if line.startswith('VmHWM:'):
                            peak = max(peak, int(line.split()[1]))
                except FileNotFoundError:
                    # 进程恰好在两次检查之间退出，/proc 条目已消失，忽略即可
                    pass
                # 采样间隔 50 毫秒，兼顾内存峰值的捕捉和脚本自身开销
                time.sleep(0.05)
            # 程序必须正常退出，否则报告不可信
            assert p.returncode == 0, (label, p.returncode)
        row = json.loads(path.read_text())
        # 先核对两种模式做了相同的工作量，再比较 FPS。
        # 读取、推理、输出三个帧数都必须等于 30，漏处理帧会让耗时看起来更短。
        assert row['output_frames'] == row['inferred_frames'] == row['read_frames'] == 30
        # 附上轮次编号和采样到的内存峰值，一并存入汇总
        row.update(repeat=repeat, peak_rss_kib_sampled=peak)
        record['runs'].append(row)
        print(label, 'fps', round(row['output_fps'], 3), flush=True)
# 把测量条件和全部运行结果写成一份汇总 JSON，保留中文不转义，缩进便于人工查看
(OUT / 'controlled-comparison.json').write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + '\n'
)
# 再完整跑一遍本地完整视频（150 帧），确认程序不仅能处理前 30 帧，而是能跑完整段。
# 使用同样的环境变量，运行时间上限 180 秒，超过会抛出异常。
subprocess.run(
    ['bash', 'scripts/demo.sh'],
    cwd=ROOT,
    check=True,
    stdout=(OUT / 'demo.log').open('w'),
    stderr=subprocess.STDOUT,
    timeout=180,
    env=environment,
)
demo = json.loads((OUT / 'demo.json').read_text())
# 完整片段同样要求读取、推理、输出帧数全部为 150，没有漏帧
assert demo['read_frames'] == demo['inferred_frames'] == demo['output_frames'] == 150
print('full 150-frame clip passed', flush=True)
