# YOLO 检测部署 | tools/cpu_quantize.py
# 离线比较 YOLO 的 FP32 和静态 INT8 模型。前一组视频帧用于校准，后一组用于比较。
# 同一输入分别跑两个模型，报告文件大小、推理时间和原始张量偏差。
# 要了解框有没有检对，接着看 validate_coco8.py 的带标注评估。


# 这是一个YOLO 目标检测模型（ONNX 格式）的 INT8 量化实验工具。
# 它把 FP32 模型转成 INT8 模型，并自动对比两者的体积、速度和输出差异，
# 最后生成一份报告。

# 整体流程

# 步骤 1｜读视频取帧：从视频里连续读出 16 帧（默认 8 帧校准 + 8 帧评估），两组帧编号互不重叠。

# 步骤 2｜预处理：每帧按 YOLO 常用方式处理，即等比缩放到 640×640、不足处用灰色（114）
# 居中填充、BGR 转 RGB、像素除以 255、排成 NCHW 格式。
# 注释里说明这与 C++ 端的 prepare_yolo 保持一致，
# 保证 Python 测的和实际部署用的是同一种输入。

# 步骤 3｜检查原模型接口：必须是单输入单输出，输入 [1,3,640,640]，
# 输出 [1,84,8400]（本项目所用 YOLO11n 导出模型的接口）。

# 步骤 4｜量化：用前 8 帧做校准，生成 QDQ 格式的静态 INT8 模型。

# 步骤 5｜验证量化是否生效：检查模型合法、有 QDQ 节点、接口没变。

# 步骤 6｜对比测试：用后 8 帧，让 FP32 和 INT8 各跑若干轮：

# 速度：记录每次推理耗时，取中位数
# 数值偏差：计算两者输出的差异（MAE、RMSE、类别分数 MAE）
# 健壮性：输出形状错误或出现 NaN/Inf 就报错

# 步骤 7｜写报告：进入实验主体后会尝试写出 report.json；参数解析或建目录失败会更早退出。

# 几个值得注意的设计
# 只在 CPU 上测，线程数默认 3，上限 6。
# 输出目录必须是新的，已存在就拒绝，防止覆盖之前的实验证据。
# 可追溯：报告里记录了模型、视频、每帧张量的 SHA-256，以及各库版本，方便以后复现。
# 会检查完全重复的输入：如果校准帧和评估帧里有内容完全相同的张量（比如视频里有静止画面），会在报告里发出警告。
# 诚实标注局限：报告里明确写了 accuracy_map: None，
# 说明这是用无标注的相邻视频帧做的数值冒烟测试，不是真正的精度（mAP）评估。
# 它只能说明"INT8 和 FP32 的输出差得多不多"，
# 不能说明检测效果掉了多少。
# 计时口径说明：只计 Session.run，不含预处理和排队，避免被误解为端到端延迟。


# 模块：CPU量化工具：校准、静态INT8转换及正确性/性能对比


"""CPU INT8 QDQ experiment using disjoint video frames. No implicit downloads."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


# 模块：文件校验：分块计算SHA-256


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# 模块：图像预处理：生成与C++输入约定一致的校准数据


def prepare(image):
    """Match prepare_yolo: round positive dimensions, center padding, RGB NCHW."""
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('expected nonempty BGR uint8 image')
    height, width = image.shape[:2]
    if not width or not height:
        raise ValueError('empty image')
    scale = min(640 / width, 640 / height)
    resized_width = max(1, min(640, math.floor(width * scale + 0.5)))
    resized_height = max(1, min(640, math.floor(height * scale + 0.5)))
    left = (640 - resized_width) // 2
    top = (640 - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        640 - resized_height - top,
        left,
        640 - resized_width - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return cv2.dnn.blobFromImage(padded, 1 / 255, swapRB=True, crop=False, ddepth=cv2.CV_32F)


# 模块：校准数据接口：按批提供已经准备好的输入


class Reader(CalibrationDataReader):
    def __init__(self, name, tensors):
        self.name = name
        self.tensors = tensors
        self.rewind()

    def get_next(self):
        tensor = next(self.iterator, None)
        return None if tensor is None else {self.name: tensor}

    def rewind(self):
        self.iterator = iter(self.tensors)


# 模块：推理会话：建立CPU执行会话


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


# 模块：模型约定检查：验证输入输出类型和形状


def check_contract(engine):
    # 获取模型的全部输入和输出节点信息
    inputs, outputs = engine.get_inputs(), engine.get_outputs()

    # 要求模型有且仅有一个输入和一个输出
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('one input and output required')

    # 校验输入：类型必须为 FP32，形状必须为 [1, 3, 640, 640]（批次、通道、高、宽）
    if inputs[0].type != 'tensor(float)' or inputs[0].shape != [1, 3, 640, 640]:
        raise ValueError('expected FP32 input [1,3,640,640]')

    # 校验输出：类型必须为 FP32，形状必须为 [1, 84, 8400]（84 = 4 个框坐标 + 80 个类别分数，8400 个候选框）
    if outputs[0].type != 'tensor(float)' or outputs[0].shape != [1, 84, 8400]:
        raise ValueError('expected FP32 output [1,84,8400]')


# 模块：执行入口：读取参数并组织该工具的完整流程


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new experiment directory')
    parser.add_argument('--calibration-frames', type=int, default=8)
    parser.add_argument('--evaluation-frames', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if not (
        1 <= args.calibration_frames <= 128
        and 1 <= args.evaluation_frames <= 128
        and 1 <= args.repeats <= 20
        and 1 <= args.threads <= 6
    ):
        parser.error('frame/repeat/thread limits exceeded')
    if args.output.exists():
        parser.error('output already exists; choose a new directory to preserve prior evidence')
    args.output.mkdir(parents=True)
    report = {
        'passed': False,
        'backend': 'CPU',
        'accuracy_map': None,
        'accuracy_note': 'Unlabelled adjacent video frames; numerical smoke test only.',
        'versions': {
            name: importlib.metadata.version(name)
            for name in ('numpy', 'onnx', 'onnxruntime', 'opencv-python-headless')
        },
        'threads': args.threads,
        'repeats': args.repeats,
    }
    try:
        tensors = []
        capture = cv2.VideoCapture(str(args.video))
        try:
            for _ in range(args.calibration_frames + args.evaluation_frames):
                ok, image = capture.read()
                if not ok:
                    raise ValueError('video has insufficient frames')
                tensors.append(prepare(image))
        finally:
            capture.release()
        calibration = tensors[: args.calibration_frames]
        evaluation = tensors[args.calibration_frames :]
        report['calibration_frame_ids'] = list(range(args.calibration_frames))
        report['evaluation_frame_ids'] = list(range(args.calibration_frames, len(tensors)))
        report['source_model_sha256'] = sha256(args.model)
        report['video_sha256'] = sha256(args.video)
        report['calibration_tensor_sha256'] = [
            hashlib.sha256(t.tobytes()).hexdigest() for t in calibration
        ]
        report['evaluation_tensor_sha256'] = [
            hashlib.sha256(t.tobytes()).hexdigest() for t in evaluation
        ]
        calibration_hashes = set(report['calibration_tensor_sha256'])
        evaluation_hashes = set(report['evaluation_tensor_sha256'])
        report['unique_calibration_tensors'] = len(calibration_hashes)
        report['unique_evaluation_tensors'] = len(evaluation_hashes)
        report['calibration_evaluation_content_overlap'] = len(
            calibration_hashes & evaluation_hashes
        )
        if report['calibration_evaluation_content_overlap']:
            report['data_warning'] = (
                'Distinct frame indices contain duplicate tensors; not an independent accuracy set.'
            )
        original = session(args.model, args.threads)
        check_contract(original)
        name = original.get_inputs()[0].name
        quantized = args.output / 'model.int8.onnx'
        prepared = args.output / 'model.prepared.onnx'
        quant_pre_process(str(args.model), str(prepared))
        
        # 静态量化：把 FP32 模型转成 INT8 模型，目的是让模型更小、推理更快。
        # 通俗讲：FP32 像用很精细的刻度尺（小数点后很多位）量东西，INT8 是换成只有 256 格的粗刻度尺。
        # "静态"的意思是：量化前先拿一批样例数据（calibration）跑一遍，
        # 观察每一层数值大概在什么范围，再据此定好刻度，推理时不再调整。
        quantize_static(
            str(prepared),            # 输入：预处理好的 FP32 模型
            str(quantized),           # 输出：量化后模型的保存路径
            Reader(name, calibration),  # 校准数据读取器，负责一批一批喂样例数据，用来统计数值范围
            # QDQ 格式：在原有算子前后插入 QuantizeLinear（浮点转整数）和 DequantizeLinear（整数转回浮点）节点。
            # 好处是模型结构清晰、兼容性好，由运行时决定哪些部分真正用整数算。
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,  # 激活值（层与层之间传递的数据）用有符号 8 位整数
            weight_type=QuantType.QInt8,      # 权重（模型学到的参数）用有符号 8 位整数
            # 按通道量化：每个卷积输出通道各用一把自己的刻度尺，而不是整层共用一把。
            # 通俗讲：不同通道数值大小差别很大，各配各的尺子，有助于减小数值偏差，实际效果由测量判断。
            per_channel=True,
            # 把 8 位范围缩成 7 位使用。主要是为了避免部分 CPU 指令在整数相乘累加时溢出，
            # 代价是精度略降。如果目标 CPU 支持 VNNI 等新指令，可以考虑关掉。
            reduce_range=True,
            # 只量化卷积和矩阵乘法，其余类型未列入量化目标；运行速度和精度需要分别测量
            op_types_to_quantize=['Conv', 'MatMul'],
        )
        # 检查量化后的模型是否是合法的 ONNX 模型（结构、类型、版本等）
        onnx.checker.check_model(str(quantized))
        model = onnx.load(str(quantized))
        # 统计计算图里每种算子出现了多少次，例如 {'Conv': 60, 'QuantizeLinear': 120, ...}
        counts = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        # 安全检查：QDQ 格式的量化一定会生成这两种节点。
        # 如果一个都没有，说明量化被悄悄跳过了，继续测下去没有意义，直接报错
        if not counts.get('QuantizeLinear') or not counts.get('DequantizeLinear'):
            raise RuntimeError('no QDQ nodes created')
        # 用量化后的模型创建推理会话，并复用前面的输入输出约定检查，
        # 确保量化没有改变模型对外的接口（仍是 [1,3,640,640] 输入、[1,84,8400] 输出）
        reduced = session(quantized, args.threads)
        check_contract(reduced)
        # 把量化模型的哈希值写入报告，方便以后确认测的是哪个文件
        report['quantized_sha256'] = sha256(quantized)
        # 记录两个模型的文件大小（字节），用来算压缩比
        report['model_bytes'] = {
            'fp32': args.model.stat().st_size,
            'int8': quantized.stat().st_size,
        }
        # 记录各算子数量，事后可核对量化覆盖了哪些部分
        report['quantized_node_counts'] = counts
        # 在报告里写明计时口径，避免别人误以为这是端到端耗时：
        # 只计 Session.run，不含图片预处理、不含排队等待；每个模型预热 2 次
        report['timing_scope'] = (
            'Python ORT Session.run, no preprocessing or queue wait; 2 warmups per model'
        )
        # 把两个模型放进字典，方便后面统一循环处理
        engines = {'fp32': original, 'int8': reduced}
        # 为每个模型准备一个空列表，用来存每次推理的耗时
        timings = {key: [] for key in engines}
        # 预热：每个模型先空跑 2 次，结果不计入统计。
        # 通俗讲：第一次运行时要分配内存、初始化线程、加载缓存，会特别慢，
        # 不预热的话这部分"开机时间"会污染测速结果
        for engine in engines.values():
            for _ in range(2):
                engine.run(None, {name: evaluation[0]})
        errors = []
        # 整个评估集重复跑 args.repeats 轮，样本多了中位数才更稳
        for repeat in range(args.repeats):
            # 奇偶轮交替两个模型的先后顺序。
            # 通俗讲：先跑的那个可能占了"热缓存"或"CPU 睿频"的便宜，轮流先跑才公平
            order = ['fp32', 'int8'] if repeat % 2 == 0 else ['int8', 'fp32']
            for tensor in evaluation:
                outputs = {}
                for key in order:
                    # perf_counter 是高精度计时器，乘 1000 把秒换成毫秒
                    begin = time.perf_counter()
                    value = engines[key].run(None, {name: tensor})[0]
                    timings[key].append((time.perf_counter() - begin) * 1000)
                    # 合法性检查：形状必须正确，且不能有 NaN 或 Inf。
                    # 量化出问题时常见的症状就是输出变成无穷大或非数字
                    if value.shape != (1, 84, 8400) or not np.isfinite(value).all():
                        raise RuntimeError('invalid inference output')
                    outputs[key] = value
                # 数值差异只在第一轮算一次即可，因为同一输入下每轮结果是一样的，重复算没有意义
                if repeat == 0:
                    # 转成 float64 再相减，避免用 float32 做减法时自身引入误差
                    difference = outputs['int8'].astype(np.float64) - outputs['fp32']
                    errors.append(
                        {
                            # MAE（平均绝对误差）：INT8 与 FP32 输出平均差了多少，直观好懂
                            'raw_mae': float(np.abs(difference).mean()),
                            # RMSE（均方根误差）：和 MAE 类似，但对个别大偏差更敏感。
                            # 如果 RMSE 远大于 MAE，说明有少数位置误差特别大
                            'raw_rmse': float(np.sqrt(np.square(difference).mean())),
                            # 输出 84 行 = 前 4 行是框的位置，后 80 行是 80 个类别的分数。
                            # 这里只看第 4 行之后，即类别分数的误差，
                            # 因为分数直接决定"识别成什么、置信度多高"，对检测结果影响最大
                            'score_mae': float(np.abs(difference[:, 4:, :]).mean()),
                        }
                    )
        # 汇总延迟：保存每个模型的全部原始样本，并给出中位数。
        # 用中位数而不是平均值，是因为偶尔的系统卡顿会产生极端大值，把平均值拉偏，中位数不受影响
        report['latency_ms'] = {
            key: {'samples': values, 'median': float(np.median(values))}
            for key, values in timings.items()
        }
        report['numerical_difference'] = errors
        # 代码能走到这一行，说明上面所有检查都通过了
        report['passed'] = True
    except Exception as error:
        # 出错时把"错误类型: 错误信息"写进报告，再用 raise 原样抛出，
        # 这样既留了记录，程序也会如实报错，不会假装成功
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        # finally 无论成功还是报错都会执行，尝试将实验主体的执行结果写入 report.json，
        # 失败时报告里就有 error 字段，方便排查
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    # 在终端打印精简摘要：是否通过、模型大小对比、延迟对比，完整细节看 report.json
    print(
        json.dumps({key: report[key] for key in ('passed', 'model_bytes', 'latency_ms')}, indent=2)
    )


if __name__ == '__main__':
    main()
