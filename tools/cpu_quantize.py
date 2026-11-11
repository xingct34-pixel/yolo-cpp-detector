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
    inputs, outputs = engine.get_inputs(), engine.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('one input and output required')
    if inputs[0].type != 'tensor(float)' or inputs[0].shape != [1, 3, 640, 640]:
        raise ValueError('expected FP32 input [1,3,640,640]')
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
        quantize_static(
            str(prepared),
            str(quantized),
            Reader(name, calibration),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            op_types_to_quantize=['Conv', 'MatMul'],
        )
        onnx.checker.check_model(str(quantized))
        model = onnx.load(str(quantized))
        counts = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        if not counts.get('QuantizeLinear') or not counts.get('DequantizeLinear'):
            raise RuntimeError('no QDQ nodes created')
        reduced = session(quantized, args.threads)
        check_contract(reduced)
        report['quantized_sha256'] = sha256(quantized)
        report['model_bytes'] = {
            'fp32': args.model.stat().st_size,
            'int8': quantized.stat().st_size,
        }
        report['quantized_node_counts'] = counts
        report['timing_scope'] = (
            'Python ORT Session.run, no preprocessing or queue wait; 2 warmups per model'
        )
        engines = {'fp32': original, 'int8': reduced}
        timings = {key: [] for key in engines}
        for engine in engines.values():
            for _ in range(2):
                engine.run(None, {name: evaluation[0]})
        errors = []
        for repeat in range(args.repeats):
            order = ['fp32', 'int8'] if repeat % 2 == 0 else ['int8', 'fp32']
            for tensor in evaluation:
                outputs = {}
                for key in order:
                    begin = time.perf_counter()
                    value = engines[key].run(None, {name: tensor})[0]
                    timings[key].append((time.perf_counter() - begin) * 1000)
                    if value.shape != (1, 84, 8400) or not np.isfinite(value).all():
                        raise RuntimeError('invalid inference output')
                    outputs[key] = value
                if repeat == 0:
                    difference = outputs['int8'].astype(np.float64) - outputs['fp32']
                    errors.append(
                        {
                            'raw_mae': float(np.abs(difference).mean()),
                            'raw_rmse': float(np.sqrt(np.square(difference).mean())),
                            'score_mae': float(np.abs(difference[:, 4:, :]).mean()),
                        }
                    )
        report['latency_ms'] = {
            key: {'samples': values, 'median': float(np.median(values))}
            for key, values in timings.items()
        }
        report['numerical_difference'] = errors
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps({key: report[key] for key in ('passed', 'model_bytes', 'latency_ms')}, indent=2)
    )


if __name__ == '__main__':
    main()
