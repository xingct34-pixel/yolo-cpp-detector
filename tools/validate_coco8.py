#!/usr/bin/env python3
# YOLO 检测部署 | tools/validate_coco8.py
# 一次小样本回归的入口。它把已有数据集的标签换算成原图坐标，分别调用
# yolo_coco_predict 导出两个模型的预测，再用项目评估器和官方 COCOeval 交叉核对。
# 先看 main 里的文件流向，再看 convert 如何把归一化框换成像素。


"""Small labeled CPU regression using externally prepared official COCO8 data, never a full COCO result."""



import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 把项目根目录加入模块搜索路径，这样才能导入上一级目录里的 tools 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# prepare：图片预处理（与量化时一致）；sha256：文件哈希；evaluate：项目自己写的 AP 评估函数
from tools.cpu_quantize import prepare, sha256
from tools.evaluate_detection import evaluate


# 模块：格式转换（convert）
# 作用：把外部准备好的 COCO8 验证集（4 张图片 + YOLO 格式标注）转成 COCO 标注格式，
#       供后面的 AP 评估使用。只读本地文件，不下载、不伪造真值。
# 通俗讲：YOLO 和 COCO 是两种"标注记账方式"，评估工具只认 COCO 的，所以要翻译一遍。


def convert(dataset, names):
    # images：图片信息列表；annotations：标注框列表；hashes：每张图预处理后的指纹
    images, annotations, hashes = [], [], {}
    # sorted 保证图片顺序固定，每次运行得到的编号一致；编号从 1 开始
    for index, path in enumerate(sorted((dataset / 'images/val').glob('*.jpg')), 1):
        image = cv2.imread(str(path))
        # 读不出来（文件损坏等）就直接报错，不能悄悄跳过
        if image is None:
            raise ValueError('cannot decode validation image')
        height, width = image.shape[:2]
        images.append({'id': index, 'file_name': path.name, 'width': width, 'height': height})
        # 计算"预处理后张量"的哈希。后面用它和校准数据的哈希对比，
        # 目的是确认评估用的图和量化校准用的图不是同一批内容（避免自己考自己）
        hashes[path.name] = hashlib.sha256(prepare(image).tobytes()).hexdigest()
        # 读取同名的 .txt 标注文件，一行一个目标
        for line in (
            (dataset / 'labels/val' / path.with_suffix('.txt').name).read_text().splitlines()
        ):
            # YOLO 每行格式：类别 中心x 中心y 宽 高，后四个都是 0~1 的相对比例
            category, x, y, w, h = map(float, line.split())
            # 合法性检查：类别必须是整数且在类别表范围内，四个坐标必须在 0~1 之间
            if (
                not category.is_integer()
                or not 0 <= category < len(names)
                or not all(0 <= z <= 1 for z in (x, y, w, h))
            ):
                raise ValueError('invalid external YOLO label')
            # 转换坐标：YOLO 是"中心点 + 宽高（相对比例）"，
            # COCO 是"左上角 x、y + 宽高（像素）"。
            # 先用中心点减去半个宽/高得到左上角，再乘以图片宽高换成像素
            box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
            annotations.append(
                {
                    'id': len(annotations) + 1,  # 标注框全局编号，从 1 开始
                    'image_id': index,  # 这个框属于哪张图
                    'category_id': int(category) + 1,  # YOLO 类别从 0 起，COCO 这里从 1 起，所以加 1
                    'bbox': box,
                    'area': box[2] * box[3],  # 面积 = 宽 × 高，COCO 评估时用来区分大中小目标
                    'iscrowd': 0,  # 0 表示不是"人群"类密集标注，当作普通单个目标
                }
            )
    # 数据集规模校验：必须正好 4 张图，防止拿错数据集
    if len(images) != 4:
        raise ValueError('expected exactly four COCO8 val images')
    return {
        'info': {
            # 说明数据来源：这 4 张图出自 COCO train2017，不是独立的测试集
            'description': 'COCO8 val smoke subset from COCO train2017; not held-out full COCO'
        },
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(names)],
    }, hashes


# 模块：官方复核（official_ap）
# 作用：用固定版本的官方 pycocotools 对同一份预测结果计算 AP，
#       作为"标准答案"，用来检验项目自己写的评估函数算得对不对。
# 通俗讲：自己写的评分程序可能有 bug，所以再找官方评分程序算一遍，两边分数一致才放心。


def official_ap(dataset, predictions):
    # 没有任何预测框时官方工具会报错，这种情况直接定义为 AP 全为 0
    if not predictions:
        return {'AP50': 0.0, 'AP75': 0.0, 'mAP50_95': 0.0}
    # pycocotools 会往终端打印大量过程信息，这里把标准输出丢到内存里，保持终端干净
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        # 直接把内存里的字典当作真值数据集，不需要写文件再读
        coco.dataset = dataset
        coco.createIndex()  # 建立索引，之后才能按图片、类别查询
        result = coco.loadRes(predictions)  # 加载预测结果
        evaluator = COCOeval(coco, result, 'bbox')  # 'bbox' 表示按检测框评估
        evaluator.evaluate()  # 逐图逐类别匹配预测框与真值框
        evaluator.accumulate()  # 汇总出精度-召回曲线
        evaluator.summarize()  # 计算最终指标，结果存在 evaluator.stats 里
    # stats[0]：IoU 从 0.5 到 0.95 的平均 AP（最常引用的 mAP）
    # stats[1]：IoU=0.5 时的 AP（要求宽松）
    # stats[2]：IoU=0.75 时的 AP（要求严格）
    return {
        'mAP50_95': float(evaluator.stats[0]),
        'AP50': float(evaluator.stats[1]),
        'AP75': float(evaluator.stats[2]),
    }


# 模块：完整回归流程（main）
# 作用：串起整个测试：转换标注 → 校验数据和模型身份 → 用 C++ 程序分别跑 FP32 和 INT8 模型出预测 →
#       计算 AP 并和官方结果交叉验证 → 汇总 INT8 相对 FP32 的精度变化，写入 report.json。
# 通俗讲：这是"量化之后检测效果到底掉了多少"的小型验收考试。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)  # COCO8 数据集目录
    parser.add_argument('--fp32', type=Path, required=True)  # 原始 FP32 模型
    parser.add_argument('--int8', type=Path, required=True)  # 量化后的 INT8 模型
    parser.add_argument('--calibration-report', type=Path, required=True)  # 量化脚本生成的 report.json
    parser.add_argument('--binary', type=Path, default=Path('build-local/yolo_coco_predict'))  # C++ 预测程序
    parser.add_argument('--output', type=Path, required=True)  # 本次实验的输出目录
    args = parser.parse_args()
    # 输出目录必须是新的，防止覆盖之前的实验记录
    if args.output.exists():
        parser.error('choose a new output directory to retain prior evidence')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    # 读取 80 个类别名称，一行一个
    names = (root / 'data/coco.txt').read_text().splitlines()
    dataset, hashes = convert(args.dataset, names)
    # 把转换好的 COCO 标注存盘，C++ 程序要用它
    annotation = args.output / 'annotations.json'
    annotation.write_text(json.dumps(dataset))
    calibration = json.loads(args.calibration_report.read_text())
    # 求交集：评估图的指纹和校准图的指纹有没有重复。
    # 重复就意味着量化时"见过"这些图，评估结果会偏乐观
    overlap = set(hashes.values()) & set(calibration['calibration_tensor_sha256'])
    # 先把报告的基本信息填好，各模型的结果后面逐个补充
    report = {
        'dataset': 'Ultralytics COCO8 val, four images sourced from COCO train2017',
        # 明确声明结论的适用范围：只是小规模流程回归，且模型预训练时可能见过这些图，不能当作总体精度
        'scope': 'small labeled pipeline regression; not independent of model pretraining; no population accuracy claim',
        'pycocotools': importlib.metadata.version('pycocotools'),  # 记录版本，方便复现
        'validation_images': 4,
        'ground_truth_boxes': len(dataset['annotations']),
        'calibration_tensor_overlap': len(overlap),
        'validation_tensor_sha256': hashes,
        'annotation_sha256': sha256(annotation),
        'models': {},
    }
    # 三道"身份与独立性"关卡，任何一道不过就停止：
    # 1. 校准数据与评估数据有重复 → 不独立，终止
    if overlap:
        raise RuntimeError('calibration and evaluation tensors overlap')
    # 2. 传入的 FP32 模型哈希必须与量化时记录的一致，确认没拿错模型文件
    if sha256(args.fp32) != calibration['source_model_sha256']:
        raise RuntimeError('FP32 identity differs from supplied calibration evidence')
    # 3. 传入的 INT8 模型哈希同理
    if sha256(args.int8) != calibration['quantized_sha256']:
        raise RuntimeError('INT8 identity differs from supplied calibration evidence')
    # 两个模型走完全相同的流程，保证对比公平
    for label, model in [('fp32', args.fp32), ('int8', args.int8)]:
        destination = args.output / (label + '-predictions.json')
        # 调用 C++ 原生程序推理，stdout 和 stderr 都写进日志文件留证。
        # 用原生程序而不是 Python 推理，是为了测到真正部署时的整条链路
        with (args.output / (label + '-native.log')).open('w') as log:
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    str(annotation.resolve()),
                    str((args.dataset / 'images/val').resolve()),
                    str(model.resolve()),
                    str(root / 'data/coco.txt'),
                    str(destination.resolve()),
                ],
                stdout=log,
                stderr=log,
                check=True,  # 程序返回非零就抛异常，不会带着失败结果继续往下走
            )
        predictions = json.loads(destination.read_text())['predictions']
        # 项目自己写的评估函数算一遍
        local = evaluate(dataset, predictions)
        # 官方工具再算一遍
        reference = official_ap(dataset, predictions)
        # 逐项求两者差的绝对值
        difference = {name: abs(local[name] - reference[name]) for name in reference}
        # 差异超过百万分之一就认为自研评估有问题，终止，避免用错误的评分程序下结论
        if max(difference.values()) > 1e-6:
            raise RuntimeError('non-crowd evaluator differs from official COCOeval')
        report['models'][label] = {
            'model_sha256': sha256(model),
            'predictions': len(predictions),  # 预测框总数
            'reference_ap': reference,  # 官方 AP，作为最终采信的结果
            'custom_ap': {k: local[k] for k in reference},  # 自研 AP，仅用于对照
            'reference_difference': difference,
        }
    # 计算 INT8 减 FP32 的 AP 变化量：
    # 负数表示量化后精度下降，接近 0 表示几乎无损，正数则可能只是小样本波动
    report['int8_minus_fp32'] = {
        key: report['models']['int8']['reference_ap'][key]
        - report['models']['fp32']['reference_ap'][key]
        for key in ('AP50', 'AP75', 'mAP50_95')
    }
    # ensure_ascii=False 让中文等非 ASCII 字符原样保存，不转成 \uXXXX
    (args.output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


# 模块：脚本入口
# 作用：只有直接运行本文件时才执行 main()；被其他文件 import 时不会自动运行。


if __name__ == '__main__':
    main()
