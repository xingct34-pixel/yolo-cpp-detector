#!/usr/bin/env python3
# YOLO 检测部署 | tools/evaluate_detection.py
# 读取标注框和预测框，为每个类别按置信度从高到低匹配，累计精确率与召回率。
# IoU 门槛决定两个框能否配对，在多个门槛上计算 AP 后汇总成报告。
# 该实现面向普通 bbox 标注；validate_coco8.py 另用官方工具复核同一批结果。
#
# 阅读地图（按源码顺序）
#   1 输入校验 → 2 框交并比 → 3 单类匹配
#   4 匹配同类目标 → 5 求单类 AP → 6 评估汇总
#   7 多阈值汇总 → 8 命令入口 → 9 核对输入身份


"""Dependency-free bbox AP evaluator for non-crowd COCO-format annotations."""


import argparse  # 读取命令行输入并检查参数格式
import hashlib  # 按文件或字节内容计算指纹
import json  # 在 Python 对象和 JSON 文本之间转换
import math  # 检查有限数值或做数值运算
from pathlib import Path  # 拼接目录、文件名并读取文件


# 模块 1：输入校验
# 功能：拒绝无效框、未知图片/类别及暂不支持的crowd/ignore标注


# 先验证图片、类别和框能相互引用。无效引用如果直接进入匹配，会被误当成模型本身的漏检。
def validate(dataset, predictions):
    images = {row['id'] for row in dataset['images']}
    categories = {row['id'] for row in dataset['categories']}
    if len(images) != len(dataset['images']) or len(categories) != len(dataset['categories']):
        raise ValueError('duplicate image/category ids')
    for kind, rows in [('annotation', dataset['annotations']), ('prediction', predictions)]:
        for row in rows:
            if row['image_id'] not in images or row['category_id'] not in categories:
                raise ValueError(f'{kind}: unknown image/category id')
            box = row['bbox']
            if len(box) != 4 or any(not math.isfinite(x) for x in box) or min(box[2:]) <= 0:
                raise ValueError(f'{kind}: bbox must be finite xywh with positive size')
            if kind == 'annotation' and (row.get('iscrowd', 0) or row.get('ignore', 0)):
                raise ValueError('crowd/ignore annotations require the official COCO evaluator')
            if kind == 'prediction' and (
                not math.isfinite(row['score']) or not 0 <= row['score'] <= 1
            ):
                raise ValueError('score must be finite and in [0,1]')


# 模块 2：框交并比
# 功能：输入为原图xywh坐标，两个不重叠的框交集为0


def iou(a, b):
    # 两个 10×10 框横向错开 2 像素，交集是 8×10，并集是 100+100−80，IoU=80/120。
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    )
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


# 模块 3：单类匹配
# 功能：每张图最多100个预测，同一真值只能被当前类别匹配一次


def class_ap(ground_truth, predictions, threshold):


    # 模块 4：匹配同类目标
    # 这次函数调用只处理一个类别和一个 IoU 阈值。先按图片找到真值，再让高分预测优先匹配。


    truths = {}
    for row in ground_truth:
        # 字典的键是图片 ID，值是这张图的所有同类真值框；setdefault 只在第一次遇见该图时建空列表。
        truths.setdefault(row['image_id'], []).append(row['bbox'])
    limited = []
    for image in {p['image_id'] for p in predictions}:
        rows = sorted(
            (p for p in predictions if p['image_id'] == image),
            key=lambda p: p['score'],
            reverse=True,
        )
        # 每图每类只取分数最高的 100 个。先在全数据集截断会让后面的图片没有参与机会。
        limited.extend(rows[:100])
    # 相同分数按图片ID稳定排序，使输出不依赖set遍历顺序。
    # 按负分数升序相当于高分优先；同分时以图片 ID 排序，保持复核次序一致。
    limited.sort(key=lambda p: (-p['score'], p['image_id']))
    matched, tp, fp, points = set(), 0, 0, []
    for row in limited:
        best, best_iou = None, threshold
        for index, box in enumerate(truths.get(row['image_id'], [])):
            key = row['image_id'], index
            overlap = iou(row['bbox'], box)
            # 同一张图里的真值用 (图片 ID, 框下标) 标识。一条船被报两次，第二个框不能再用第一次占用的真值。
            if key not in matched and overlap >= best_iou:
                best, best_iou = key, overlap
        if best is not None:
            matched.add(best)
            tp += 1
        else:
            fp += 1
        # 一票预测产生一个点。召回率分母是全部真值，精确率分母是到目前为止已经检查的预测。
        points.append((tp / len(ground_truth), tp / (tp + fp)))
    # 101点插值：每个召回位置取其右侧最高精确率，再求平均。


    # 模块 5：求单类 AP
    # 从右向左保留后面能达到的最高精确率，再读取 0、0.01、…、1 这 101 个召回位置，未达到的位置填零。


    precision_envelope = [0.0] * len(points)
    peak = 0.0
    for index in range(len(points) - 1, -1, -1):
        peak = max(peak, points[index][1])
        precision_envelope[index] = peak
    # 与官方linspace网格一致：0.70可能表示0.7000000000000001；不要用epsilon改变查找位置。
    values = []
    cursor = 0
    for step in range(101):
        while cursor < len(points) and points[cursor][0] < step * (1.0 / 100):
            cursor += 1
        # 如果模型最多找回一半目标，召回率 0.51 到 1 的位置就都记 0，不能删掉这些位置后再平均。
        values.append(precision_envelope[cursor] if cursor < len(points) else 0.0)
    return sum(values) / 101


# 模块 6：评估汇总
# 功能：有真值的类别参与平均；明确不是完整COCO官方评估实现


def evaluate(dataset, predictions):
    validate(dataset, predictions)
    # 仅对有真值的类别平均。若这批图里没有船，就没有可用来度量船的召回率的分母。
    categories = sorted({a['category_id'] for a in dataset['annotations']})
    if not categories:
        raise ValueError('at least one ground-truth box is required')


    # 模块 7：多阈值汇总
    # 0.50 到 0.95 每次增加 0.05。因此 AP75 在索引 5，mAP50_95 对 10 个阈值再取平均。


    thresholds = [0.5 + ((0.95 - 0.5) / 9) * i for i in range(10)]
    by_class = {}
    for category in categories:
        gt = [a for a in dataset['annotations'] if a['category_id'] == category]
        pred = [p for p in predictions if p['category_id'] == category]
        by_class[str(category)] = [class_ap(gt, pred, t) for t in thresholds]
    ap = [sum(row[i] for row in by_class.values()) / len(categories) for i in range(10)]
    return {
        'evaluator': 'bbox-101point-noncrowd-v2',
        'AP50': ap[0],
        'AP75': ap[5],
        'mAP50_95': sum(ap) / len(ap),
        'per_class_ap': by_class,
        'images': len(dataset['images']),
        'ground_truth_boxes': len(dataset['annotations']),
        'predictions': len(predictions),
        'scope': 'all areas; max 100 detections per image/category; no crowd/ignore; not official COCOeval',
    }


# 模块 8：命令入口
# 功能：保存输入身份，允许在同一标注集上对比FP32和INT8结果


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    dataset = json.loads(args.annotations.read_text(encoding='utf-8'))
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    result = evaluate(
        dataset,
        predictions.get('predictions', []) if isinstance(predictions, dict) else predictions,
    )


    # 模块 9：核对输入身份
    # 记录的是标注和预测文件的字节摘要。以后分数不同，先判断是不是换了输入文件。


    result['input_sha256'] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [('annotations', args.annotations), ('predictions', args.predictions)]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
