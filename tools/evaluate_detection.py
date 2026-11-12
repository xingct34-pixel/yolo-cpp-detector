#!/usr/bin/env python3
"""Dependency-free bbox AP evaluator for non-crowd COCO-format annotations."""
import argparse
import hashlib
import json
import math
from pathlib import Path


# 模块：输入校验：拒绝无效框、未知图片/类别及暂不支持的crowd/ignore标注


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


# 模块：框交并比：输入为原图xywh坐标，两个不重叠的框交集为0


def iou(a, b):
    intersection = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    )
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


# 模块：单类匹配：每张图最多100个预测，同一真值只能被当前类别匹配一次


def class_ap(ground_truth, predictions, threshold):
    truths = {}
    for row in ground_truth:
        truths.setdefault(row['image_id'], []).append(row['bbox'])
    limited = []
    for image in {p['image_id'] for p in predictions}:
        rows = sorted(
            (p for p in predictions if p['image_id'] == image),
            key=lambda p: p['score'],
            reverse=True,
        )
        limited.extend(rows[:100])
    # 相同分数按图片ID稳定排序，使输出不依赖set遍历顺序。
    limited.sort(key=lambda p: (-p['score'], p['image_id']))
    matched, tp, fp, points = set(), 0, 0, []
    for row in limited:
        best, best_iou = None, threshold
        for index, box in enumerate(truths.get(row['image_id'], [])):
            key = row['image_id'], index
            overlap = iou(row['bbox'], box)
            if key not in matched and overlap >= best_iou:
                best, best_iou = key, overlap
        if best is not None:
            matched.add(best)
            tp += 1
        else:
            fp += 1
        points.append((tp / len(ground_truth), tp / (tp + fp)))
    # 101点插值：每个召回位置取其右侧最高精确率，再求平均。
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
        values.append(precision_envelope[cursor] if cursor < len(points) else 0.0)
    return sum(values) / 101


# 模块：评估汇总：有真值的类别参与平均；明确不是完整COCO官方评估实现


def evaluate(dataset, predictions):
    validate(dataset, predictions)
    categories = sorted({a['category_id'] for a in dataset['annotations']})
    if not categories:
        raise ValueError('at least one ground-truth box is required')
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


# 模块：命令入口：保存输入身份，允许在同一标注集上对比FP32和INT8结果


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
