#!/usr/bin/env python3
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cpu_quantize import prepare, sha256
from tools.evaluate_detection import evaluate


# 模块：格式转换：只读取外部准备的4张val图片及YOLO标注，不下载文件、不伪造真值


def convert(dataset, names):
    images, annotations, hashes = [], [], {}
    for index, path in enumerate(sorted((dataset / 'images/val').glob('*.jpg')), 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('cannot decode validation image')
        height, width = image.shape[:2]
        images.append({'id': index, 'file_name': path.name, 'width': width, 'height': height})
        hashes[path.name] = hashlib.sha256(prepare(image).tobytes()).hexdigest()
        for line in (
            (dataset / 'labels/val' / path.with_suffix('.txt').name).read_text().splitlines()
        ):
            category, x, y, w, h = map(float, line.split())
            if (
                not category.is_integer()
                or not 0 <= category < len(names)
                or not all(0 <= z <= 1 for z in (x, y, w, h))
            ):
                raise ValueError('invalid external YOLO label')
            box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
            annotations.append(
                {
                    'id': len(annotations) + 1,
                    'image_id': index,
                    'category_id': int(category) + 1,
                    'bbox': box,
                    'area': box[2] * box[3],
                    'iscrowd': 0,
                }
            )
    if len(images) != 4:
        raise ValueError('expected exactly four COCO8 val images')
    return {
        'info': {
            'description': 'COCO8 val smoke subset from COCO train2017; not held-out full COCO'
        },
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(names)],
    }, hashes


# 模块：官方复核：使用固定版本pycocotools同一预测作all-area/maxDets100评估


def official_ap(dataset, predictions):
    if not predictions:
        return {'AP50': 0.0, 'AP75': 0.0, 'mAP50_95': 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        result = coco.loadRes(predictions)
        evaluator = COCOeval(coco, result, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        'mAP50_95': float(evaluator.stats[0]),
        'AP50': float(evaluator.stats[1]),
        'AP75': float(evaluator.stats[2]),
    }


# 模块：完整回归：原生C++导出、标注AP和校准内容独立性分别留证


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--fp32', type=Path, required=True)
    parser.add_argument('--int8', type=Path, required=True)
    parser.add_argument('--calibration-report', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('build-local/yolo_coco_predict'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output directory to retain prior evidence')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    names = (root / 'data/coco.txt').read_text().splitlines()
    dataset, hashes = convert(args.dataset, names)
    annotation = args.output / 'annotations.json'
    annotation.write_text(json.dumps(dataset))
    calibration = json.loads(args.calibration_report.read_text())
    overlap = set(hashes.values()) & set(calibration['calibration_tensor_sha256'])
    report = {
        'dataset': 'Ultralytics COCO8 val, four images sourced from COCO train2017',
        'scope': 'small labeled pipeline regression; not independent of model pretraining; no population accuracy claim',
        'pycocotools': importlib.metadata.version('pycocotools'),
        'validation_images': 4,
        'ground_truth_boxes': len(dataset['annotations']),
        'calibration_tensor_overlap': len(overlap),
        'validation_tensor_sha256': hashes,
        'annotation_sha256': sha256(annotation),
        'models': {},
    }
    if overlap:
        raise RuntimeError('calibration and evaluation tensors overlap')
    if sha256(args.fp32) != calibration['source_model_sha256']:
        raise RuntimeError('FP32 identity differs from supplied calibration evidence')
    if sha256(args.int8) != calibration['quantized_sha256']:
        raise RuntimeError('INT8 identity differs from supplied calibration evidence')
    for label, model in [('fp32', args.fp32), ('int8', args.int8)]:
        destination = args.output / (label + '-predictions.json')
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
                check=True,
            )
        predictions = json.loads(destination.read_text())['predictions']
        local = evaluate(dataset, predictions)
        reference = official_ap(dataset, predictions)
        difference = {name: abs(local[name] - reference[name]) for name in reference}
        if max(difference.values()) > 1e-6:
            raise RuntimeError('non-crowd evaluator differs from official COCOeval')
        report['models'][label] = {
            'model_sha256': sha256(model),
            'predictions': len(predictions),
            'reference_ap': reference,
            'custom_ap': {k: local[k] for k in reference},
            'reference_difference': difference,
        }
    report['int8_minus_fp32'] = {
        key: report['models']['int8']['reference_ap'][key]
        - report['models']['fp32']['reference_ap'][key]
        for key in ('AP50', 'AP75', 'mAP50_95')
    }
    (args.output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
