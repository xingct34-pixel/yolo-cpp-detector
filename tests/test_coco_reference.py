"""Optional project-local comparison against pinned official pycocotools."""

import importlib.util
from pathlib import Path
import random
import sys
import unittest

if not importlib.util.find_spec('pycocotools'):
    print('SKIP: project-local pycocotools is unavailable')
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_detection import evaluate
from tools.validate_coco8 import official_ap


# 模块：官方交叉验证：固定随机种子、相同分数、空类别和maxDets截断均有已知参考


class ReferenceTests(unittest.TestCase):
    def test_reference_parity(self):
        for seed in range(12):
            rng = random.Random(seed)
            dataset = {
                'info': {},
                'images': [{'id': i} for i in range(1, 4)],
                'categories': [{'id': 1}, {'id': 2}],
                'annotations': [],
            }
            predictions = []
            for image in range(1, 4):
                for category in (1, 2):
                    for _ in range(3):
                        box = [rng.randrange(40), rng.randrange(40), 10, 10]
                        dataset['annotations'].append(
                            {
                                'id': len(dataset['annotations']) + 1,
                                'image_id': image,
                                'category_id': category,
                                'bbox': box,
                                'area': 100,
                                'iscrowd': 0,
                            }
                        )
                        shifted = [box[0] + rng.choice([0, 1, 3, 7]), box[1], 10, 10]
                        predictions.append(
                            {
                                'image_id': image,
                                'category_id': category,
                                'bbox': shifted,
                                'score': rng.choice([0.3, 0.5, 0.8]),
                            }
                        )
            if seed == 0:
                predictions.extend(
                    [
                        {
                            'image_id': 1,
                            'category_id': 1,
                            'bbox': [100 + i, 100, 10, 10],
                            'score': 0.9,
                        }
                        for i in range(105)
                    ]
                )
            custom = evaluate(dataset, predictions)
            reference = official_ap(dataset, predictions)
            for key in reference:
                self.assertAlmostEqual(custom[key], reference[key], places=10, msg=(seed, key))


if __name__ == '__main__':
    unittest.main()
