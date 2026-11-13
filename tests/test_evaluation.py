# YOLO 检测部署 | tests/test_evaluation.py
# 为 evaluate_detection.py 准备有已知答案的小样本。一个真值配一个完全重合的正确框，
# 与漏检、重复预测或错分类别比较，可以单独判断匹配和召回率计算有无偏差。


import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'evaluation', Path(__file__).resolve().parents[1] / 'tools/evaluate_detection.py'
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


# 模块：已知答案：用可手算的预测验收AP、重复框、错误类别和非法数据


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            'images': [{'id': 1}, {'id': 2}],
            'categories': [{'id': 1}, {'id': 2}],
            'annotations': [
                {'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
                {'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
            ],
        }
        self.predictions = [dict(row, score=0.8) for row in self.dataset['annotations']]

    def test_perfect_and_empty(self):
        self.assertAlmostEqual(evaluation.evaluate(self.dataset, self.predictions)['mAP50_95'], 1)
        self.assertEqual(evaluation.evaluate(self.dataset, [])['AP50'], 0)

    def test_half_recall_and_false_positive(self):
        self.assertAlmostEqual(
            evaluation.evaluate(self.dataset, self.predictions[:1])['AP50'], 51 / 101
        )
        false = dict(self.predictions[0], score=0.9, bbox=[50, 50, 10, 10])
        self.assertAlmostEqual(
            evaluation.evaluate(self.dataset, [false] + self.predictions)['AP50'], 2 / 3
        )

    def test_class_and_iou(self):
        wrong = [dict(p, category_id=2) for p in self.predictions]
        self.assertEqual(evaluation.evaluate(self.dataset, wrong)['AP50'], 0)
        shifted = [dict(p, bbox=[2, 0, 10, 10]) for p in self.predictions]
        result = evaluation.evaluate(self.dataset, shifted)
        self.assertEqual(result['AP50'], 1)
        self.assertEqual(result['AP75'], 0)

    def test_recall_grid_floating_boundary(self):
        truths = [
            {'image_id': 1, 'category_id': 1, 'bbox': [index * 20, 0, 10, 10]}
            for index in range(10)
        ]
        predictions = [dict(row, score=0.99 - index * 0.01) for index, row in enumerate(truths[:7])]
        predictions += [
            dict(truths[0], bbox=[1000, 0, 10, 10], score=0.5),
            dict(truths[7], score=0.4),
        ]
        # r=0.70的官方浮点网格略大于7/10，须取下一个召回位置；不能人为加epsilon。
        expected = (70 + 11 * (8 / 9)) / 101
        self.assertAlmostEqual(evaluation.class_ap(truths, predictions, 0.5), expected)

    def test_reject_unsupported_and_corrupt(self):
        dataset = copy.deepcopy(self.dataset)
        dataset['annotations'][0]['iscrowd'] = 1
        with self.assertRaises(ValueError):
            evaluation.evaluate(dataset, self.predictions)
        for change in [{'bbox': [0, 0, 0, 1]}, {'score': float('nan')}, {'image_id': 999}]:
            with self.assertRaises(ValueError):
                evaluation.evaluate(self.dataset, [dict(self.predictions[0], **change)])


if __name__ == '__main__':
    unittest.main()
