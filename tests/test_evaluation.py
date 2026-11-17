# YOLO 检测部署 | tests/test_evaluation.py
# 功能：给 tools/evaluate_detection.py 准备几组“答案能手算”的小数据，逐项检查它算出的 AP 对不对。
# 目的：评估结果不对时，靠这些小数据快速判断问题出在哪一类：
#       框匹配错了、召回率算错了，还是输入数据格式不合法没有被拒绝。
# 配合：被测代码在 tools/evaluate_detection.py，下面通过路径直接加载，不需要把 tools 装成包。
#       运行方式：在项目根目录执行 python -m unittest tests.test_evaluation


import copy
import importlib.util
from pathlib import Path
import unittest

# 按文件路径加载 evaluate_detection.py，加载后用 evaluation 这个名字调用里面的函数。
spec = importlib.util.spec_from_file_location(
    'evaluation', Path(__file__).resolve().parents[1] / 'tools/evaluate_detection.py'
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


# 模块：已知答案（EvaluationTests）
# 功能：用能手算结果的预测框，检查 AP、漏检、多余框、错误类别、框偏移和非法数据这几种情况。
# 目的：每个测试只改动一个因素，失败时能直接定位到是哪一环出了偏差。
# 基础数据：两张图各有 1 个同类真值（框都是 [0, 0, 10, 10]），
#           基础预测就是把真值原样复制一份并给 0.8 分，也就是“两个都预测对”的理想情况。
# 说明：下面的框格式是 [x, y, 宽, 高]，和 COCO 数据集一致。
class EvaluationTests(unittest.TestCase):
    # 准备：每个测试开始前都会重新生成一份数据集和理想预测，测试之间互不影响。
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


    # 测试：全对与没预测
    # 功能：预测和真值完全重合时指标应满分；一个框都不预测时 AP50 应为 0。
    # 目的：确认最理想和最糟的两个端点没有算偏。
    def test_perfect_and_empty(self):
        self.assertAlmostEqual(evaluation.evaluate(self.dataset, self.predictions)['mAP50_95'], 1)
        self.assertEqual(evaluation.evaluate(self.dataset, [])['AP50'], 0)


    # 测试：漏检与多余框
    # 功能：分别检查“只检出一个真值”和“最前面多了一个错框”时的 AP50。
    # 目的：确认召回率不足、精确率下降这两种情况的扣分都和手算一致。
    def test_half_recall_and_false_positive(self):
        # 情况一：两个真值只检出一个，召回率最高只到 0.5。
        # AP 把召回率 0 到 1 均分成 101 个采样点，0 到 0.5（含两端）有 51 个点有精确率 1，其余点为 0，
        # 所以 AP = 51/101。
        self.assertAlmostEqual(
            evaluation.evaluate(self.dataset, self.predictions[:1])['AP50'], 51 / 101
        )
        # 情况二：在两个正确框前面加一个分数更高（0.9）、位置离真值很远的错框。
        # 按分数从高到低排：错框、正确框、正确框，精确率依次是 0/1、1/2、2/3。
        # 每个召回率位置取“它右边最大的精确率”，整条曲线都是 2/3，所以 AP = 2/3。
        false = dict(self.predictions[0], score=0.9, bbox=[50, 50, 10, 10])
        self.assertAlmostEqual(
            evaluation.evaluate(self.dataset, [false] + self.predictions)['AP50'], 2 / 3
        )


    # 测试：类别与 IoU 门槛
    # 功能：检查类别预测错时不能算对，以及框偏移后在不同 IoU 门槛下结果不同。
    # 目的：确认匹配时同时看类别和重合度，且 AP50 与 AP75 用的是各自的门槛。
    def test_class_and_iou(self):
        # 位置全对但类别写成 2，真值是类别 1，应一个都匹配不上，AP50 为 0。
        wrong = [dict(p, category_id=2) for p in self.predictions]
        self.assertEqual(evaluation.evaluate(self.dataset, wrong)['AP50'], 0)
        # 把框向右平移 2 像素：交集 8×10=80，并集 100+100-80=120，IoU≈0.67，
        # 介于 0.5 和 0.75 之间，所以同一批预测在 AP50 下算对，在 AP75 下算错。
        shifted = [dict(p, bbox=[2, 0, 10, 10]) for p in self.predictions]
        result = evaluation.evaluate(self.dataset, shifted)
        self.assertEqual(result['AP50'], 1)
        self.assertEqual(result['AP75'], 0)


    # 测试：召回率网格的浮点边界
    # 功能：构造“召回率恰好 7/10”的场景，检查 class_ap 在 0.70 这个采样点怎么取值。
    # 目的：防止浮点误差让采样点被错分到左边或右边，结果和官方 COCO 评估对不上。
    def test_recall_grid_floating_boundary(self):
        # 同一张图里放 10 个互不重叠的真值。
        truths = [
            {'image_id': 1, 'category_id': 1, 'bbox': [index * 20, 0, 10, 10]}
            for index in range(10)
        ]
        # 分数从高到低依次是：前 7 个真值命中 → 1 个远离真值的错框（0.5）→ 第 8 个真值命中（0.4）。
        predictions = [dict(row, score=0.99 - index * 0.01) for index, row in enumerate(truths[:7])]
        predictions += [
            dict(truths[0], bbox=[1000, 0, 10, 10], score=0.5),
            dict(truths[7], score=0.4),
        ]
        # 前 7 个命中时召回率是 7/10，精确率是 1；随后错框让精确率降到 7/8，第 8 个命中后召回率 8/10、精确率 8/9。
        # 官方做法里第 70 个采样点在浮点数中是 0.7000000000000001，比 7/10 略大，
        # 所以它不能算作“已达到 0.7 召回率”，要取更右边的精确率 8/9。
        # 因此：第 0 到 69 个点（共 70 个）精确率为 1，第 70 到 80 个点（共 11 个）为 8/9，其余为 0。
        # 这里不能人为加一个很小的数（epsilon）去修正，否则会和官方结果不一致。
        expected = (70 + 11 * (8 / 9)) / 101
        self.assertAlmostEqual(evaluation.class_ap(truths, predictions, 0.5), expected)


    # 测试：拒绝不支持和损坏的数据
    # 功能：检查遇到 iscrowd 标注、宽高为 0 的框、分数为 nan、图片编号不存在时会抛出 ValueError。
    # 目的：确认评估代码遇到不合规的输入时会报错，不会悄悄算出一个看似正常的分数。
    def test_reject_unsupported_and_corrupt(self):
        # 本评估不支持 iscrowd（人群标注），出现就应报错。
        dataset = copy.deepcopy(self.dataset)
        dataset['annotations'][0]['iscrowd'] = 1
        with self.assertRaises(ValueError):
            evaluation.evaluate(dataset, self.predictions)
        # 依次检查三种坏预测：框宽度为 0、分数是 nan、预测引用了数据集中没有的图片编号 999。
        for change in [{'bbox': [0, 0, 0, 1]}, {'score': float('nan')}, {'image_id': 999}]:
            with self.assertRaises(ValueError):
                evaluation.evaluate(self.dataset, [dict(self.predictions[0], **change)])


if __name__ == '__main__':
    unittest.main()
