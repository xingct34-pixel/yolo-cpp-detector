# YOLO 检测部署 | tests/test_coco_reference.py
# 用固定随机种子生成预测，再把同一份标注与预测交给两个评估器。
# 对照 pycocotools 检查排序同分、每图预测数量限制等容易让 AP 悄悄变化的情况。
#
# 阅读地图（按源码顺序）
#   1 官方交叉验证 → 2 可重复造样本 → 3 截断边界


"""Optional project-local comparison against pinned official pycocotools."""

import importlib.util  # 按文件路径装载被测模块或检查依赖
from pathlib import Path  # 拼接目录、文件名并读取文件
import random  # 用种子产生可重复的随机样本
import sys  # 访问解释器参数、模块搜索路径等运行环境
import unittest  # 组织测试用例并核对预期结果

if not importlib.util.find_spec('pycocotools'):
    print('SKIP: project-local pycocotools is unavailable')
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_detection import evaluate
from tools.validate_coco8 import official_ap


# 模块 1：官方交叉验证
# 功能：固定随机种子、相同分数、空类别和maxDets截断均有已知参考


class ReferenceTests(unittest.TestCase):
    def test_reference_parity(self):


        # 模块 2：可重复造样本
        # 固定随机种子后，失败消息中的 seed 可以重现具体的框与分数。


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


            # 模块 3：截断边界
            # 额外放入超过 100 个高分误报，检查两套评分器是否在每图每类相同的位置截断。


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
