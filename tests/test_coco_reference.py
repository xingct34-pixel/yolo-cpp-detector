# YOLO 检测部署 | tests/test_coco_reference.py
# 功能：用固定随机种子生成一批预测，把同一份标注和预测分别交给本地评估器和官方 pycocotools 打分。
# 重点覆盖两类容易让 AP 悄悄变化的情况：分数相同时的排序，以及每张图每个类别最多保留 100 个预测的截断。
# 目的：两边结果必须在小数点后 10 位内一致，一旦不一致就说明本地评估在同分或截断边界上有偏差。
# 配合：本地评估是 tools/evaluate_detection.py 的 evaluate，官方结果来自 tools/validate_coco8.py 的 official_ap。
#       没装 pycocotools 时整个文件跳过，不算失败。


"""Optional project-local comparison against pinned official pycocotools."""

import importlib.util
from pathlib import Path
import random
import sys
import unittest

# 没有 pycocotools 就无法和官方结果比，打印提示后用退出码 77 结束（77 是通用的“跳过”约定）。
if not importlib.util.find_spec('pycocotools'):
    print('SKIP: project-local pycocotools is unavailable')
    raise SystemExit(77)
# 把项目根目录加入搜索路径，下面才能写 from tools.xxx import。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_detection import evaluate
from tools.validate_coco8 import official_ap


# 模块：官方交叉验证（ReferenceTests）
# 功能：对 12 组随机数据（种子 0 到 11）逐组比较本地与官方 AP，其中种子 0 额外加入大量高分误报来触发截断。
# 目的：覆盖三种边界：分数相同、框偏移跨过不同 IoU 门槛、每图每类超过 100 个预测。
# 说明：数据里每张图、每个类别都有标注，没有“某类别完全没有真值”的情况，空类别需另外单独测试。
class ReferenceTests(unittest.TestCase):
    # 测试：本地结果与官方一致
    # 功能：每个种子造一份数据，分别算出两套 AP，逐项对比。
    # 目的：任何一个指标对不上，失败信息里的 (seed, key) 能直接告诉你是哪组数据、哪个指标。
    def test_reference_parity(self):


        # 步骤：生成可重复的样本
        # 功能：3 张图 × 2 个类别 × 每格 3 个真值，共 18 个真值，位置由种子决定。
        # 目的：同一个种子永远生成同样的框和分数，失败后拿 seed 就能原样重现。
        for seed in range(12):
            rng = random.Random(seed)
            # 'info' 是官方 pycocotools 读数据时要求存在的字段，本地评估用不到。
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
                        # 真值：随机位置、固定 10×10 大小，标注格式和 COCO 一致。
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
                        # 预测：把真值向右平移 0、1、3 或 7 像素，对应 IoU 约 1、0.82、0.54、0.18，
                        # 正好分布在不同的 IoU 门槛两侧。
                        # 分数只有 0.3、0.5、0.8 三种，会出现大量同分预测，用来检验同分时的排序是否和官方一致。
                        shifted = [box[0] + rng.choice([0, 1, 3, 7]), box[1], 10, 10]
                        predictions.append(
                            {
                                'image_id': image,
                                'category_id': category,
                                'bbox': shifted,
                                'score': rng.choice([0.3, 0.5, 0.8]),
                            }
                        )


            # 步骤：构造截断边界
            # 功能：只在种子 0 中，给图 1 的类别 1 额外加入 105 个分数 0.9 的误报，
            #       这些框远离任何真值，不可能匹配成功。
            # 目的：官方规则是每张图每个类别只保留分数最高的 100 个预测。
            #       这 105 个误报分数最高，会把原来的正常预测挤出前 100 名，
            #       检查本地评估器是否在同一个位置截断、同样丢掉被挤出去的预测。
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
            # 步骤：两套评估器吃同一份数据，逐项对比。
            # places=10 表示小数点后 10 位都要相等，实际上就是要求两边算法完全一致。
            custom = evaluate(dataset, predictions)
            reference = official_ap(dataset, predictions)
            for key in reference:
                self.assertAlmostEqual(custom[key], reference[key], places=10, msg=(seed, key))


if __name__ == '__main__':
    unittest.main()
