# YOLO 检测部署 | tests/test_profile.py
# 功能：构造短小的 ORT 时间线，交给 bench/ort_profile.py 的 summarize 汇总。
# 把同类算子放在不同后端，并混入其他事件，检验报告会正确分组和筛选。
# 目的：防止把会话总时间或其他后端时间混进某个算子的统计。


import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'profile_report', Path(__file__).resolve().parents[1] / 'bench/ort_profile.py'
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


# 模块：Profile口径（ProfileTests）
# 功能：排除Session总计和fence事件，避免把同一耗时重复相加
class ProfileTests(unittest.TestCase):
    def test_kernel_only_and_provider_separation(self):
        events = [
            {
                'cat': 'Node',
                'name': 'a_kernel_time',
                'dur': 10,
                'args': {'op_name': 'Conv', 'provider': 'CPUExecutionProvider'},
            },
            {
                'cat': 'Node',
                'name': 'b_kernel_time',
                'dur': 20,
                'args': {'op_name': 'Conv', 'provider': 'CPUExecutionProvider'},
            },
            {'cat': 'Node', 'name': 'a_fence_before', 'dur': 1000},


            # 模拟耗时
            # Session 事件的 1000 不应算进节点。两个 CPU Conv 的 10+20 应算 30，另一个 provider 单独保留。
            # CPU Conv 两次用时 10 和 20，期望合计 30。旁边特意放一个 1000 的会话事件；若把它也累加，测试会立刻发现计时范围变了。
            {'cat': 'Session', 'name': 'model_run', 'dur': 1000},
            {
                'cat': 'Node',
                'name': 'c_kernel_time',
                'dur': 15,
                'args': {'op_name': 'Conv', 'provider': 'CUDAExecutionProvider'},
            },
        ]
        rows = profile.summarize(events)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['duration_us'], 30)
        self.assertEqual(rows[0]['calls'], 2)
        self.assertEqual(rows[1]['provider'], 'CUDAExecutionProvider')


if __name__ == '__main__':
    unittest.main()
