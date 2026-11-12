import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'profile_report', Path(__file__).resolve().parents[1] / 'bench/ort_profile.py'
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


# 模块：Profile口径：排除Session总计和fence事件，避免把同一耗时重复相加


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
