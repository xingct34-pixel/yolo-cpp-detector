"""Optional preprocessing contract tests for the CPU quantization environment."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('cpu_quantize', Path(__file__).resolve().parents[1] / 'tools/cpu_quantize.py')
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)


class PreprocessingTests(unittest.TestCase):
    def test_rgb_and_normalization(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        tensor = quant.prepare(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor[0, 0] == 1))
        self.assertTrue(np.all(tensor[0, 1:] == 0))

    def test_letterbox_padding(self):
        tensor = quant.prepare(np.zeros((333, 1000, 3), dtype=np.uint8))
        np.testing.assert_allclose(tensor[0, :, :213], 114 / 255, atol=1e-7)
        self.assertTrue(np.all(tensor[0, :, 213:426] == 0))
        np.testing.assert_allclose(tensor[0, :, 426:], 114 / 255, atol=1e-7)

    def test_invalid_input(self):
        for image in (None, np.zeros((2, 2)), np.zeros((0, 2, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(image, 'shape', None)):
                with self.assertRaises(ValueError):
                    quant.prepare(image)


if __name__ == '__main__':
    unittest.main()
