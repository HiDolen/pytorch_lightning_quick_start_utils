import unittest
import torch
import torch.nn as nn
from pl_utils.misc.model_summary import get_human_readable_count, get_model_parameters_count


class TestGetHumanReadableCount(unittest.TestCase):
    """测试数字可读化函数"""

    def test_zero(self):
        """测试零值"""
        self.assertEqual(get_human_readable_count(0), "0")

    def test_small_numbers(self):
        """测试小数字（小于1000）"""
        self.assertEqual(get_human_readable_count(1), "1")
        self.assertEqual(get_human_readable_count(99), "99")
        self.assertEqual(get_human_readable_count(999), "999")

    def test_thousands(self):
        """测试千位数"""
        self.assertEqual(get_human_readable_count(1000), "1.0K")
        self.assertEqual(get_human_readable_count(1234), "1.2K")
        self.assertEqual(get_human_readable_count(99999), "100.0K")

    def test_millions(self):
        """测试百万位数"""
        self.assertEqual(get_human_readable_count(1000000), "1.0M")
        self.assertEqual(get_human_readable_count(2500000), "2.5M")
        self.assertEqual(get_human_readable_count(100000000), "100M")

    def test_billions(self):
        """测试十亿位数"""
        self.assertEqual(get_human_readable_count(1000000000), "1.0B")
        self.assertEqual(get_human_readable_count(5500000000), "5.5B")

    def test_trillions(self):
        """测试万亿位数"""
        self.assertEqual(get_human_readable_count(1000000000000), "1.0T")


class TestGetModelParametersCount(unittest.TestCase):
    """测试模型参数统计函数"""

    def test_simple_linear_model(self):
        """测试简单线性模型"""
        model = nn.Linear(10, 5)
        # 参数量: 10*5 + 5 = 55
        result = get_model_parameters_count(model)
        self.assertEqual(result["total"], 55)
        self.assertEqual(result["trainable"], 55)
        self.assertEqual(result["total_readable"], "55")
        self.assertEqual(result["trainable_readable"], "55")

    def test_model_with_frozen_params(self):
        """测试包含冻结参数的模型"""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        # 冻结第一层
        for param in model[0].parameters():
            param.requires_grad = False

        result = get_model_parameters_count(model)
        # 总参数: (10*5+5) + (5*2+2) = 67
        # 可训练参数: (5*2+2) = 12
        self.assertEqual(result["total"], 67)
        self.assertEqual(result["trainable"], 12)

    def test_empty_model(self):
        """测试空模型"""
        model = nn.Module()
        result = get_model_parameters_count(model)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["trainable"], 0)
        self.assertEqual(result["total_readable"], "0")

    def test_conv_model(self):
        """测试卷积模型"""
        model = nn.Conv2d(3, 16, kernel_size=3)
        # 参数量: 3*16*3*3 + 16 = 448
        result = get_model_parameters_count(model)
        self.assertEqual(result["total"], 448)
        self.assertEqual(result["trainable"], 448)

    def test_large_model_readable_format(self):
        """测试大模型的可读格式"""
        model = nn.Linear(1000, 1000)
        # 参数量: 1000*1000 + 1000 = 1,001,000
        result = get_model_parameters_count(model)
        self.assertEqual(result["total"], 1001000)
        self.assertEqual(result["total_readable"], "1.0M")


if __name__ == '__main__':
    unittest.main()
