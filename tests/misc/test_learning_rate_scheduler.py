import unittest
import math
from pl_utils.misc import LinearWarmupCosineAnnealingLR, LinearWarmupStepDecayLR


class TestLinearWarmupCosineAnnealingLR(unittest.TestCase):
    """测试线性预热余弦退火调度器"""

    def test_warmup_phase(self):
        """测试预热阶段学习率线性增长"""
        scheduler = LinearWarmupCosineAnnealingLR(
            max_steps=100, lr_initial=0.0, lr_max=1.0, lr_warmup_steps=10
        )
        # 预热阶段应线性增长
        self.assertAlmostEqual(scheduler(0), 0.0)
        self.assertAlmostEqual(scheduler(5), 0.5)
        self.assertAlmostEqual(scheduler(10), 1.0)

    def test_cosine_phase(self):
        """测试余弦退火阶段"""
        scheduler = LinearWarmupCosineAnnealingLR(
            max_steps=100, lr_initial=0.0, lr_max=1.0, lr_end=0.0, lr_warmup_steps=0
        )
        # 开始时应为 lr_max
        self.assertAlmostEqual(scheduler(0), 1.0)
        # 结束时应接近 lr_end
        self.assertAlmostEqual(scheduler(99), 0.0, places=5)

    def test_cycle_phase(self):
        """测试循环阶段"""
        scheduler = LinearWarmupCosineAnnealingLR(
            max_steps=100, lr_max=1.0, lr_end=0.0, lr_warmup_steps=0, lr_cycle_steps=10
        )
        # 超过 max_steps 后应进入循环
        lr_at_100 = scheduler(100)
        lr_at_110 = scheduler(110)
        self.assertAlmostEqual(lr_at_100, lr_at_110, places=5)

    def test_no_cycle(self):
        """测试无循环时返回 lr_end"""
        scheduler = LinearWarmupCosineAnnealingLR(
            max_steps=100, lr_max=1.0, lr_end=0.1, lr_warmup_steps=0, lr_cycle_steps=0
        )
        self.assertAlmostEqual(scheduler(150), 0.1)


class TestLinearWarmupStepDecayLR(unittest.TestCase):
    """测试线性预热阶梯衰减调度器"""

    def test_warmup_phase(self):
        """测试预热阶段"""
        scheduler = LinearWarmupStepDecayLR(
            max_steps=100, lr_initial=0.0, lr_max=1.0, lr_warmup_steps=10
        )
        self.assertAlmostEqual(scheduler(0), 0.0)
        self.assertAlmostEqual(scheduler(10), 1.0)

    def test_step_decay(self):
        """测试阶梯衰减"""
        scheduler = LinearWarmupStepDecayLR(
            max_steps=100, lr_max=1.0, lr_warmup_steps=0, decay_factor=0.5, decay_steps=10
        )
        self.assertAlmostEqual(scheduler(0), 1.0)
        self.assertAlmostEqual(scheduler(10), 0.5)
        self.assertAlmostEqual(scheduler(20), 0.25)

    def test_warmup_steps_as_float(self):
        """测试预热步数为浮点数（比例）"""
        scheduler = LinearWarmupStepDecayLR(
            max_steps=100, lr_initial=0.0, lr_max=1.0, lr_warmup_steps=0.1  # 10%
        )
        self.assertEqual(scheduler.lr_warmup_steps, 10)


if __name__ == '__main__':
    unittest.main()
