import unittest
import torch
from torch import nn
from pl_utils.configs import TrainingConfig, OneOptimizerConfig
from pl_utils.modules import BaseModule


class TestModel(nn.Module):
    """包含各种类型参数的测试模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)  # 2D weight + 1D bias
        self.norm = nn.LayerNorm(10)  # 1D weight + 1D bias
        self.embed = nn.Embedding(10, 10)  # 2D
        self.scalar = nn.Parameter(torch.zeros(1))  # 1D

    def forward(self, x):
        return x


class DummyModule(BaseModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)


class TestAllParamsOptimized(unittest.TestCase):
    """测试所有可学习参数都被优化器覆盖"""

    def _get_all_optimized_params(self, optimizers):
        """获取所有优化器中的参数"""
        all_params = []
        for opt in optimizers:
            for group in opt.param_groups:
                all_params.extend(group['params'])
        return all_params

    def _get_all_trainable_params(self, model):
        """获取模型中所有可学习参数"""
        return [p for p in model.parameters() if p.requires_grad]

    def test_single_optimizer_covers_all(self):
        """单个优化器应覆盖所有参数"""
        model = TestModel()
        training_config = TrainingConfig(optimizers=[OneOptimizerConfig()])
        pl_module = DummyModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        trainable = self._get_all_trainable_params(model)

        self.assertEqual(len(optimized), len(trainable))
        for p in trainable:
            self.assertTrue(any(p is op for op in optimized), f"参数未被优化器覆盖")

    def test_multi_optimizer_covers_all(self):
        """多个优化器应覆盖所有参数，无遗漏"""
        model = TestModel()
        opt1 = OneOptimizerConfig(keywords=['linear'])
        opt2 = OneOptimizerConfig(keywords=['norm'])
        opt3 = OneOptimizerConfig()  # 剩余参数
        training_config = TrainingConfig(optimizers=[opt1, opt2, opt3])
        pl_module = DummyModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        trainable = self._get_all_trainable_params(model)

        self.assertEqual(len(optimized), len(trainable))
        for p in trainable:
            self.assertTrue(any(p is op for op in optimized), f"参数未被优化器覆盖")

    def test_no_duplicate_params(self):
        """参数不应被多个优化器重复使用"""
        model = TestModel()
        opt1 = OneOptimizerConfig(keywords=['linear'])
        opt2 = OneOptimizerConfig()
        training_config = TrainingConfig(optimizers=[opt1, opt2])
        pl_module = DummyModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        # 检查无重复（通过 id 比较）
        param_ids = [id(p) for p in optimized]
        self.assertEqual(len(param_ids), len(set(param_ids)), "存在重复参数")

    def test_frozen_params_excluded(self):
        """冻结的参数不应被优化器使用"""
        model = TestModel()
        # 冻结 linear 层
        for p in model.linear.parameters():
            p.requires_grad = False

        training_config = TrainingConfig(optimizers=[OneOptimizerConfig()])
        pl_module = DummyModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        trainable = self._get_all_trainable_params(model)

        self.assertEqual(len(optimized), len(trainable))
        # 确保冻结的参数不在优化器中
        for p in model.linear.parameters():
            self.assertFalse(any(p is op for op in optimized))

    def test_muon_skips_1d_but_fallback_catches(self):
        """Muon 跳过 1D 参数，但后续优化器应捕获它们"""
        model = TestModel()
        # Muon 优化器（跳过 1D）+ AdamW 兜底
        opt1 = OneOptimizerConfig(
            optimizer=torch.optim.Muon,
            keywords=['linear', 'embed'],
            optimizer_args={'lr': 0.01, 'momentum': 0.95},
        )
        opt2 = OneOptimizerConfig()  # AdamW 兜底
        training_config = TrainingConfig(optimizers=[opt1, opt2])
        pl_module = DummyModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        trainable = self._get_all_trainable_params(model)

        self.assertEqual(
            len(optimized),
            len(trainable),
            f"优化器参数数量 {len(optimized)} != 可训练参数数量 {len(trainable)}",
        )

    def test_muon_as_last_optimizer_skips_1d_with_warning(self):
        """Muon 作为最后一个优化器时，跳过 1D 参数并发出警告"""
        model = TestModel()
        # 只有 Muon，作为最后一个优化器
        opt1 = OneOptimizerConfig(
            optimizer=torch.optim.Muon, optimizer_args={'lr': 0.01, 'momentum': 0.95}
        )
        training_config = TrainingConfig(optimizers=[opt1])
        pl_module = DummyModule(model, training_config)

        # 应该能正常运行，但会有警告（1D 参数未被优化）
        optimizers, _ = pl_module.configure_optimizers()

        optimized = self._get_all_optimized_params(optimizers)
        # Muon 只优化 2D 参数
        params_2d = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        self.assertEqual(len(optimized), len(params_2d))


if __name__ == '__main__':
    unittest.main()
