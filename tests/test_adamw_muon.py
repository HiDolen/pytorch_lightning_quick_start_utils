import unittest
import torch
from torch import nn
from pl_utils.configs import TrainingConfig, OneOptimizerConfig
from pl_utils.modules import BaseModule
from pl_utils.misc import LinearWarmupCosineAnnealingLR, LinearWarmupStepDecayLR


class TestModel(nn.Module):
    """测试模型：包含输入层、中间层和输出层"""

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(10, 20)
        self.middle_layer1 = nn.Linear(20, 20)
        self.middle_layer2 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(self.middle_layer1(x))
        x = torch.relu(self.middle_layer2(x))
        return self.output_layer(x).mean()


class TestModule(BaseModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0)


class TestAdamWMuon(unittest.TestCase):

    def test_adamw_muon_dual_optimizer(self):
        """测试 AdamW + Muon 双优化器配置"""
        model = TestModel()

        # Muon 优化器：专注于中间层权重（不包括 bias），使用余弦退火调度器
        muon_config = OneOptimizerConfig(
            optimizer=torch.optim.Muon,
            optimizer_args={
                'lr': 1,
                'momentum': 0.95,
                'weight_decay': 0.01,
            },
            scheduler=LinearWarmupCosineAnnealingLR(
                max_steps=1000,
                lr_initial=1e-5,
                lr_max=0.02,
                lr_end=1e-6,
                lr_warmup_steps=100,
            ),
            keywords=['middle_layer'],
            excluded_from_weight_decay=['bias'],  # Muon 只支持 2D 参数，自动排除 bias
        )

        # AdamW 优化器：处理剩余参数（输入层和输出层），使用阶梯衰减调度器
        adamw_config = OneOptimizerConfig(
            optimizer=torch.optim.AdamW,
            optimizer_args={
                'lr': 1,
                'betas': (0.9, 0.999),
                'weight_decay': 0.01,
            },
            scheduler=LinearWarmupStepDecayLR(
                max_steps=1000,
                lr_initial=1e-6,
                lr_max=1e-3,
                lr_warmup_steps=100,
                decay_factor=0.9,
                decay_steps=200,
            ),
            excluded_from_weight_decay=['bias'],
        )

        training_config = TrainingConfig(
            optimizers=[muon_config, adamw_config],
            accumulate_grad_batches=2,
        )

        pl_module = TestModule(model, training_config)
        optimizers, schedulers = pl_module.configure_optimizers()

        # 验证优化器数量
        self.assertEqual(len(optimizers), 2)
        self.assertEqual(len(schedulers), 2)

        # 验证 Muon 优化器类型和参数（Muon 只支持 2D 参数，即权重矩阵）
        self.assertIsInstance(optimizers[0], torch.optim.Muon)
        muon_params = [p for group in optimizers[0].param_groups for p in group['params']]
        middle_weights = [model.middle_layer1.weight, model.middle_layer2.weight]
        self.assertEqual(len(muon_params), 2)  # 中间层的权重矩阵

        # 验证 AdamW 优化器类型和参数
        self.assertIsInstance(optimizers[1], torch.optim.AdamW)
        adamw_params = [p for group in optimizers[1].param_groups for p in group['params']]
        other_params = list(model.input_layer.parameters()) + list(model.output_layer.parameters())
        self.assertEqual(len(adamw_params), 6)  # 输入输出 Linear 和中间态的 bias

        # 验证所有可学习参数都被优化器覆盖
        all_model_params = set(model.parameters())
        all_optimizer_params = set(muon_params + adamw_params)
        self.assertEqual(all_model_params, all_optimizer_params, "所有可学习参数必须被优化器覆盖")

        # 验证没有参数被重复分配
        self.assertEqual(
            len(muon_params) + len(adamw_params), len(all_optimizer_params), "参数不应被重复分配"
        )

        # 验证学习率调度器
        self.assertEqual(schedulers[0]['interval'], 'step')
        self.assertEqual(schedulers[1]['interval'], 'step')

        print("✓ AdamW + Muon 双优化器配置测试通过")

    def test_learning_rate_schedules(self):
        """测试两个优化器的学习率调度"""
        # Muon: 余弦退火
        muon_scheduler = LinearWarmupCosineAnnealingLR(
            max_steps=1000,
            lr_initial=1e-5,
            lr_max=0.02,
            lr_end=1e-6,
            lr_warmup_steps=100,
        )

        # AdamW: 阶梯衰减
        adamw_scheduler = LinearWarmupStepDecayLR(
            max_steps=1000,
            lr_initial=1e-6,
            lr_max=1e-3,
            lr_warmup_steps=100,
            decay_factor=0.9,
            decay_steps=200,
        )

        # 验证预热阶段
        self.assertAlmostEqual(muon_scheduler(0), 1e-5, places=7)
        self.assertAlmostEqual(adamw_scheduler(0), 1e-6, places=7)

        # 验证预热结束
        self.assertAlmostEqual(muon_scheduler(100), 0.02, places=7)
        self.assertAlmostEqual(adamw_scheduler(100), 1e-3, places=7)

        # 验证最终学习率
        self.assertAlmostEqual(muon_scheduler(1000), 1e-6, places=7)
        self.assertLess(adamw_scheduler(1000), 1e-3)

        print("✓ 学习率调度测试通过")

    def test_weight_decay_exclusion(self):
        """测试 weight decay 排除机制"""
        model = TestModel()

        muon_config = OneOptimizerConfig(
            optimizer=torch.optim.Muon,
            optimizer_args={'lr': 1, 'momentum': 0.95, 'weight_decay': 0.01},
            keywords=['middle_layer'],
            excluded_from_weight_decay=['bias'],
        )

        training_config = TrainingConfig(optimizers=[muon_config, OneOptimizerConfig()])
        pl_module = TestModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        # Muon 优化器应该有两个参数组
        muon_optimizer = optimizers[0]
        self.assertEqual(len(muon_optimizer.param_groups), 2)

        # 验证 bias 参数在无 weight decay 组中
        no_wd_group = muon_optimizer.param_groups[1]
        self.assertEqual(no_wd_group['weight_decay'], 0.0)

        print("✓ Weight decay 排除测试通过")

    def test_all_parameters_covered(self):
        """测试所有可学习参数都被优化器覆盖"""
        model = TestModel()

        muon_config = OneOptimizerConfig(
            optimizer=torch.optim.Muon,
            optimizer_args={'lr': 1, 'momentum': 0.95, 'weight_decay': 0.01},
            keywords=['middle_layer'],
            excluded_from_weight_decay=['bias'],
        )

        adamw_config = OneOptimizerConfig(
            optimizer=torch.optim.AdamW,
            optimizer_args={'lr': 1e-3, 'weight_decay': 0.01},
            excluded_from_weight_decay=['bias'],
        )

        training_config = TrainingConfig(optimizers=[muon_config, adamw_config])
        pl_module = TestModule(model, training_config)
        optimizers, _ = pl_module.configure_optimizers()

        # 收集所有模型参数
        all_model_params = set(model.parameters())

        # 收集所有优化器中的参数
        all_optimizer_params = set()
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                all_optimizer_params.update(group['params'])

        # 验证覆盖完整性
        self.assertEqual(all_model_params, all_optimizer_params, "所有可学习参数必须被优化器覆盖")

        # 验证无重复分配
        total_params_count = sum(
            len(group['params']) for opt in optimizers for group in opt.param_groups
        )
        self.assertEqual(
            total_params_count, len(all_optimizer_params), "参数不应被重复分配到多个优化器"
        )

        # 验证参数总数
        self.assertEqual(len(all_model_params), 8, "模型应有 8 个参数（4 个权重 + 4 个偏置）")

        print("✓ 参数覆盖完整性测试通过")


if __name__ == '__main__':
    unittest.main()
