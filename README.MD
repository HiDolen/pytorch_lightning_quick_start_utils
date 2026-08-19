# PyTorch Lightning Quick Start Utils

本库为 PyTorch Lightning 的训练提前写好一些符合通常的（主要是我的）训练习惯的代码，以便快速开始训练。

PyTorch Lightning 将训练代码的编写规范化，让代码复用更加便利。本库在此基础上加入了一些小的个性化，把经常使用的代码放了上去，希望能帮助通用的模型训练。虽然还是比较简陋，但应该会随着时间推移渐渐完善起来。

## 谁适合用这个库？

目前需要是刚好符合这些习惯的人：

- **使用 VSCode 且用 ipynb 写训练代码**
- 使用 PyTorch + PyTorch Lightning 训练模型
- 使用 TensorBoard 记录 log
  - log 被 Lightning 存入默认路径 `./lightning_logs`，各 log 命名格式为 `version_xx_xxxx`
- 使用线性预热 + 余弦退火 / 固定比例衰减 的学习率调度策略
- 需要灵活配置多优化器（如 AdamW + Muon 混合优化）并统一管理学习率
- 不把模型结构定义写在 `lightning.LightningModule` 内

## 安装

直接通过仓库安装：

```bash
pip install git+https://github.com/HiDolen/pytorch_lightning_quick_start_utils
```

或是选择将本项目 clone 到本地，使用 `pip install -e .` 安装。

环境要求：Python >= 3.12, PyTorch >= 2.9, Lightning >= 2.4.0

## 使用示例

完整示例可见仓库中的 `example.ipynb` 文件，以下为关键步骤摘要。

### 训练前初始化

```python
from pl_utils import init_before_training

init_before_training()  # 清理缓存，设置随机种子，设置浮点数精度
```

### 配置训练

```python
from pl_utils import TrainingConfig, OneOptimizerConfig
from pl_utils.misc import LinearWarmupCosineAnnealingLR

scheduler = LinearWarmupCosineAnnealingLR(
    max_steps=100,
    lr_warmup_steps=10,
    lr_initial=1e-5,
    lr_max=1e-4,
    lr_end=1e-5,
)

training_config = TrainingConfig(
    optimizers=[
        OneOptimizerConfig(
            optimizer=torch.optim.AdamW,
            optimizer_args={
                'betas': (0.9, 0.95),
                'weight_decay': 1e-3,
                'fused': True,
            },
        )
    ],
    scheduler=scheduler,
)
```

### 定义 LightningModule

```python
from pl_utils import BaseModule

class ResNetForMNIST(BaseModule):
    def __init__(self, model, training_config: TrainingConfig):
        super().__init__(model, training_config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        return loss
```

### 数据加载

```python
from pl_utils import get_train_val_dataloader

train_loader, val_loader = get_train_val_dataloader(
    [train_dataset, val_dataset],
    batch_size=64,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
    persistent_workers=True,
    train_shuffle=True,
)
```

### 训练与恢复

```python
from lightning.pytorch.loggers import TensorBoardLogger
from pl_utils.lightning import format_next_version_name, get_specified_version_name

logger = TensorBoardLogger(save_dir="./", version=format_next_version_name("训练测试"))
trainer = L.Trainer(logger=logger, **trainer_config)
trainer.fit(pl_model, train_loader, val_loader)

# 从最新检查点恢复训练
current_version = get_specified_version_name()
trainer.fit(pl_model, train_loader, val_loader, ckpt_path="last")
```

## 这个库帮忙提前写好了什么？

### `BaseModule`

将 `lightning.LightningModule` 再次封装为 `BaseModule`，实现了以下接口和功能：

- `on_train_start()`
  - 使用 `__vsc_ipynb_file__` 获取 ipynb 或 py 文件路径，并保存一份文件到 log 路径
  - 根据训练配置实例化学习率调度器
  - 将本次训练全程的学习率变化写入到 TensorBoard log
- `configure_optimizers()`
  - 支持多优化器配置，通过 `JointOptimizer` 打包为单个优化器供 Lightning 调用
  - 支持按参数名关键词分配参数到不同优化器，且Muon 优化器自动跳过 1D 参数
  - 默认取消 bias、norm 和 embed 权重的 weight decay
- `on_save_checkpoint()`
  - 当 `save_weights_only=True` 时，补充 optimizer/scheduler 状态，保证 Lightning 能恢复训练

### 配置类

- `TrainingConfig`：配置训练相关内容，包含多个优化器配置和学习率调度器
- `OneOptimizerConfig`：配置单个优化器，包括优化器类、参数、关键词匹配和 weight decay 排除规则

### 学习率调度器

- `LinearWarmupCosineAnnealingLR`：线性预热 + 余弦退火，支持循环（cosine restart）
- `LinearWarmupStepDecayLR`：线性预热 + 固定比例衰减

### 工具函数

- `init_before_training()`：清理缓存，设置随机种子，设置浮点数精度
- `get_train_val_dataloader()`：辅助获取 DataLoader
- `get_model_parameters_count()`：获取模型的参数统计信息
- `is_notebook()`：判断当前环境是否为 Jupyter Notebook
- `format_next_version_name()` / `get_specified_version_name()`：管理 Lightning log 版本目录名称

### CLI

安装后会注册 `pl` 命令行工具：

```bash
pl log  # 启动 TensorBoard，带有预设参数
```

## 这个库没涉及什么？

库不会涉及这些内容。

- 与数据加载相关的一切。因为我认为不同的数据要写成不同的 dataset 和 dataloader，除非是针对特定领域，否则很难有唯一的标准
- `lightning.LightningModule` 的 `training_step()`、`validation_step()` 等接口。原因同上，不同领域、不同模型有不同的写法
- 与 loss 计算相关的一切
