# PyTorch Lightning Quick Start Utils

本库为 PyTorch Lightning 的训练提前写好一些符合通常的（主要是我的）训练习惯的代码，以便快速开始训练。

PyTorch Lightning 将训练代码的编写规范化，让代码复用更加便利。本库在此基础上加入了一些小的个性化，把经常使用的代码放了上去，希望能帮助通用的模型训练。虽然还是比较简陋，但应该会随着时间推移渐渐完善起来。

## 谁适合用这个库？

目前需要是刚好符合这些习惯的人：

- **使用 VSCode 且用 ipynb 写训练代码**
- 使用 PyTorch + PyTorch Lightning 训练模型
- 使用 Tensorboard 记录 log
  - log 被 Lightning 存入默认路径 `./lightning_logs`，各 log 命名格式为 `version_xx_xxxx`
- 使用线性预热、余弦退火的策略进行学习率调度
- 不把模型结构定义写在 `pytorch_lightning.LightningModule` 内

## 安装

直接通过仓库安装：

```bash
pip install git+https://github.com/HiDolen/pytorch_lightning_quick_start_utils
```

或是选择将本项目 clone 到本地，使用 `pip install -e .` 安装。

## 使用示例

完整示例可见仓库中的 `example.ipynb` 文件。

```python
from pl_utils import init_before_training

init_before_training()  # 清理缓存，设置随机种子，设置浮点数精度
```

```python
from pl_utils import LearningRateConfig, TrainingConfig
from pl_utils import BaseModule

class ResNetForMNIST(BaseModule):
    def __init__(
        self,
        model,
        lr_config: LearningRateConfig,
        training_config: TrainingConfig,
    ):
        super(ResNetForMNIST, self).__init__(model, lr_config, training_config)
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

learning_rate_config = LearningRateConfig(
    lr_warmup_steps=10,
    lr_initial=1e-5,
    lr_max=1e-4,
    lr_end=1e-5,
)

training_config = TrainingConfig(optimizer="adamw")

model = ResNet(ResidualBlock, [2, 2, 2, 2])
pl_model = ResNetForMNIST(model, learning_rate_config, training_config)
```

## 这个库帮忙提前写好了什么？

将 `lightning.LightningModule` 再次封装为 `BaseModule`，实现了以下几个接口和功能：

- `on_train_start()`
  - 使用 `__vsc_ipynb_file__` 获取 ipynb 文件路径，并保存一份文件到 log 路径
  - 实例化一个线性预热、余弦退火的学习率调度器
  - 将本次训练全程的学习率变化写入到 log
- `configure_optimizers()`
  - 注册优化器、学习率调度器
  - 默认取消 bias、norm 和 embed 权重的 weight decay

为了配置方便，有这些 config 类：

- `LearningRateConfig`，控制预热退火各个阶段的学习率
- `TrainingConfig`，配置优化器

额外的功能：

- `init_before_training()`，清理缓存，设置随机种子，设置浮点数精度

## 这个库没涉及什么？

库不会涉及这些内容。

- 与数据加载相关的一切。因为我认为不同的数据要写成不同的 dataset 和 dataloader，除非是针对特定领域，否则很难有唯一的标准
- `pytorch_lightnin.LightningModule` 的 `training_step()`、`validation_step()` 等接口。原因同上，不同领域、不同模型有不同的写法
- 与 loss 计算相关的一切
