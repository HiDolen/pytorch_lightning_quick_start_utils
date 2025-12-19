from dataclasses import dataclass, field
from typing import Union, Literal, TypedDict
from functools import partial

import torch


@dataclass
class OneOptimizerConfig:
    """
    配置单个优化器及其相关的学习率调度器。

    Args:
        optimizer (`type`, default to `torch.optim.AdamW`):
            优化器类，支持未实例化的自定义优化器类。
        optimizer_args (`dict`):
            优化器参数。由于有学习率调度器，`lr` 必须设为 1。
            默认值：`lr=1, weight_decay=1e-2`。
        scheduler (`object`):
            学习率调度器函数（lambda x: ...）。默认是一个返回 1.0 的 lambda 函数。
        keywords (`list[str]`):
            用于匹配参数名的关键词列表。如果参数名包含其中任一关键词，则归该优化器管理。
        excluded_from_weight_decay (`list[str]`, default to `["bias", "norm", "embed"]`):
            包含这些关键词的参数将不使用 weight decay。
    """

    optimizer: type = torch.optim.AdamW
    optimizer_args: dict = field(default_factory=dict)
    scheduler: object = field(default_factory=lambda: lambda x: 1.0)
    keywords: list[str] = field(default_factory=list)
    excluded_from_weight_decay: list[str] = field(default_factory=lambda: ["bias", "norm", "embed"])

    def __post_init__(self):
        default_args = {
            "lr": 1,  # 该参数应当固定为 1。真正 lr 由学习率调度器控制
            # 'betas': (0.9, 0.95),
            'weight_decay': 1e-2,
        }
        self.optimizer_args = {**default_args, **self.optimizer_args}


@dataclass
class TrainingConfig:
    """
    配置训练相关内容。实际就是包含多个优化器配置。

    Args:
        optimizers (`list[OneOptimizerConfig]`):
            优化器配置列表。根据传入顺序依次取出模型参数，剩下的参数都给最后一个优化器来优化。
        accumulate_grad_batches (`int`, default to `1`):
            梯度累积批次数。用于手动优化模式下的梯度累积。
    """

    optimizers: list[OneOptimizerConfig] = field(default_factory=lambda: [OneOptimizerConfig()])
    accumulate_grad_batches: int = 1
