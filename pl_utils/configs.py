from dataclasses import dataclass, field
from typing import Union, Literal
from functools import partial


@dataclass
class LearningRateConfig:
    """

    Args:
        lr_initial: float = 1e-6
            初始学习率。
        lr_max: float = 1e-4
            预热结束后，会达到的最大学习率。
        lr_end: float = 1e-6
            退火完成后，会达到的最终学习率。
        lr_warmup_steps: int = 1000
            预热步数。
        lr_cycle_steps: int = 0
            学习率循环周期步数。
        lr_cycle_decay: int = 0.4
            学习率周期衰减率。未使用。
        max_steps: int = 0
            最大步数。若为 0，则会自动替换为整体训练的步数。
    """

    lr_initial: float = 1e-6
    lr_max: float = 1e-4
    lr_end: float = 1e-6
    lr_warmup_steps: int = 1000
    lr_cycle_steps: int = 0
    lr_cycle_decay: int = 0.4  # TODO 未使用
    max_steps: int = 0  # 若为 0，会自动替换为整体训练的步数
    scheduler_type: Literal["epoch", "step"] = "step"


@dataclass
class TrainingConfig:
    """
    配置训练相关内容。

    Args:
        optimizer: Union[Literal["adamw", "apex_adamw", "bnb_adamw"], object] = "adamw"
            优化器类型，支持 "adamw", "apex_adamw", "bnb_adamw"，或未实例化的自定义优化器。
        optimizer_args: dict = {}
            优化器参数。由于有学习率调度器，`lr` 参数会默认未 1。
    """

    optimizer: Union[Literal["adamw", "apex_adamw", "bnb_adamw"], object] = "adamw"
    optimizer_args: dict = field(default_factory=dict)
    no_weight_decay_module_names: list[str] = field(default_factory=lambda: ["bias", "norm", "embed"])

    def __post_init__(self):
        default_args = {
            "lr": 1,  # 该参数应当固定为 1。真正 lr 由学习率调度器控制
            'betas': (0.9, 0.95),
            'weight_decay': 1e-2,
        }
        self.optimizer_args = {**default_args, **self.optimizer_args}
        if self.optimizer == "adamw":
            default_args["fused"] = True
            from torch.optim import AdamW

            self.optimizer = partial(AdamW, **self.optimizer_args)
        elif self.optimizer == "apex_adamw":
            from apex.optimizers import FusedAdam

            self.optimizer = partial(FusedAdam, adam_w_mode=True, **self.optimizer_args)
        elif self.optimizer == "bnb_adamw":
            from bitsandbytes.optim import AdamW8bit

            self.optimizer = partial(AdamW8bit, **self.optimizer_args)
        else:
            self.optimizer = partial(self.optimizer, **self.optimizer_args)
