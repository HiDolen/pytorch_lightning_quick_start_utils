import importlib
from functools import partial

from pl_utils.configs import TrainingConfig

OPTIMIZER_DICT = {
    'adamw': ('torch.optim', 'AdamW'),
    'apex_adamw': ('apex.optimizers', 'FusedAdam'),
    'bnb_adamw': ('bitsandbytes.optim', 'AdamW8bit'),
}


def _get_optimizer_class(name):
    """
    根据优化器名称动态导入并返回优化器类。
    Args:
        name (str): 优化器名称
    """
    if name.lower() not in OPTIMIZER_DICT:
        raise ValueError(f"Optimizer '{name}' not found in OPTIMIZER_DICT.")
    module_name, class_name = OPTIMIZER_DICT[name.lower()]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_optimizer_partial(config: TrainingConfig):
    """
    获取优化器的 partial 函数。拿到 partial 函数后传入 model.parameters() 就可以使用了。

    Example:
        >>> from pl_utils.configs import TrainingConfig
        >>> config = TrainingConfig(optimizer="adamw", optimizer_args={"betas": (0.9, 0.95)})
        >>> optimizer_partial = get_optimizer_partial(config)
        >>> optimizer = optimizer_partial(model.parameters())

    Args:
        config (TrainingConfig): 包含优化器配置的 TrainingConfig 对象。
    """
    default_args = {
        "lr": 1,  # 该参数应当固定为 1。真正 lr 由学习率调度器控制
        'betas': (0.9, 0.95),
        'weight_decay': 1e-2,
    }
    kwargs = {**default_args, **config.optimizer_args}
    opt = config.optimizer
    optimizer_class = _get_optimizer_class(opt) if isinstance(opt, str) else opt
    return partial(optimizer_class, **kwargs)
