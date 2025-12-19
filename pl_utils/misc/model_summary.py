import torch
import torch.nn as nn
import math


def get_human_readable_count(number: int) -> str:
    """
    将整数缩写为 K, M, B, T 格式。
    例如: 1,234 -> '1.2 K', 2,000,000 -> '2.0 M'
    """
    if number == 0:
        return "0"

    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(math.floor(math.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(math.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1

    if index < 1 or number >= 100:
        return f"{int(number):,d}{labels[index]}".strip()

    return f"{number:,.1f}{labels[index]}".strip()


def get_model_parameters_count(model: nn.Module):
    """
    获取模型的参数统计信息
    """

    def is_uninitialized(p):
        # 检查是否为未初始化的参数（Lazy Modules）
        return isinstance(p, torch.nn.parameter.UninitializedParameter)

    # 计算总参数量
    total_params = sum(p.numel() if not is_uninitialized(p) else 0 for p in model.parameters())

    # 计算可训练参数量
    trainable_params = sum(
        p.numel() if not is_uninitialized(p) else 0 for p in model.parameters() if p.requires_grad
    )

    return {
        "total": total_params,
        "trainable": trainable_params,
        "total_readable": get_human_readable_count(total_params),
        "trainable_readable": get_human_readable_count(trainable_params),
    }
