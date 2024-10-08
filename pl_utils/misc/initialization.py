import gc
from typing import Literal
import os
import pytorch_lightning as pl
import torch


def init_before_training(
    seed: int = 42,
    matmul_precision: Literal["highest", "high", "medium"] = "high",
    empty_cache: bool = True,
    tokenizers_parallelism: bool = False,
):
    """
    清理缓存，设置随机种子，设置浮点数精度。

    Args:
        seed (int, optional): 随机种子
        matmul_precision (Literal["highest", "high", "medium"], optional): 矩阵乘法精度
        empty_cache (bool, optional): 是否清理缓存
        tokenizers_parallelism (bool, optional): 是否启用 tokenizers 并行，针对 `transformers` 库。设为 False 可避免死锁警告
    """
    if empty_cache:
        gc.collect()
        torch.cuda.empty_cache()
    pl.seed_everything(seed=seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"
