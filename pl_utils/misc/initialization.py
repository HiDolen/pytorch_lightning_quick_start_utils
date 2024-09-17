import gc
from typing import Literal
import os
import pytorch_lightning as pl
import torch


def init_before_training(
    seed: int = 42,
    matmul_precision: Literal["highest", "high", "medium"] = "high",
    tokenizers_parallelism: bool = False,
):
    """
    清理缓存，设置随机种子，设置浮点数精度。
    """
    gc.collect()
    torch.cuda.empty_cache()
    pl.seed_everything(seed=seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)

    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"
