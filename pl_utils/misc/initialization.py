import gc
import pytorch_lightning as pl
import torch


def init_before_training(
    seed: int = 42,
    matmul_precision: str = 'medium',
):
    """
    清理缓存，设置随机种子，设置浮点数精度。
    """
    gc.collect()
    torch.cuda.empty_cache()
    pl.seed_everything(seed=seed, workers=True)
    torch.set_float32_matmul_precision(matmul_precision)
