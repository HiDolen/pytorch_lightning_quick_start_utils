import unittest
import logging
import shutil
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pl_utils import TrainingConfig, BaseModule, OneOptimizerConfig
from pl_utils.misc import LinearWarmupCosineAnnealingLR

logging.getLogger("lightning").setLevel(logging.ERROR)


class DummyModule(BaseModule):
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers

        need_step = (batch_idx + 1) % self.training_config.accumulate_grad_batches == 0
        need_step = need_step or self.trainer.is_last_batch

        for optimizer in optimizers:
            with optimizer.toggle_model(sync_grad=need_step):
                y = self(batch)
                loss = y.mean()
                self.manual_backward(loss)
                if need_step:
                    self.log('train/loss', loss.item(), sync_dist=True)

        if need_step:
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            schedulers = self.lr_schedulers()
            schedulers = [schedulers] if not isinstance(schedulers, list) else schedulers
            for scheduler in schedulers:
                scheduler.step()


class TestTraningProcess(unittest.TestCase):
    def test_training_process(self):
        # 配置
        training_config = TrainingConfig(
            optimizers=[
                OneOptimizerConfig(
                    optimizer=torch.optim.AdamW,
                    scheduler=LinearWarmupCosineAnnealingLR(max_steps=10, lr_max=1e-4),
                )
            ]
        )
        # 模型与数据
        dummy_dataset = torch.randn(32, 10)
        dummy_loader = DataLoader(dummy_dataset, batch_size=16)
        dummy_model = nn.LazyLinear(1)
        pl_model = DummyModule(dummy_model, training_config)
        # 训练
        trainer = L.Trainer(
            max_epochs=3,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(pl_model, dummy_loader)

    def test_training_checkpoint(self):
        # 配置
        training_config = TrainingConfig(
            optimizers=[
                OneOptimizerConfig(
                    optimizer=torch.optim.AdamW,
                    scheduler=LinearWarmupCosineAnnealingLR(max_steps=10, lr_max=1e-4),
                )
            ]
        )
        # 模型与数据
        dummy_dataset = torch.randn(32, 10)
        dummy_loader = DataLoader(dummy_dataset, batch_size=16)
        dummy_model = nn.LazyLinear(1)
        pl_model = DummyModule(dummy_model, training_config)
        # 检查点回调
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            save_weights_only=False,
            save_last=True,
        )
        # logger
        logger = TensorBoardLogger(save_dir="./", version="test_training_checkpoint")
        # 首次训练
        trainer = L.Trainer(
            max_epochs=2,
            logger=logger,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(pl_model, dummy_loader)
        # 加载最后一个检查点
        last_checkpoint = checkpoint_callback.last_model_path
        if not last_checkpoint:
            self.fail("No checkpoint was created.")
        # 继续训练
        pl_model = DummyModule.load_from_checkpoint(last_checkpoint, model=dummy_model)
        trainer = L.Trainer(
            max_epochs=4,
            logger=logger,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(pl_model, dummy_loader)

    def tearDown(self):
        log_dir = "./lightning_logs"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)


if __name__ == '__main__':
    unittest.main()
