import logging
import tempfile
import unittest

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader

from pl_utils import BaseModule, OneOptimizerConfig, TrainingConfig
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


class TestCheckpointResume(unittest.TestCase):
    def _get_tensorboard_scalar_step(self, log_dir, tag):
        """从 TensorBoard 日志中提取指定标量的最大 step"""
        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()
        return max(event.step for event in event_accumulator.Scalars(tag))

    def test_lightweight_checkpoint_resume_continues_tensorboard_log(self):
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

        with tempfile.TemporaryDirectory() as temp_dir:  # 创建临时目录，with 块结束后自动删除
            # 检查点回调
            checkpoint_callback = ModelCheckpoint(
                dirpath=temp_dir,
                every_n_epochs=1,
                save_weights_only=True,
                save_last=True,
            )
            # logger
            logger = TensorBoardLogger(
                save_dir=temp_dir,
                name="lightning_logs",
                version="test_checkpoint_resume",
            )
            # 首次训练
            pl_model = DummyModule(nn.LazyLinear(1), training_config)
            trainer = L.Trainer(
                max_epochs=2,
                logger=logger,
                enable_model_summary=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                callbacks=[checkpoint_callback],
                log_every_n_steps=1,
            )
            trainer.fit(pl_model, dummy_loader)
            logger.experiment.flush()

            # 加载最后一个检查点
            last_checkpoint = checkpoint_callback.last_model_path
            if not last_checkpoint:
                self.fail("No checkpoint was created.")
            # 获得 train/loss 的 step 数
            loss_step_before_resume = self._get_tensorboard_scalar_step(
                logger.log_dir, 'train/loss'
            )

            # 从检查点恢复训练
            pl_model = DummyModule(nn.Linear(10, 1), training_config)
            logger = TensorBoardLogger(
                save_dir=temp_dir,
                name="lightning_logs",
                version="test_checkpoint_resume",
            )
            trainer = L.Trainer(
                max_epochs=4,
                logger=logger,
                enable_model_summary=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
            trainer.fit(pl_model, dummy_loader, ckpt_path=last_checkpoint)
            logger.experiment.flush()

            # 验证恢复后 TensorBoard 日志按 checkpoint step 继续续写
            loss_step_after_resume = self._get_tensorboard_scalar_step(logger.log_dir, 'train/loss')
            # 检查 train/loss 的 step 数是否继续增长
            self.assertGreater(loss_step_after_resume, loss_step_before_resume)


if __name__ == '__main__':
    unittest.main()
