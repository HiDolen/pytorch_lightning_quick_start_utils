import torch
import gc
import lightning as L
import torch.nn.functional as F
from IPython import get_ipython
import shutil
import math
import __main__

from pl_utils.configs import LearningRateConfig, TrainingConfig
from pl_utils.misc import LinearWarmupCosineAnnealingLR


class BaseModule(L.LightningModule):

    def __init__(
        self,
        model,
        lr_config: LearningRateConfig,
        training_config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr_config = lr_config
        self.lr_scheduler = None  # 延迟初始化。在 on_train_start() 中初始化
        self.training_config = training_config

    def _lr_scheduler(self, step):
        if self.lr_scheduler is None:
            return self.lr_config.lr_initial
        return self.lr_scheduler(step)

    def forward(self, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
        inputs, labels = batch
        logits = self(**inputs)["logits"]
        pred = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError

    def on_train_start(self) -> None:
        if self.global_step != 0:
            return

        # 将 ipynb 文件保存到 log 目录
        try:
            ip = get_ipython()
            if '__vsc_ipynb_file__' in ip.user_ns:
                this_file = ip.user_ns['__vsc_ipynb_file__']
                shutil.copy(this_file, self.logger.log_dir)
        except Exception as e:
            print("Copy ipynb file failed. But it's ok.")
        # 将 py 文件保存到 log 目录
        try:
            this_file = getattr(__main__, '__file__', None)
            shutil.copy(this_file, self.logger.log_dir)
        except Exception as e:
            print("Copy py file failed. But it's ok.")

        # 实例化 LRScheduler
        max_steps = math.ceil(
            self.trainer.num_training_batches
            * self.trainer.max_epochs
            / self.trainer.accumulate_grad_batches
        )
        self.lr_scheduler = LinearWarmupCosineAnnealingLR(self.lr_config, max_steps)

        # 记录 lr_schedule
        try:
            lr_schedule = [self.lr_scheduler(e) for e in range(self.global_step, max_steps)]
            for step, lr in enumerate(lr_schedule):
                if self.lr_config.scheduler_type == "step":  # step 模式下只记录部分步长
                    if step % self.trainer.log_every_n_steps != 0:
                        continue
                self.logger.experiment.add_scalar('lr', lr, step)
        except Exception as e:
            print("Record lr_schedule failed. But it's ok.")

    def configure_optimizers(self):
        optimizer = self.training_config.optimizer(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: self._lr_scheduler(e))
        scheduler = {
            "scheduler": scheduler,
            "interval": self.lr_config.scheduler_type,
            "frequency": 1,
        }
        return [optimizer], [scheduler]
