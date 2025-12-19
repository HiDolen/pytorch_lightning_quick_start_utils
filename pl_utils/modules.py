import torch
import lightning as L
from IPython import get_ipython
import shutil
import math
import __main__
from contextlib import nullcontext

from lightning.pytorch.strategies import ParallelStrategy

from pl_utils.configs import TrainingConfig
from pl_utils.misc import LinearWarmupCosineAnnealingLR, LinearWarmupStepDecayLR


class BaseModule(L.LightningModule):

    def __init__(
        self,
        model,
        training_config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.training_config = training_config

        self.automatic_optimization = False

    def forward(self, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        optimizers = [optimizers] if not isinstance(optimizers, list) else optimizers

        need_step = (batch_idx + 1) % self.training_config.accumulate_grad_batches == 0
        need_step = need_step or self.trainer.is_last_batch

        ######## only for example purpose ########
        x = batch
        ######## only for example purpose ########

        context = nullcontext()
        if not need_step and isinstance(self.trainer.strategy, ParallelStrategy):
            context = self.trainer.strategy.block_backward_sync()
        with context:
            loss = self.model(x)
            self.manual_backward(loss)
            if need_step:
                self.log('train/loss', loss.item(), sync_dist=True)

        # 梯度和优化器
        if need_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.log('train/grad_norm', grad_norm.item(), sync_dist=True)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
        # 学习率调度器
        schedulers = self.lr_schedulers()
        schedulers = [schedulers] if not isinstance(schedulers, list) else schedulers
        for scheduler in schedulers:
            scheduler.step()

        raise NotImplementedError

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
        if self.trainer.max_epochs == -1:
            step_num = self.trainer.max_steps
        else:
            step_num = self.trainer.num_training_batches * self.trainer.max_epochs
        max_steps = math.ceil(step_num / self.trainer.accumulate_grad_batches)

        # 记录 lr_schedule
        try:
            for i, opt_config in enumerate(self.training_config.optimizers):
                scheduler_func = opt_config.scheduler
                lr_schedule = [scheduler_func(e) for e in range(self.global_step, max_steps)]

                # 记录 lr。筛选需要记录的 step
                steps_to_log = set()
                last_lr = None
                for step, lr in enumerate(lr_schedule):
                    # 每隔 log_every_n_steps 记录学习率
                    if step % self.trainer.log_every_n_steps != 0:
                        continue
                    if lr != last_lr:
                        if last_lr is not None:
                            steps_to_log.add(step - self.trainer.log_every_n_steps)  # 变化前
                        steps_to_log.add(step)  # 变化后
                        last_lr = lr
                steps_to_log.add(len(lr_schedule) - 1)  # 最后一个 step
                # 记录到 TensorBoard
                for step in sorted(steps_to_log):
                    if 0 <= step < len(lr_schedule):
                        self.logger.experiment.add_scalar(
                            f'lr/optimizer_{i}', lr_schedule[step], step
                        )
        except Exception as e:
            print("Record lr_schedule failed. But it's ok.")

    def configure_optimizers(self):
        remaining_params = dict(self.model.named_parameters())
        remaining_params = {n: p for n, p in remaining_params.items() if p.requires_grad}

        optimizers = []
        schedulers = []

        for i, opt_config in enumerate(self.training_config.optimizers):
            current_opt_params_dict = {}
            is_last = i == len(self.training_config.optimizers) - 1

            to_remove = []
            for name, param in remaining_params.items():
                # Muon 跳过 1D 参数
                if opt_config.optimizer == torch.optim.Muon and param.dim() < 2:
                    continue
                # 最后一个优化器取所有剩余参数，否则根据 keywords 匹配
                if is_last or any(kw in name for kw in opt_config.keywords):
                    current_opt_params_dict[name] = param
                    to_remove.append(name)
            for name in to_remove:
                del remaining_params[name]

            if not current_opt_params_dict:
                print(f"Warning: Optimizer has no parameters: {opt_config.optimizer}")
                continue

            # weight decay
            params_with_wd = []
            params_without_wd = []
            for name, param in current_opt_params_dict.items():
                if any(kw in name for kw in opt_config.excluded_from_weight_decay):
                    params_without_wd.append(param)
                else:
                    params_with_wd.append(param)

            # 实例化优化器
            optimizer_grouped_parameters = [
                {"params": params_with_wd},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]
            optimizer = opt_config.optimizer(
                optimizer_grouped_parameters, **opt_config.optimizer_args
            )
            optimizers.append(optimizer)

            # 实例化学习率调度器
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=opt_config.scheduler)
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            )

        # 警告：存在未被任何优化器使用的参数
        if remaining_params:
            print(
                f"Warning: {len(remaining_params)} parameters not optimized: {list(remaining_params.keys())}"
            )

        return optimizers, schedulers
