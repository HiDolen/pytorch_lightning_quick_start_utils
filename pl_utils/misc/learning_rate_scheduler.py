from torch.optim import lr_scheduler
import numpy as np
import math
from pl_utils.configs import LearningRateConfig


class LinearWarmupCosineAnnealingLR:
    """
    线性预热、余弦退火 的学习率调度器。
    """

    def __init__(
        self,
        lr_config: LearningRateConfig,
        max_steps: int,
    ):
        # max_steps
        if lr_config.max_steps == 0:
            self.max_steps = max_steps
        elif isinstance(lr_config.max_steps, float):
            self.max_steps = int(lr_config.max_steps * max_steps)
        else:
            self.max_steps = lr_config.max_steps

        # lr_warmup_steps
        if isinstance(lr_config.lr_warmup_steps, float):
            self.lr_warmup_steps = int(lr_config.lr_warmup_steps * self.max_steps)
        else:
            self.lr_warmup_steps = lr_config.lr_warmup_steps

        # lr_cycle_steps
        if isinstance(lr_config.lr_cycle_steps, float):
            self.lr_cycle_steps = int(lr_config.lr_cycle_steps * self.max_steps)
        else:
            self.lr_cycle_steps = lr_config.lr_cycle_steps

        # lr_initial, lr_max, lr_end
        self.lr_initial = lr_config.lr_initial
        self.lr_max = lr_config.lr_max
        self.lr_end = lr_config.lr_end

        assert self._warmup(0) == self.lr_initial
        assert math.isclose(self._warmup(self.lr_warmup_steps), self.lr_max, rel_tol=1e-9)
        assert math.isclose(self._cosine(self.lr_warmup_steps), self.lr_max, rel_tol=1e-9)

    def _warmup(self, step):
        return self.lr_initial + (self.lr_max - self.lr_initial) * step / self.lr_warmup_steps

    def _cosine(self, step):
        cos = np.cos(
            np.pi * (step - self.lr_warmup_steps) / (self.max_steps - self.lr_warmup_steps - 1)
        )
        return self.lr_end + (self.lr_max - self.lr_end) * (1 + cos) / 2

    def _cycle(self, step):
        cos = np.cos(
            np.pi * ((step - self.max_steps) % self.lr_cycle_steps) / (self.lr_cycle_steps - 1)
        )
        return self.lr_end + (self.lr_max - self.lr_end) * (1 + cos) / 2

    def __call__(self, step):
        if step < self.lr_warmup_steps:
            return self._warmup(step)
        elif step < self.max_steps:
            return self._cosine(step)
        else:
            if self.lr_cycle_steps == 0:
                return self.lr_end
            else:
                return self._cycle(step)
