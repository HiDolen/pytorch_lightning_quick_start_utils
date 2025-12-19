from torch.optim import lr_scheduler
import numpy as np
import math


class LinearWarmupCosineAnnealingLR:
    """
    线性预热、余弦退火 的学习率调度器。
    """

    def __init__(
        self,
        max_steps: int,
        lr_initial: float = 1e-6,
        lr_max: float = 1e-4,
        lr_end: float = 1e-6,
        lr_warmup_steps: int | float = 1000,
        lr_cycle_steps: int | float = 0,
    ):
        # max_steps
        self.max_steps = max_steps

        # lr_warmup_steps
        if isinstance(lr_warmup_steps, float):
            self.lr_warmup_steps = int(lr_warmup_steps * self.max_steps)
        else:
            self.lr_warmup_steps = lr_warmup_steps

        # lr_cycle_steps
        if isinstance(lr_cycle_steps, float):
            self.lr_cycle_steps = int(lr_cycle_steps * self.max_steps)
        else:
            self.lr_cycle_steps = lr_cycle_steps

        # lr_initial, lr_max, lr_end
        self.lr_initial = lr_initial
        self.lr_max = lr_max
        self.lr_end = lr_end

        if self.lr_warmup_steps == 0:
            pass
        else:
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


class LinearWarmupStepDecayLR:
    """
    线性预热、固定比例退火 的学习率调度器。

    每隔固定步数将学习率乘以衰减因子。
    """

    def __init__(
        self,
        max_steps: int,
        lr_initial: float = 1e-6,
        lr_max: float = 1e-4,
        lr_warmup_steps: int | float = 1000,
        decay_factor: float = 0.9,
        decay_steps: int = 20000,
    ):
        self.max_steps = max_steps

        # lr_warmup_steps
        if isinstance(lr_warmup_steps, float):
            self.lr_warmup_steps = int(lr_warmup_steps * self.max_steps)
        else:
            self.lr_warmup_steps = lr_warmup_steps

        self.lr_initial = lr_initial
        self.lr_max = lr_max
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps

        if self.lr_warmup_steps > 0:
            assert self._warmup(0) == self.lr_initial
            assert math.isclose(self._warmup(self.lr_warmup_steps), self.lr_max, rel_tol=1e-9)

    def _warmup(self, step):
        return self.lr_initial + (self.lr_max - self.lr_initial) * step / self.lr_warmup_steps

    def _step_decay(self, step):
        num_decays = (step - self.lr_warmup_steps) // self.decay_steps
        return self.lr_max * (self.decay_factor**num_decays)

    def __call__(self, step):
        if step < self.lr_warmup_steps:
            return self._warmup(step)
        else:
            return self._step_decay(step)
