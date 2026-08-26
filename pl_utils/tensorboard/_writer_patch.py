"""导入即向 torch.utils.tensorboard.SummaryWriter 注入曲线写入方法。

monkey-patch 方案：把 add_curve / add_eq_curve 绑定为 SummaryWriter 的
方法（self 即 writer），使 Lightning 的 logger.experiment 可直接调用。
"""

import warnings

from torch.utils.tensorboard import SummaryWriter

from .eq_curve import add_eq_curve
from .xy_curve import add_curve

for _name, _func in (
    ("add_curve", add_curve),
    ("add_eq_curve", add_eq_curve),
):
    if hasattr(SummaryWriter, _name):
        warnings.warn(f"SummaryWriter 已存在属性 {_name!r}，已跳过 pl_utils 的注入。")
        continue
    setattr(SummaryWriter, _name, _func)
