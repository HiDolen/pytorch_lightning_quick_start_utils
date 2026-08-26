"""导入即向 torch.utils.tensorboard.SummaryWriter 注入曲线写入方法。"""

import warnings

from torch.utils.tensorboard import SummaryWriter

from cli.tensorboard_plugins.eq_curve import add_eq_curve
from cli.tensorboard_plugins.xy_curve import add_curve

for _name, _func in (
    ("add_curve", add_curve),
    ("add_eq_curve", add_eq_curve),
):
    if hasattr(SummaryWriter, _name):
        warnings.warn(f"SummaryWriter 已存在属性 {_name!r}，已跳过 pl_utils 的注入。")
        continue
    setattr(SummaryWriter, _name, _func)
