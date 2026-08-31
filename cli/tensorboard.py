from __future__ import annotations

from tensorboard import default, program
from tensorboard.plugins.scalar import scalars_plugin


class _HiddenTabScalarsPlugin(scalars_plugin.ScalarsPlugin):
    """不激活插件的前端标签页，保留插件其他所有功能"""

    # 判活优先看数据交集（application.py:431），须先清空才会走到 is_active
    def data_plugin_names(self):
        return ()

    def is_active(self):
        return False


def run_tensorboard() -> int:
    command = [
        "pl log",
        "--logdir",
        "./",
        "--samples_per_plugin",
        "scalars=20000,images=200,eq_curves=500,xy_curves=500",
    ]
    # 只替换默认插件列表中的 ScalarsPlugin
    plugins = [
        _HiddenTabScalarsPlugin if p is scalars_plugin.ScalarsPlugin else p
        for p in default.get_plugins()
    ]
    tensorboard = program.TensorBoard(plugins=plugins)
    tensorboard.configure(command)
    return tensorboard.main()
