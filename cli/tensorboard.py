from __future__ import annotations

from tensorboard import default, program
from tensorboard.plugins.scalar import scalars_plugin

from .tensorboard_plugins.scalar_compare import ScalarComparePlugin


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
        ScalarComparePlugin if p is scalars_plugin.ScalarsPlugin else p
        for p in default.get_plugins()
    ]
    tensorboard = program.TensorBoard(plugins=plugins)
    tensorboard.configure(command)
    return tensorboard.main()
