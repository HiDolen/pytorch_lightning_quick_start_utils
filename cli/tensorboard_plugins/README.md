# TensorBoard 曲线插件

该路径提供自定义 TensorBoard 插件：

- XY Curves，允许为每个 step 写入曲线
- EQ Curves，允许为每个 step 写入 EQ，x 轴有专门优化

装有 pl_utils 的环境中可以通过指令唤起带有插件的 Tensorboard：

```bash
pl log
```

## 使用方法

`import pl_utils` 或导入其任意子模块，`torch.utils.tensorboard.SummaryWriter`
就会自动获得插件方法。

### XY Curves

`add_curve()`。

```python
from pl_utils import BaseModule

class MyModule(BaseModule):
    def on_validation_epoch_end(self, batch, batch_idx):
        ...
        self.logger.experiment.add_curve(
            "xy/test", self.xy_points, self.global_step
        )
```

### EQ Curves

`add_eq_curve`。

```python
from pl_utils import BaseModule

class MyModule(BaseModule):
    def on_validation_epoch_end(self, batch, batch_idx):
        ...
        self.logger.experiment.add_eq_curve(
            "eq/test", self.eq_points, self.global_step
        )
```

## 关于 `cli/tensorboard_plugins/shared/histogram` 目录

`shared/histogram` 目录下是从 TensorBoard 移植来的共用 UI 库，一般来说不应被改动。
