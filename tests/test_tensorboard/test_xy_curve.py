"""XY Curves 插件测试:DataProvider 读取与 JSON 解析容错。"""

import os
import shutil
import tempfile
import types
import unittest

import numpy as np
from tensorboard import context
from tensorboard.backend.event_processing import data_provider as event_data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from torch.utils.tensorboard import SummaryWriter

import pl_utils  # noqa: F401
from cli.tensorboard_plugins.xy_curve import XyCurvePlugin


class _FakeContext:
    """只携带插件初始化所需字段的极简 context。"""

    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.sampling_hints = {}


def _make_plugin(logdir):
    multiplexer = plugin_event_multiplexer.EventMultiplexer(
        tensor_size_guidance={"eq_curves": 100, "xy_curves": 100}
    )
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    data_provider = event_data_provider.MultiplexerDataProvider(multiplexer, logdir)
    return XyCurvePlugin(_FakeContext(data_provider))


class XyCurvePluginTest(unittest.TestCase):
    """针对 xy_curves 插件后端的端到端行为。"""

    def setUp(self):
        # 构成时间序列的两条切片
        self.logdir = tempfile.mkdtemp(prefix="xy_curve_test_")
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_curve("curves/sine", [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], 0)
        writer.add_curve("curves/sine", [[0.0, 0.1], [0.5, 0.9], [1.0, 0.1]], 1)
        writer.close()
        self.ctx = context.RequestContext()

    def tearDown(self):
        shutil.rmtree(self.logdir, ignore_errors=True)

    def test_provider_finds_runs_tags_and_points(self):
        """DataProvider 应按 (run, tag) 组织，同 tag 多 step 按时间排序。"""
        plugin = _make_plugin(self.logdir)
        tags = plugin._tags(self.ctx, "experiment")
        self.assertIn("train", tags)
        self.assertIn("curves/sine", tags["train"])
        data = plugin._data(self.ctx, "experiment", "train", "curves/sine")
        self.assertEqual(len(data), 2)
        self.assertEqual([d["step"] for d in data], [0, 1])
        self.assertEqual(data[0]["points"][1], [0.5, 1.0])

    def test_provider_lists_plugin(self):
        """插件激活状态由 TensorBoard 的 DataProvider 插件列表决定。"""
        plugin = _make_plugin(self.logdir)
        plugins = plugin._data_provider.list_plugins(self.ctx, experiment_id="experiment")
        self.assertIn(plugin.plugin_name, plugins)
        self.assertFalse(plugin.is_active())

    def test_parse_tensor_ignores_non_string_tensors(self):
        """数据约定是 DT_STRING(承载 JSON);其他 dtype 应被解析层丢弃。"""
        plugin = _make_plugin(self.logdir)
        event = types.SimpleNamespace(
            numpy=np.array(1.0),
            wall_time=0.0,
            step=0,
        )
        self.assertIsNone(plugin._parse_tensor_event(event))


if __name__ == "__main__":
    unittest.main()
