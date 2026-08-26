"""EQ Curves 插件测试:写入 proto 标记与后端 event 扫描。"""

import os
import shutil
import tempfile
import unittest

from tensorboard import context
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import data_provider as event_data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.compat.proto import summary_pb2
from torch.utils.tensorboard import SummaryWriter

import pl_utils  # noqa: F401
from cli.tensorboard_plugins.eq_curve import EQ_CURVE_PLUGIN_NAME, EqCurvePlugin


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
    return EqCurvePlugin(_FakeContext(data_provider))


class EqCurvePluginTest(unittest.TestCase):
    """针对 eq_curves 插件后端的端到端行为。"""

    def setUp(self):
        self.logdir = tempfile.mkdtemp(prefix="eq_curve_test_")
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_eq_curve("eq/response", [[20.0, 0.0], [1000.0, 3.0], [16000.0, 0.0]], 0)
        writer.close()
        self.ctx = context.RequestContext()

    def tearDown(self):
        shutil.rmtree(self.logdir, ignore_errors=True)

    def test_eq_curve_writes_eq_plugin_name(self):
        """写入的 summary 应携带 eq_curves 插件名(直接读 event 文件验证)。"""
        run_dir = os.path.join(self.logdir, "train")
        names = [n for n in os.listdir(run_dir) if "tfevents" in n]
        loader = event_file_loader.EventFileLoader(os.path.join(run_dir, names[0]))
        for event in loader.Load():
            for value in event.summary.value:
                if value.tag == "eq/response":
                    self.assertEqual(value.metadata.plugin_data.plugin_name, EQ_CURVE_PLUGIN_NAME)
                    self.assertEqual(value.metadata.data_class, summary_pb2.DATA_CLASS_TENSOR)
                    return
        self.fail("未找到 eq/response")

    def test_provider_finds_eq_data(self):
        """DataProvider 应按 (run, tag) 组织数据并原样保留曲线点。"""
        plugin = _make_plugin(self.logdir)
        tags = plugin._tags(self.ctx, "experiment")
        self.assertIn("train", tags)
        self.assertIn("eq/response", tags["train"])
        data = plugin._data(self.ctx, "experiment", "train", "eq/response")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["points"][1], [1000.0, 3.0])

    def test_provider_ignores_other_plugin_data(self):
        """xy_curves 的数据不应出现在 eq_curves 插件的索引中。"""
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_curve("curves/sine", [[0.0, 0.0], [1.0, 1.0]], 0)
        writer.close()
        plugin = _make_plugin(self.logdir)
        tags = plugin._tags(self.ctx, "experiment")
        self.assertIn("eq/response", tags["train"])
        self.assertNotIn("curves/sine", tags["train"])


if __name__ == "__main__":
    unittest.main()
