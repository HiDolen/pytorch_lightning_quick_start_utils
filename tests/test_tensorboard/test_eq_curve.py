"""EQ Curves 插件测试:写入 proto 标记与后端 event 扫描。"""

import os
import shutil
import tempfile
import unittest

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import plugin_event_multiplexer
from torch.utils.tensorboard import SummaryWriter

import pl_utils  # noqa: F401
from cli.tensorboard_plugins.eq_curve import EQ_CURVE_PLUGIN_NAME, EqCurvePlugin


class _FakeContext:
    """只携带 logdir 的极简 context"""

    def __init__(self, logdir, multiplexer):
        self.logdir = logdir
        self.multiplexer = multiplexer


def _make_plugin(logdir):
    # 绕过 __init__，不需要真实 TensorBoard 上下文
    plugin = EqCurvePlugin.__new__(EqCurvePlugin)
    multiplexer = plugin_event_multiplexer.EventMultiplexer(
        tensor_size_guidance={"eq_curves": 100, "xy_curves": 100}
    )
    multiplexer.AddRunsFromDirectory(logdir)
    multiplexer.Reload()
    plugin._context = _FakeContext(logdir, multiplexer)
    plugin._cache_signature = None
    plugin._cache_result = None
    return plugin


class EqCurvePluginTest(unittest.TestCase):
    """针对 eq_curves 插件后端的端到端行为。"""

    def setUp(self):
        self.logdir = tempfile.mkdtemp(prefix="eq_curve_test_")
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_eq_curve("eq/response", [[20.0, 0.0], [1000.0, 3.0], [16000.0, 0.0]], 0)
        writer.close()

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
                    return
        self.fail("未找到 eq/response")

    def test_scan_finds_eq_data(self):
        """扫描应按 (run, tag) 组织数据并原样保留曲线点。"""
        plugin = _make_plugin(self.logdir)
        scan = plugin._scan()
        self.assertIn("train", scan)
        self.assertIn("eq/response", scan["train"])
        data = scan["train"]["eq/response"]["data"]
        self.assertEqual(len(data), 1)
        # 点内容应与写入时完全一致(浮点原样透传)
        self.assertEqual(data[0]["points"][1], [1000.0, 3.0])

    def test_scan_ignores_other_plugin_data(self):
        """xy_curves 的数据不应出现在 eq_curves 插件的扫描结果里。"""
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_curve("curves/sine", [[0.0, 0.0], [1.0, 1.0]], 0)
        writer.close()
        plugin = _make_plugin(self.logdir)
        scan = plugin._scan()
        self.assertIn("eq/response", scan["train"])
        self.assertNotIn("curves/sine", scan["train"])

    def test_is_active(self):
        """logdir 中存在本插件数据时插件应处于激活状态。"""
        plugin = _make_plugin(self.logdir)
        self.assertTrue(plugin.is_active())


if __name__ == "__main__":
    unittest.main()
