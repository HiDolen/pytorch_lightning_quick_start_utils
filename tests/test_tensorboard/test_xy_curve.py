"""XY Curves 插件测试:扫描、缓存与 _parse_value 解析容错。"""

import os
import shutil
import tempfile
import unittest

from torch.utils.tensorboard import SummaryWriter

import pl_utils  # noqa: F401
from cli.tensorboard_plugins.xy_curve import XyCurvePlugin


class _FakeContext:
    """只携带 logdir 的极简 context"""

    def __init__(self, logdir):
        self.logdir = logdir
        self.multiplexer = None


def _make_plugin(logdir):
    # 绕过 __init__，不需要真实 TensorBoard 上下文
    plugin = XyCurvePlugin.__new__(XyCurvePlugin)
    plugin._context = _FakeContext(logdir)
    plugin._cache_signature = None
    plugin._cache_result = None
    return plugin


class XyCurvePluginTest(unittest.TestCase):
    """针对 xy_curves 插件后端的端到端行为。"""

    def setUp(self):
        # 构成时间序列的两条切片
        self.logdir = tempfile.mkdtemp(prefix="xy_curve_test_")
        writer = SummaryWriter(os.path.join(self.logdir, "train"))
        writer.add_curve("curves/sine", [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], 0)
        writer.add_curve("curves/sine", [[0.0, 0.1], [0.5, 0.9], [1.0, 0.1]], 1)
        writer.close()

    def tearDown(self):
        shutil.rmtree(self.logdir, ignore_errors=True)

    def test_scan_finds_runs_tags_and_points(self):
        """扫描应按 (run, tag) 组织,同 tag 多 step 按时间排序。"""
        plugin = _make_plugin(self.logdir)
        scan = plugin._scan()
        self.assertIn("train", scan)
        self.assertIn("curves/sine", scan["train"])
        data = scan["train"]["curves/sine"]["data"]
        self.assertEqual(len(data), 2)
        self.assertEqual([d["step"] for d in data], [0, 1])
        self.assertEqual(data[0]["points"][1], [0.5, 1.0])

    def test_scan_caches_by_file_signature(self):
        """event 文件未变化时,两次扫描应命中缓存返回同一对象。"""
        plugin = _make_plugin(self.logdir)
        first = plugin._scan()
        second = plugin._scan()
        # 缓存命中表现为 identity 相同,而非重新构造
        self.assertIs(first, second)

    def test_is_active(self):
        """logdir 中存在本插件数据时插件应处于激活状态。"""
        plugin = _make_plugin(self.logdir)
        self.assertTrue(plugin.is_active())

    def test_parse_value_ignores_non_string_tensors(self):
        """数据约定是 DT_STRING(承载 JSON);其他 dtype 应被解析层丢弃。"""
        from tensorboard.compat.proto import summary_pb2, tensor_pb2, types_pb2

        class _Event:
            # _parse_value 只读 wall_time/step 两个字段
            wall_time = 123.0
            step = 7

        plugin = _make_plugin(self.logdir)
        value = summary_pb2.Summary.Value(
            tag="x",
            tensor=tensor_pb2.TensorProto(dtype=types_pb2.DT_FLOAT),
        )
        self.assertIsNone(plugin._parse_value(_Event(), value))


if __name__ == "__main__":
    unittest.main()
