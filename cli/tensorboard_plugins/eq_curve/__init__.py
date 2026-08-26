"""EQ Curves 插件：写入侧 summary 构造 + 后端 /tags /data /ui/* 路由。"""

import json

from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins import base_plugin

from ..plugin_base import JsonTensorPluginBase

EQ_CURVE_PLUGIN_NAME = "eq_curves"


def add_eq_curve(writer, tag, points, step):
    """向 SummaryWriter 写入一条 EQ 曲线（x 轴为频率 Hz）。"""
    tensor = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING)
    tensor.string_val.append(json.dumps(points, separators=(",", ":")).encode("utf-8"))
    metadata = summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(plugin_name=EQ_CURVE_PLUGIN_NAME),
        data_class=summary_pb2.DATA_CLASS_TENSOR,
    )
    summary_proto = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag, tensor=tensor, metadata=metadata)]
    )
    writer._get_file_writer().add_summary(summary_proto, step)


class EqCurvePlugin(JsonTensorPluginBase):
    """在 TensorBoard 中渲染每步 EQ 曲线（x 轴为频率）的插件。"""

    plugin_name = EQ_CURVE_PLUGIN_NAME
    tab_name = "EQ Curves"
    es_module_path = "/ui/eq_curve/entry.js"


class EqCurveLoader(base_plugin.TBLoader):
    def load(self, context):
        return EqCurvePlugin(context)
