"""JSON tensor 插件通用后端基类。

数据约定：每个 (run, tag, step) 记录一个 DT_STRING tensor，内容为 JSON
序列化的列表（载荷语义由各插件前端解释，如曲线点列表）。基类提供 run
发现、event 扫描与缓存、/tags /data 路由、/ui/* 静态资源服务，子类只需
声明 plugin_name / tab_name / es_module_path（可选覆写 ui_root）。
"""

import json
import os
import urllib.parse

from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins import base_plugin

_DEFAULT_UI_ROOT = os.path.dirname(os.path.abspath(__file__))  # 本包目录

_MIME_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
}


class JsonTensorPluginBase(base_plugin.TBPlugin):
    """按 (run, tag, step) 组织的 JSON tensor 数据插件基类。"""

    plugin_name = None  # 由子类声明
    tab_name = None  # 由子类声明
    es_module_path = None  # 由子类声明
    ui_root = _DEFAULT_UI_ROOT  # 静态资源根目录，子类可覆写

    def __init__(self, context):
        super().__init__(context)
        self._context = context
        self._cache_signature = None
        self._cache_result = None

    def is_active(self):
        return self.plugin_name in self._context.multiplexer.ActivePlugins()

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path=self.es_module_path,
            tab_name=self.tab_name,
            remove_dom=True,
        )

    def get_plugin_apps(self):
        return {
            "/tags": self._serve_tags,
            "/data": self._serve_data,
            "/ui/*": self._serve_static,
        }

    def _scan(self):
        """从 TensorBoard 原生 multiplexer 读取 {run: {tag: entry}}。

        entry 含 displayName/description/data（按 step 排序的曲线列表）。
        """
        multiplexer = self._context.multiplexer
        plugin_tags = multiplexer.PluginRunToTagToContent(self.plugin_name)
        snapshots = {}
        signature = []
        for run, tags in sorted(plugin_tags.items()):
            for tag in sorted(tags):
                metadata = multiplexer.SummaryMetadata(run, tag)
                events = multiplexer.Tensors(run, tag)
                snapshots[(run, tag)] = (metadata, events)
                signature.append(
                    (
                        run,
                        tag,
                        tuple((event.step, event.wall_time) for event in events),
                    )
                )
        signature = tuple(signature)
        if signature == self._cache_signature:
            return self._cache_result

        result = {}
        for (run, tag), (metadata, events) in snapshots.items():
            tag_entry = {
                "displayName": metadata.display_name or None,
                "description": metadata.summary_description or None,
                "data": [],
            }
            for event in events:
                datum = self._parse_tensor_event(event)
                if datum is not None:
                    tag_entry["data"].append(datum)
            tag_entry["data"].sort(key=lambda d: (d["step"], d["wall_time"]))
            if tag_entry["data"]:
                result.setdefault(run, {})[tag] = tag_entry
        self._cache_signature = signature
        self._cache_result = result
        return result

    def _parse_tensor_event(self, event):
        points = self._parse_tensor(event.tensor_proto)
        if points is None:
            return None
        return {
            "wall_time": event.wall_time,
            "step": event.step,
            "points": points,
        }

    def _parse_tensor(self, tensor):
        if tensor.dtype != types_pb2.DT_STRING:
            return None
        if not tensor.string_val:
            return None
        try:
            points = json.loads(tensor.string_val[0])
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(points, list):
            return None
        return points

    # ------------------------------------------------------------------
    # HTTP 路由
    # ------------------------------------------------------------------
    @wrappers.Request.application
    def _serve_tags(self, request):
        scan = self._scan()
        index = {}
        for run, tags in scan.items():
            index[run] = {
                tag: {
                    "displayName": entry["displayName"],
                    "description": entry["description"],
                }
                for tag, entry in tags.items()
            }
        return http_util.Respond(request, json.dumps(index), "application/json")

    @wrappers.Request.application
    def _serve_data(self, request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        data = []
        if run is not None and tag is not None:
            entry = self._scan().get(run, {}).get(tag)
            if entry:
                data = entry["data"]
        return http_util.Respond(request, json.dumps(data), "application/json")

    @wrappers.Request.application
    def _serve_static(self, request):
        # 前缀路由：/data/plugin/<name>/ui/<相对 ui_root 的路径>。
        # ui_root 默认是本包目录，故仅放行 _MIME_TYPES 白名单后缀。
        prefix = "/data/plugin/%s/ui/" % self.plugin_name
        rel = request.path[len(prefix) :] if request.path.startswith(prefix) else ""
        rel = urllib.parse.unquote(rel)
        ui_root = os.path.normpath(self.ui_root)
        full = os.path.normpath(os.path.join(ui_root, rel))
        if not full.startswith(ui_root + os.sep):
            return http_util.Respond(request, "not found", "text/plain", code=404)
        if os.path.splitext(full)[1] not in _MIME_TYPES:
            return http_util.Respond(request, "not found", "text/plain", code=404)
        if not os.path.isfile(full):
            return http_util.Respond(request, "not found", "text/plain", code=404)
        return self._send_file(request, full)

    def _send_file(self, request, path):
        content_type = _MIME_TYPES.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            content = f.read()
        return http_util.Respond(request, content, content_type)
