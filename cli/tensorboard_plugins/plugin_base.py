"""JSON tensor 插件通用后端基类。

数据约定：每个 (run, tag, step) 记录一个 DT_STRING tensor，内容为 JSON
序列化的列表（载荷语义由各插件前端解释，如曲线点列表）。基类经
TensorBoard DataProvider 提供 run 发现、/tags /data 路由、/ui/* 静态资源
服务，子类只需声明 plugin_name / tab_name / es_module_path（可选覆写
ui_root）。
"""

import json
import os
import urllib.parse

from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
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
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or {}).get(self.plugin_name, 100)

    def is_active(self):
        return False

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

    def _tags(self, ctx, experiment):
        series = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=self.plugin_name,
        )
        return {
            run: {
                tag: {
                    "displayName": metadata.display_name or None,
                    "description": metadata.description or None,
                }
                for tag, metadata in tags.items()
            }
            for run, tags in series.items()
        }

    def _data(self, ctx, experiment, run, tag):
        series = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        data = []
        for event in series.get(run, {}).get(tag, []):
            datum = self._parse_tensor_event(event)
            if datum is not None:
                data.append(datum)
        data.sort(key=lambda datum: (datum["step"], datum["wall_time"]))
        return data

    def _parse_tensor_event(self, event):
        try:
            payload = event.numpy.item()
        except ValueError:
            return None
        if not isinstance(payload, (bytes, str)):
            return None
        points = self._parse_json(payload)
        if points is None:
            return None
        return {
            "wall_time": event.wall_time,
            "step": event.step,
            "points": points,
        }

    def _parse_json(self, payload):
        try:
            points = json.loads(payload)
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
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self._tags(ctx, experiment)
        return http_util.Respond(request, json.dumps(index), "application/json")

    @wrappers.Request.application
    def _serve_data(self, request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        data = []
        if run is not None and tag is not None:
            ctx = plugin_util.context(request.environ)
            experiment = plugin_util.experiment_id(request.environ)
            data = self._data(ctx, experiment, run, tag)
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
