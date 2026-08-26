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
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import types_pb2
from tensorboard.plugins import base_plugin

_DEFAULT_UI_ROOT = os.path.dirname(os.path.abspath(__file__))  # 本包目录

_MIME_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
}


def _event_files(path):
    """列出目录下的 event 文件（兼容 tfevents.* 与 events.out.tfevents.*）。"""
    try:
        names = sorted(n for n in os.listdir(path) if "tfevents" in n)
    except OSError:
        return []
    return [os.path.join(path, n) for n in names]


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
        return bool(self._scan())

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

    # ------------------------------------------------------------------
    # 数据扫描
    # ------------------------------------------------------------------
    def _run_paths(self):
        """返回 {run: 目录}。优先使用 multiplexer 的 run 发现结果。"""
        multiplexer = getattr(self._context, "multiplexer", None)
        if multiplexer is not None:
            try:
                paths = multiplexer.RunPaths()
                if paths:
                    return dict(paths)
            except Exception:  # pragma: no cover - 依赖 TensorBoard 内部行为
                pass
        logdir = self._context.logdir
        runs = {}
        if logdir and os.path.isdir(logdir):
            for name in sorted(os.listdir(logdir)):
                child = os.path.join(logdir, name)
                if os.path.isdir(child):
                    runs[name] = child
            runs["."] = logdir
        return runs

    def _scan(self):
        """扫描全部 event 文件，返回 {run: {tag: entry}}。

        entry 含 displayName/description/data（按 step 排序的曲线列表）。
        """
        runs = self._run_paths()
        signature = []
        for run, path in sorted(runs.items()):
            for full in _event_files(path):
                try:
                    stat = os.stat(full)
                    signature.append(
                        (run, os.path.basename(full), stat.st_mtime_ns, stat.st_size)
                    )
                except OSError:
                    continue
        signature = tuple(signature)
        if signature == self._cache_signature:
            return self._cache_result

        result = {}
        for run, path in runs.items():
            tags = {}
            for full in _event_files(path):
                loader = event_file_loader.EventFileLoader(full)
                for event in loader.Load():
                    if not event.HasField("summary"):
                        continue
                    for value in event.summary.value:
                        entry = self._parse_value(event, value)
                        if entry is None:
                            continue
                        tag_entry = tags.setdefault(
                            value.tag,
                            {
                                "displayName": None,
                                "description": None,
                                "data": [],
                            },
                        )
                        if entry["displayName"]:
                            tag_entry["displayName"] = entry["displayName"]
                        if entry["description"]:
                            tag_entry["description"] = entry["description"]
                        tag_entry["data"].append(entry["datum"])
            for tag_entry in tags.values():
                tag_entry["data"].sort(key=lambda d: (d["step"], d["wall_time"]))
            if tags:
                result[run] = tags
        self._cache_signature = signature
        self._cache_result = result
        return result

    def _parse_value(self, event, value):
        if not value.HasField("tensor"):
            return None
        if value.tensor.dtype != types_pb2.DT_STRING:
            return None
        if not value.tensor.string_val:
            return None
        if value.HasField("metadata"):
            plugin_name = value.metadata.plugin_data.plugin_name
            if plugin_name and plugin_name != self.plugin_name:
                return None
        try:
            points = json.loads(value.tensor.string_val[0])
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(points, list):
            return None
        display_name = None
        description = None
        if value.HasField("metadata"):
            display_name = value.metadata.display_name or None
            description = value.metadata.summary_description or None
        return {
            "displayName": display_name,
            "description": description,
            "datum": {
                "wall_time": event.wall_time,
                "step": event.step,
                # JSON 载荷原样透传，语义由插件前端解释
                "points": points,
            },
        }

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
        content_type = _MIME_TYPES.get(
            os.path.splitext(path)[1], "application/octet-stream"
        )
        with open(path, "rb") as f:
            content = f.read()
        return http_util.Respond(request, content, content_type)
